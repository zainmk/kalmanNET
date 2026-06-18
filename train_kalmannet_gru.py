#!/usr/bin/env python3
"""
KalmanNET GRU — offline training script.

Reads:  training_buffer.json   (produced by the app's calibration flight)
Writes: kalmannet_model.pt     (loaded automatically by the app on next start)

Speed strategy — fully vectorized, no Python step loop
------------------------------------------------------
Old approach: Python loop over T steps, 4 GRU calls per step     ~23 000 calls/epoch
New approach:
  1. Build (T,9) teacher-forced innovations in numpy              (no grad, fast)
  2. ONE GRU call: (T,1,9) -> (T,1,64)
  3. Get K for all sensors: 4 × Linear(T,64 -> 6×m)
  4. Vectorized parallel update: x_hat_t = F@x_true_{t-1}         (T steps at once)
                                          + sum_s K_s[t] @ innov_s (batched bmm)
  5. One backward pass through K values only (x_pred is constant)

Wall time: ~20-40 s for 500 epochs on CPU.

Training buffer assumption: all 4 sensors are active for every step
(the app gates collection on `not any(sensor_failed.values())`).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from kalmannet import (
    _GRUKalmanNET, MODEL_PATH,
    SENSOR_ORDER, SENSOR_DIMS, SENSOR_OFFSETS, TOTAL_MEAS_DIM, STATE_DIM,
)
from kalman_filter import KalmanFilter

BUFFER_PATH = Path(__file__).parent / "training_buffer.json"

EPOCHS    = 200
SUBSAMPLE = 10     # keep every 10th step — reduces GRU seq_len from ~5800 to ~580
LR        = 1e-3
LR_MIN    = 1e-4
GRAD_CLIP = 5.0


def _filter_matrices():
    kf   = KalmanFilter()
    F_np = kf.F.astype(np.float32)
    H_np = {s: kf.sensors[s]["H"].astype(np.float32) for s in SENSOR_ORDER}
    F    = torch.from_numpy(F_np)
    H    = {s: torch.from_numpy(H_np[s]) for s in SENSOR_ORDER}
    return F_np, H_np, F, H


def _run_epoch(net, x_true_np, z_np, F_np, H_np, F, H_mats, optimizer):
    """
    Fully vectorized epoch — no Python loop over T steps.

    x_true_np : (T, 6)
    z_np      : {sensor: (T, m)}  — all sensors present every step (training guarantee)

    Gradient path: loss -> x_final -> K_s -> GRU output -> GRU weights
    x_pred (F @ x_true_prev) is detached (constant), so the backward is O(T)
    with no sequential depth.
    """
    T = len(x_true_np)

    # ── Stage 1: teacher-forced innovations for GRU input (numpy) ─────────────
    iv_np = np.zeros((T, TOTAL_MEAS_DIM), dtype=np.float32)
    for s in SENSOR_ORDER:
        innov_np = z_np[s] - (H_np[s] @ x_true_np.T).T  # (T, m)
        o        = SENSOR_OFFSETS[s]
        iv_np[:, o:o + SENSOR_DIMS[s]] = innov_np

    # ── Stage 2: single GRU call over all T steps ─────────────────────────────
    iv_t       = torch.from_numpy(iv_np).unsqueeze(1)  # (T, 1, 9)
    gru_out, _ = net.gru(iv_t)                          # (T, 1, 64)
    gru_out    = gru_out.squeeze(1)                     # (T, 64)

    K_all = {
        s: 2.0 * torch.sigmoid(net.heads[s](gru_out)).view(T, STATE_DIM, SENSOR_DIMS[s])
        for s in SENSOR_ORDER
    }  # each (T, 6, m)

    # ── Stage 3: vectorized parallel update ───────────────────────────────────
    # Predicted state for steps 1..T using the PREVIOUS true state.
    # Using x_true as the "previous estimate" gives clean teacher-forced predictions.
    # x_pred has no gradient — it's a constant that K corrections are applied to.
    x_pred_np = (F_np @ x_true_np[:-1].T).T          # (T-1, 6)
    x_pred    = torch.from_numpy(x_pred_np)            # (T-1, 6), no grad

    x_final = x_pred
    for s in SENSOR_ORDER:
        # Innovation: z_s - H_s @ x_pred   (x_pred is constant, no grad)
        z_t    = torch.from_numpy(z_np[s][1:])          # (T-1, m)
        innov  = z_t - (H_mats[s] @ x_pred.T).T         # (T-1, m)
        K_s    = K_all[s][1:]                            # (T-1, 6, m)
        delta  = torch.bmm(K_s, innov.unsqueeze(2)).squeeze(2)  # (T-1, 6)
        x_final = x_final + delta                        # accumulate corrections

    # Loss: squared position error over all steps
    x_true_t = torch.from_numpy(x_true_np[1:, :3])     # (T-1, 3)
    loss = ((x_final[:, :3] - x_true_t) ** 2).sum()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
    optimizer.step()

    return float(np.sqrt(loss.item() / (T - 1)))


def train_from_buffer(buffer: list, progress_cb=None) -> bool:
    """
    Train the GRU KalmanNET on an in-memory buffer and save to MODEL_PATH.

    buffer      : list of dicts from sim.training_buffer
    progress_cb : optional callable(epoch: int, epochs: int, rms: float)
                  called after each epoch — use to stream progress to the UI.

    Returns True on success, False if the buffer is empty or training fails.
    """
    if not buffer:
        print("[KalmanNET] train_from_buffer: empty buffer, nothing to train on.")
        return False

    buffer = buffer[::SUBSAMPLE]
    T      = len(buffer)
    print(f"[KalmanNET] {T} training steps (every {SUBSAMPLE}th of {T * SUBSAMPLE} raw)")

    x_true_np = np.array([s["x_true"] for s in buffer], dtype=np.float32)
    z_np      = {
        s: np.array([step["measurements"][s] for step in buffer], dtype=np.float32)
        for s in SENSOR_ORDER
    }

    F_np, H_np, F, H_mats = _filter_matrices()
    net       = _GRUKalmanNET()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    print(f"[KalmanNET] Training {EPOCHS} epochs (lr {LR} -> {LR_MIN}) ...")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        net.train()
        rms = _run_epoch(net, x_true_np, z_np, F_np, H_np, F, H_mats, optimizer)
        scheduler.step()

        if progress_cb is not None:
            progress_cb(epoch, EPOCHS, rms)

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (EPOCHS - epoch)
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{EPOCHS}  RMS={rms:.3f} m  "
                  f"lr={lr_now:.1e}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    torch.save(net.state_dict(), MODEL_PATH)
    elapsed = time.time() - t0
    kb      = MODEL_PATH.stat().st_size / 1024
    print(f"[KalmanNET] Model saved -> {MODEL_PATH.name} ({kb:.1f} KB) in {elapsed:.1f}s")
    return True


def main():
    if not BUFFER_PATH.exists():
        print(f"ERROR: {BUFFER_PATH} not found.")
        print("Complete a training flight in the app first, then run this script.")
        sys.exit(1)

    with open(BUFFER_PATH) as f:
        buffer = json.load(f)
    print(f"Loaded {len(buffer)} steps from {BUFFER_PATH.name}")

    ok = train_from_buffer(buffer)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
