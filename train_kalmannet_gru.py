#!/usr/bin/env python3
"""
KalmanNET GRU — offline training script.

Reads:  training_buffer.json   (produced by the app's calibration flight)
Writes: kalmannet_model.pt     (loaded automatically by the app on next start)

Can also be run standalone to retrain with different hyperparameters:
    python train_kalmannet_gru.py
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

EPOCHS    = 1000
CHUNK     = 100      # Truncated BPTT window (steps per gradient update)
LR        = 1e-3
LR_MIN    = 1e-4
GRAD_CLIP = 5.0


def _filter_matrices():
    kf = KalmanFilter()
    F  = torch.tensor(kf.F, dtype=torch.float32)
    H  = {s: torch.tensor(kf.sensors[s]["H"], dtype=torch.float32) for s in SENSOR_ORDER}
    return F, H


def _run_epoch(net, buffer, F, H_mats, optimizer):
    """
    One full pass over the trajectory with truncated BPTT.

    Within each CHUNK-step window:
      1. Predict:  x̂ = F · x̂_prev
      2. Per sensor: innovation = z − H · x̂; GRU step → K; x̂ += K · innov
      3. Loss: squared position error per step

    At chunk boundaries: detach x̂ and GRU hidden state h to prevent
    gradients exploding across hundreds of steps.
    """
    T     = len(buffer)
    h     = None
    x_hat = torch.tensor(buffer[0]["x_true"], dtype=torch.float32)
    total = 0.0

    for cs in range(0, T, CHUNK):
        ce    = min(cs + CHUNK, T)
        chunk = buffer[cs:ce]

        if h is not None:
            h = h.detach()
        x_hat = x_hat.detach()

        step_losses = []

        for step in chunk:
            x_hat = F @ x_hat                      # constant-velocity prediction

            for s in SENSOR_ORDER:
                z_raw = step["measurements"].get(s)
                if z_raw is None:
                    continue

                z     = torch.tensor(z_raw, dtype=torch.float32)
                innov = z - H_mats[s] @ x_hat      # (m,)

                # 9D vector: this sensor's innovation, others zero
                iv9   = torch.zeros(TOTAL_MEAS_DIM)
                o     = SENSOR_OFFSETS[s]
                # Detach for GRU input: stabilises training by breaking the
                # feedback path x̂ → innov → GRU while keeping the gradient
                # path K → x̂ (the K application below) intact.
                iv9[o:o + SENSOR_DIMS[s]] = innov.detach()

                x_in       = iv9.unsqueeze(0).unsqueeze(0)   # (1, 1, 9)
                K_out, h   = net(x_in, h)
                K          = K_out[s].squeeze(0)             # (STATE_DIM, m)
                x_hat      = x_hat + K @ innov

            x_true = torch.tensor(step["x_true"][:3], dtype=torch.float32)
            step_losses.append(((x_hat[:3] - x_true) ** 2).sum())

        if step_losses:
            loss = sum(step_losses)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
            x_hat = x_hat.detach()

    return float(np.sqrt(total / T))    # RMS position error in metres


def main():
    if not BUFFER_PATH.exists():
        print(f"ERROR: {BUFFER_PATH} not found.")
        print("Complete a training flight in the app first, then run this script.")
        sys.exit(1)

    print(f"Loading {BUFFER_PATH.name} ...", end=" ", flush=True)
    with open(BUFFER_PATH) as f:
        buffer = json.load(f)
    print(f"{len(buffer)} steps.")

    F, H_mats = _filter_matrices()
    net       = _GRUKalmanNET()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    print(f"Training {EPOCHS} epochs (TBPTT-{CHUNK}, lr {LR}→{LR_MIN}) ...\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        net.train()
        rms = _run_epoch(net, buffer, F, H_mats, optimizer)
        scheduler.step()

        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (EPOCHS - epoch)
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{EPOCHS}  RMS={rms:.3f} m  "
                  f"lr={lr_now:.1e}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    torch.save(net.state_dict(), MODEL_PATH)
    kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nModel saved → {MODEL_PATH.name}  ({kb:.1f} KB)")
    print(f"Training complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
