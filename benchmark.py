#!/usr/bin/env python3
"""
benchmark.py — measure Kalman filter vs Kalman-NET position RMSE across scenarios.

Requires a trained model (kalmannet_model.pt). Train first by clicking
TRAIN KALMAN-NET in the app, or run train_kalmannet_gru.py on a saved buffer.

For each scenario: fresh DroneSim, fixed seed, set the environment, run a settle
period, then measure per-step position error for both filters. Reports RMS of
each and the KN improvement over the KF, as a GitHub-markdown table.

No Flask, no new dependencies — DroneSim is driven directly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from drone_sim import DroneSim
from kalmannet import MODEL_PATH

try:
    import torch
except ImportError:
    torch = None

SETTLE_STEPS   = 200    # let both filters converge before measuring
MEASURE_STEPS  = 1200   # 60 sim-seconds at dt=0.05

# (label, wind_speed m/s, wind_heading rad, temperature °C)
SCENARIOS = [
    ("calm",     0.0,  0.0, 20.0),
    ("wind",    15.0,  0.0, 20.0),
    ("heat",     0.0,  0.0, 50.0),
    ("combined", 12.0, 0.0, 40.0),
]


def run_scenario(wind_speed, wind_heading, temperature):
    """Return (kf_rmse, kn_rmse) position error over the measured window."""
    np.random.seed(42)
    if torch is not None:
        torch.manual_seed(42)

    sim = DroneSim()
    sim.set_environment(wind_speed=wind_speed,
                        wind_heading=wind_heading,
                        temperature=temperature)

    for _ in range(SETTLE_STEPS):
        sim.step()

    kf_sq = 0.0
    kn_sq = 0.0
    for _ in range(MEASURE_STEPS):
        st = sim.step()
        kf_sq += st["error"] ** 2
        kn_sq += st["kn_error"] ** 2

    kf_rmse = float(np.sqrt(kf_sq / MEASURE_STEPS))
    kn_rmse = float(np.sqrt(kn_sq / MEASURE_STEPS))
    return kf_rmse, kn_rmse


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH.name} not found — no trained model to benchmark.")
        print("Train first: click TRAIN KALMAN-NET in the app, or run "
              "train_kalmannet_gru.py on a saved training_buffer.json.")
        sys.exit(1)

    print(f"Benchmarking {MODEL_PATH.name}  "
          f"({SETTLE_STEPS} settle + {MEASURE_STEPS} measured steps per scenario)\n")

    rows = []
    for label, wind, heading, temp in SCENARIOS:
        kf_rmse, kn_rmse = run_scenario(wind, heading, temp)
        improvement = (kf_rmse - kn_rmse) / kf_rmse * 100.0 if kf_rmse > 0 else 0.0
        rows.append((label, kf_rmse, kn_rmse, improvement))
        print(f"  {label:9s}  KF={kf_rmse:6.3f} m  KN={kn_rmse:6.3f} m  "
              f"improvement={improvement:+6.1f}%")

    print("\n" + "-" * 60 + "\n")
    print("| Scenario | KF RMSE (m) | KN RMSE (m) | Improvement |")
    print("|---|---|---|---|")
    for label, kf_rmse, kn_rmse, improvement in rows:
        print(f"| {label} | {kf_rmse:.3f} | {kn_rmse:.3f} | {improvement:+.1f}% |")


if __name__ == "__main__":
    main()
