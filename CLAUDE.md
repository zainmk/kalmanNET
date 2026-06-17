# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```
pip install flask numpy
python app.py
# Open http://localhost:5000
```

No build step. No test suite. Hot-reloading: kill and restart `app.py` after any Python change; the browser picks up static file changes on next refresh.

## Architecture

The sim runs in a Python background thread at 20 Hz and pushes state to the browser over a single persistent SSE connection (`/stream`). The browser applies each JSON frame to a Three.js scene — no polling, no WebSocket.

**Thread model in `app.py`:**
- `_sim_loop` — daemon thread, calls `sim.step()` every 50 ms (or 5 ms in training fast-mode), stores result in `_state`
- `_run_training` — daemon thread launched by `POST /train`, controls the scripted calibration flight and then calls `sim.kn.train(buf)` (CPU-bound, outside `sim_lock`)
- SSE generator — reads `_state` under `_state_lock`, merges in `_training_state` and `_paused`, yields JSON

Three shared-state globals, each with its own lock: `_state/_state_lock`, `_paused/_paused_lock`, `_training_state/_training_lock`. Never hold more than one lock at a time.

**Simulation layer (`drone_sim.py`):**
- `DroneSim.step()` is the hot path: computes true state (analytical helix), generates sensor noise, runs `kf.predict()` + `kn.predict()`, then for each active sensor runs `kf.update()` + `kn.update()`, returns a single dict consumed by the SSE stream.
- `_init_filters()` preserves a trained KN across resets — checks `hasattr(self, 'kn') and self.kn.trained` before creating a new instance.
- The training buffer collects `{innovations, env}` dicts only when `collecting_data=True` and all sensors are active.

**Filter layer:**
- `kalman_filter.py` — standard linear KF. State = `[x, y, z, vx, vy, vz]`. Sensor H matrices map subsets of this state to measurements. Process noise `Q` uses dt=0.05, `process_noise=1.5`.
- `kalmannet.py` — R-adaptive wrapper around a second internal `KalmanFilter`. Each sensor has an independent `_MLP` (2-layer, 16 hidden, numpy only) that maps a 20-step innovation window → `log(R_diagonal)`. `_effective_R()` clips to `[0.8×calib, 500×calib]`. Training uses oracle R targets from the known simulation noise formulas. The `trained` flag gates whether the MLP is used; `False` falls through to calibration R.

**Frontend (`static/`):**
- `app.js` — all rendering and UI logic. Three.js r128 loaded from CDN (no bundler). SSE handler calls `applyState(s)` which updates drone mesh positions, trails, error bars, and calls `updateKNPanel(s)`.
- `tooltips.js` — badge/button tooltip content (HTML strings), registered via a `data-tip` attribute scan.
- The coordinate mapping is `s2t(pt) = [pt[0], pt[2], pt[1]]` (sim Y→Three.js Z, sim Z→Three.js Y).

## Key invariants

- `sim_lock` must be held for any read or write to `sim` outside `_sim_loop`. The training thread releases it before the CPU-bound `sim.kn.train()` call.
- Sensor noise and baro bias formulas live in `drone_sim.py:step()` **and** are duplicated in `kalmannet.py:_oracle_R()`. They must stay in sync.
- The SSE state dict keys are consumed directly in `applyState` in `app.js`. Renaming a Python key requires updating all JS references.
- `kn_r_hat` in the SSE payload drives the R̂ grid in the KN panel (`renderRhat` in `app.js`). Format: `{gps: [f,f,f], imu: [f,f,f], baro: [f], mag: [f,f]}`.
- UI buttons are locked during training via `_LOCK_IDS` / `setTrainingLock()` in `app.js`. Keyboard shortcuts (Space, R) are also guarded by `isTraining`.

## Pending architectural upgrade

The project is mid-way through replacing the numpy MLP R-prediction with a full PyTorch GRU that predicts K directly (the original paper's architecture). Key decisions already confirmed:
- Separate training script (`train_kalmannet_gru.py`), not inline in `app.py`
- Shared GRU (hidden=64, input=9) + per-sensor output heads for K matrices
- K-gain ratio display replaces R̂ grid in the KN panel
- Wind displacement restored to the true trajectory (GRU can increase K for GPS to track it)
- `training_buffer.json` saved after calibration flight; training script runs via `subprocess` from `_run_training`
