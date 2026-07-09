# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session logging (required)

Append User/Claude exchanges to `CLAUDE_CHAT_LOG.md` as the **last action of the reply** — not at end of session. **Substance only** (policy set 2026-07-09): log design questions, diagnoses, decisions, filter/ML theory, and implemented work — anything serving the project's purpose or its portfolio story. Do **not** log filler: how-to-run questions, generic tool explanations, or exchanges that only restate an earlier entry. Start a `## Session N — <topic>` header when a new session begins, then `**User:** …` / `**Claude:** …` pairs separated by `---`. Keep the substance of what was said; do not over-summarise. Append only — rewrite/curate solely on explicit user request (last curated 2026-07-09).

## Running the app

```bash
# Development (any OS)
pip install flask numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
python app.py
# Open http://localhost:5000
```

- **Do not** `pip install -r requirements.txt` on Windows/macOS — it pins a linux-only cp311 CPU torch wheel for deployment (Docker).
- On this machine there is a checked-out venv: `.venv/Scripts/python.exe app.py`.
- No build step. No test suite yet (planned — see `UPGRADE_PLAN.md`). Restart `app.py` after any Python change; the browser picks up static file changes on refresh.
- **Deployment must be a single process** — all state lives in module globals. Docker runs `gunicorn app:app -w 1 --threads 8 -t 300`. More than one worker = each user sees a different simulation. (The browser build in `web/` sidesteps all of this — static hosting, no server.)

## Architecture

The sim runs in a Python background thread at 20 Hz and pushes state to the browser over a single persistent SSE connection (`/stream`). The browser applies each JSON frame to a Three.js scene — no polling, no WebSocket.

**Thread model in `app.py`:**
- `_sim_loop` — daemon thread, calls `sim.step()` every 50 ms (5 ms during training fast-mode → 10× sim speed), stores result in `_state`.
- `_run_training` — daemon thread launched by `POST /train`: runs the scripted calibration flight (`_TRAIN_PHASES`, ~29 s wall-clock at 10× ≈ 5 800 steps), saves `training_buffer.json`, then calls `train_from_buffer()` **in-process** (imported from `train_kalmannet_gru.py`; PyTorch releases the GIL during compute so SSE stays responsive). On success it reloads weights via `sim.kn._load()`, then pauses and resets the sim.
- SSE generator — reads `_state` under `_state_lock`, merges in `_training_state` and `_paused`, yields JSON at 20 Hz.

Four shared-state globals, each with its own lock: `_state/_state_lock`, `_paused/_paused_lock`, `_training_state/_training_lock`, `_training_fast/_training_fast_lock`. `sim` is guarded by `sim_lock`. Never hold more than one lock at a time.

**Simulation layer (`drone_sim.py`):**
- `DroneSim.step()` is the hot path: true state = analytical helix + accumulated wind displacement (EMA, 3% decay/step); sensor noise scales with environment (`eff_std`, `baro_bias = -0.25·ΔT`); runs `kf.predict()` + per-sensor `kf.update()`, then `kn.step(z_values)` (one call, all sensors); returns a single dict consumed by the SSE stream.
- `_init_filters()` preserves a trained KN across resets — checks `hasattr(self, 'kn') and self.kn.trained` before creating a new instance.
- The training buffer collects `{measurements, x_true}` dicts only when `collecting_data=True` and all sensors are active.

**Filter layer:**
- `kalman_filter.py` — standard linear KF. State = `[x, y, z, vx, vy, vz]`. Sensor H matrices map subsets of this state to measurements. R fixed at calm-condition calibration values (the deliberate mismatch KalmanNET corrects). Process noise `Q` uses dt=0.05, `process_noise=1.5`.
- `kalmannet.py` — GRU KalmanNET (`_GRUKalmanNET`): shared GRU (input = 9-D concatenated innovation vector of all sensors, hidden = 64) + per-sensor linear heads that output K matrices via `2·sigmoid` → (0, 2). K is clipped element-wise to `≤ 2×|K_Riccati|` — this is the stability envelope; without it dense off-diagonal K blows up P. Falls back to standard KF math when `trained=False`. Model auto-loads from `kalmannet_model.pt` at construction. Uses Joseph-form covariance update. `_k_ratio` (‖K_learned col‖ / ‖K_Riccati col‖) is streamed for the UI.
- `train_kalmannet_gru.py` — trainer, importable (`train_from_buffer`) or standalone on `training_buffer.json`. Teacher-forced and fully vectorised: one GRU call over the whole (subsampled 10×) sequence per epoch; loss = squared position error vs analytic truth; 200 epochs, Adam + cosine LR; ~32 s CPU.

**Browser build (`web/`)** — a full client-side port (no Flask, no PyTorch; see
`BROWSER_PORT_PLAN.md`). `web/index.html` loads the **unmodified** shared frontend from
`../static/` plus `web/js/{mat,kf,kalmannet,sim}.js` (JS ports of the Python filter/sim
layer), `engine.js` (replicates `app.py`'s loop/routes; TRAIN flies the real calibration
flight, then loads pretrained `web/weights.json`) and `shim.js` (intercepts
`EventSource('/stream')` + the six fetch routes; overrides training tooltips for honest
labeling). Regenerate weights + reference vectors with `python export_web_weights.py`
after retraining; verify with `node web/test/parity.mjs`. Serve the repo root
(`python -m http.server`, open `/web/`); `vercel.json` redirects `/` there.

**Frontend (`static/`):**
- `app.js` — all rendering and UI logic. Three.js r128 from CDN (no bundler). SSE handler calls `applyState(s)` which updates drone meshes, trails, error line, uncertainty sphere, sensor readouts, and `updateKNPanel(s)`. Drone modes: white `kfDrone` when KN untrained; cyan `knDrone` + translucent `kfSilhouette` when trained.
- `tooltips.js` — tooltip HTML content keyed by element id, applied to `data-tooltip` attributes on DOMContentLoaded; `app.js` registers hover handlers for all `[data-tooltip]` elements. New tooltip targets need **both** an `id` + `data-tooltip=""` attribute in `index.html` and an entry in `tooltips.js`.
- Coordinate mapping: `s2t(pt) = [pt[0], pt[2], pt[1]]` (sim Y→Three.js Z, sim Z→Three.js Y).

## Key invariants

- `sim_lock` must be held for any read or write to `sim` outside `_sim_loop`. The training thread must NOT hold it during the CPU-bound `train_from_buffer()` call.
- The sensor noise formulas live in `drone_sim.py:step()` **and its port `web/js/sim.js`**, and their numeric values are quoted in prose in `static/tooltips.js` and `README.md` — changing one requires updating all four.
- The SSE state dict keys are consumed directly in `applyState` in `app.js`. Renaming a Python key requires updating all JS references **and `web/js/sim.js`, which produces the same dict for the browser build**.
- Filter math changes in `kalman_filter.py`/`kalmannet.py` must be mirrored in `web/js/kf.js`/`web/js/kalmannet.js`; rerun `python export_web_weights.py` and `node web/test/parity.mjs` after changing either side.
- `_TRAIN_PHASES` is triplicated: `app.py`, `TRAIN_PHASES` in `web/js/engine.js`, and `KN_PHASES` in `static/app.js` — keep all three in sync.
- `kn_k_ratio` in the SSE payload drives the K-GAIN RATIO grid in the KN panel (`renderKratio`). Format: `{gps: [f,f,f], imu: [f,f,f], baro: [f], mag: [f,f]}`.
- KalmanNET must always fall back to classical KF behaviour when untrained; never remove the `≤ 2×|K_Riccati|` clip.
- UI buttons are locked during training via `_LOCK_IDS` / `setTrainingLock()` in `app.js`. Keyboard shortcuts (Space, R) are also guarded by `isTraining`.
- Deployment: exactly one gunicorn worker (state is in-process).

## Planned work

`UPGRADE_PLAN.md` is the authoritative backlog: bug fixes, UX upgrades, tooltip corrections, README work, and edge-AI portfolio milestones. When the user asks to "work on the plan" (or references it), read that file fully, follow its Ground Rules section, execute tasks in order within the requested phase, and tick checkboxes as tasks complete.
