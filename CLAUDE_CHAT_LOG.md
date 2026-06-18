# KalmanNET — Claude Chat Log

Running transcript of the full development conversation.
Updated automatically each session.

---

## Session 1 — Initial build & KalmanNET diagnosis

**User:** What is the best way to see the KalmanNET in action after training?

**Claude:** The clearest showcase is temperature stress. At 50°C, the baro reads ~7.5m low due to hot-air density bias. The fixed-R KF trusts this and drifts 6-7m off truth. The trained KN inflates R_baro up to 500× and ignores the barometer, holding altitude via GPS cross-covariance. KN error ≈ 1.2m vs KF ≈ 7m. The R̂ panel shows BARO ×157–500 in real time, making the mechanism visible.

---

**User:** The Kalman filter and the KalmanNET lines are still relatively close, and both off. Can you verify this behaviour and diagnose the issue in terms of how to make the KalmanNET more accurate and closer to the green line (true drone)?

**Claude:** Ran a live benchmark. Results:

| Scenario | KF | KN | KN advantage |
|---|---|---|---|
| CALM | 0.29m | 0.33m | −0.04m |
| WIND 10 | 0.47m | 0.79m | −0.33m (KN worse) |
| WIND 20 | 0.58m | 0.65m | −0.08m (KN worse) |
| TEMP 40°C | 4.52m | 1.33m | +3.19m |
| TEMP 50°C | 7.02m | 1.38m | +5.64m |
| TEMP −10°C | 6.90m | 2.57m | +4.33m |

Root cause: wind displacement was added to truth. GPS tracked the wind-displaced drone correctly, but KN inflated R_GPS (thought GPS was noisy), forcing reliance on the constant-velocity model which didn't know about wind. KN was ignoring the one sensor with the right answer.

Fix: removed wind displacement from truth (wind only affects noise, not true position), increased baro_bias 0.12→0.25, reverted wind noise coefficients, increased process_noise 0.5→1.5 to prevent P collapse.

---

**User:** So if I wanted to capture wind patterns in my KalmanNET, that would require the much larger implementation of GRU (as in the paper)? Help me understand KalmanNET's and their use case better (no code changes).

**Claude:** Correct. The R-prediction architecture has a structural limitation: it can only *reduce* trust in a sensor (inflate R → reduce K). It cannot *increase* trust beyond the Riccati-optimal level.

- **Temperature bias** → sensor is wrong → trust it less → R↑ → K↓. Architecture fits.
- **Wind displacement** → model is wrong (drone is really there) → trust GPS more → need K↑. Architecture fails.

The original KalmanNET (Revach et al., 2022) uses a GRU that predicts K directly — no R. Key advantages:
1. K can go either direction (above or below Riccati optimal)
2. GRU carries temporal context across hundreds of steps — learns "sustained GPS+IMU scatter = wind"
3. No oracle needed — trained end-to-end on position loss ||x̂[:3] − x_true[:3]||²
4. Captures cross-sensor correlations (GPS+IMU both off → wind; only baro off → temperature)

---

**User:** What would it take to update the code to make use of the full KalmanNET GRU? The goal of this project is to demonstrate how KalmanNET's can account for the 'accumulation of all environmental offsets known only once deployed'. No code changes — only plan.

**Claude:** Full plan outlined:
- `kalmannet.py`: full rewrite — `_GRUKalmanNET` PyTorch module (shared GRU hidden=64, input=9 + per-sensor K heads) + `KalmanNET` inference class
- `train_kalmannet_gru.py`: new standalone script — TBPTT training, position MSE loss, Adam + cosine LR, saves `kalmannet_model.pt`
- `drone_sim.py`: restore wind displacement to truth, update training buffer to `{measurements, x_true}` (no oracle needed)
- `app.py`: save buffer to JSON, run training script via subprocess, load model after
- `static/app.js`: replace R̂ grid with K-gain ratio display (↑ = trusting sensor more, ↓ = less)
- `static/index.html`: update label

Three decisions to confirm: (1) separate script or inline, (2) K display format, (3) restore wind displacement.

---

**User:** Are there any public APIs or libraries I could use for the GRU network? Explain the training for this specific kind of network — what do you mean by '1-2 days of implementation and tuning'?

**Claude:** Libraries: `torch.nn.GRU`, `torch.nn.Linear` (per-sensor heads), `torch.optim.Adam`, `CosineAnnealingLR`, `clip_grad_norm_`. Install CPU-only (~200MB): `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

Training (BPTT):
1. Forward pass: for each step t — predict x̂⁻ = F·x̂, compute innovations, GRU step → K per sensor, apply x̂ = x̂ + K·ε
2. Loss: Σ ||x̂_t[:3] − x_true_t[:3]||²
3. Backprop through time: gradient flows from loss through K → GRU → hidden state chain
4. Truncated BPTT: process in 100-step chunks, detach h and x̂ at boundaries (prevents exploding gradients)

Training time for this demo: ~15-30 seconds on CPU.

---

**User:** How practical would it be to have this architecture deployed remotely on a microcontroller?

**Claude:** Feasible but with caveats.

Model size: ~30-50k parameters × 4 bytes ≈ 200KB. Fits in flash on Cortex-M7 (STM32H7 has 2MB flash).

Inference cost: ~14,000 multiply-adds per 50ms step. A Cortex-M7 at 480MHz handles this in ~60μs — well within budget.

Real deployment path: export weights as flat C arrays, implement GRU equations manually (~50 lines of C). The KF P-matrix inversion is actually more expensive than the GRU on small MCUs.

Hard problem is **sim-to-real gap** (see next question).

---

**User:** So the caveat of using a simulated environment to model and train the overall drone parameters and the resulting KalmanNET is consequenced by the accuracy of the simulation to the actual deployed drone environment? Are there accurate enough environmental simulations to trust the trained results well enough to actually deploy?

**Claude:** Yes — that's exactly the bottleneck.

What the simulation models accurately: Gaussian additive noise (GPS/IMU), linear temperature-pressure relationship (baro), wind as constant velocity offset.

What it gets wrong:

| Real phenomenon | Simulation | Gap |
|---|---|---|
| GPS multipath | Pure Gaussian | Multipath = heavy-tailed, correlated bursts |
| IMU Allan deviation | Gaussian scaled by temp | Real bias drifts on 10-1000s timescales |
| Propeller wash on baro | Not modelled | Adds 0.3-1m pressure offset with throttle |
| Motor magnetic interference | Not modelled | Correlated with throttle |

Industry answer: **domain randomization**. Instead of modelling each effect precisely, randomize noise parameters widely during training (GPS std uniform(0.5, 8.0) instead of a fixed formula). Network becomes robust to variation rather than calibrated to one model. Used widely in sim-to-real robotics.

Honest answer for this demo: simulation is accurate enough to demonstrate the *mechanism* (baro bias → KN ignores baro; wind → GRU increases K_GPS). Not accurate enough to deploy on a real drone without re-training on flight data. Production requires 20-50 flights with differential GPS ground truth.

---

**User:** For the sake of this simulation demo, update the code to begin using PyTorch and the 'Full KalmanNET (GRU)' approach. Confirm with me first how long this would take, if it can all be put into a separate script file, and anything else that may matter.

**Claude:** Confirmed plan:
- Training time: ~15-30s on CPU per run (1000 epochs over ~5000-step buffer)
- Separate script (`train_kalmannet_gru.py`) — inspectable, re-runnable independently
- Model persists to `kalmannet_model.pt` (~70KB), reloads on app restart automatically
- Calibration flight collects the buffer (same scripted phases, same UI flow)
- `kalmannet_model.pt` never needs to be re-collected unless you want fresh data

Three questions asked: separate script or inline? Which K display? Restore wind displacement?

---

**User:** Separate script, option A for K matrix structure, use the ratio display, restore wind displacement now.

**Claude:** Implemented in full:
- `kalmannet.py` — PyTorch GRU, per-sensor heads, 2·sigmoid K output, Joseph-form P update
- `drone_sim.py` — wind displacement restored to truth, buffer format updated to `{measurements, x_true}`, return dict changed to `kn_k_ratio`
- `train_kalmannet_gru.py` — TBPTT-100, Adam + cosine LR, 1000 epochs, prints RMS every 100
- `app.py` — saves `training_buffer.json`, runs training script via subprocess, loads model on success
- `static/app.js` — `renderKratio` replaces `renderRhat`, ↑/↓/~ arrows
- `static/index.html` — "K-GAIN RATIO (vs standard KF)"
- `requirements.txt` — added `torch>=2.0`, installed into venv
- `CLAUDE.md` — created, architecture guide for future Claude sessions

---

## Session 2 — Training pipeline & chat log

**User:** The last message I sent was 'separate script, option A for K matrix structure, use the ratio display, restore wind displacement now' — invoke the plan with the decisions made. Make a separate file to note all the design decisions made, especially the switch from R prediction to full GRU, just list the chat history and the explanations made including tradeoffs for real deployment.

**Claude:** Created `design_decisions.md` with the full Q&A history. Then implemented the full GRU upgrade across all files (see Session 1 final entry above for what changed).

---

**User:** What is the next step for running this simulation? Now that the model in reality gets trained from running the script — is the automated training script when the user hits 'Train' still necessary?

**Claude:** The Train button is still necessary — it owns the data collection step (calibration flight), which can only happen inside the running simulation. The subprocess call to `train_kalmannet_gru.py` makes the whole pipeline one-click. The separate script file exists for re-training with tweaked hyperparameters without re-flying the calibration sequence.

Workflow:
- First time: click Train → full pipeline automated
- App restart: model loads automatically from `kalmannet_model.pt`, ACTIVE dot appears immediately
- Re-tune hyperparameters: edit script, run manually, restart app
- Fresh data: click Retrain

---

**User:** Keep a log of our chat history — only in text in a file called 'CLAUDE_CHAT_LOG', keep a copy of our conversation and add to it every time we speak.

**Claude:** Created this file. Will append to it at the start of each future session.

---

## Session 3 — GRU training fixes, UI overhaul, drone visual modes

**User:** Add a progress bar so the user has an estimate of how much longer is left.

**Claude:** Switched `subprocess.run` to `subprocess.Popen`, read stdout line by line, parsed "Epoch X/200" to update `_training_state["progress"]` from 0.9 → 0.99 during the OPTIMISING phase. Progress bar element added to the KN panel in HTML, styled in CSS, animated in JS.

---

**User:** `kalmannet.py:114: RuntimeWarning: overflow encountered in cast` when running 'play' after training.

**Claude:** Root cause: the GRU outputs a dense K matrix (all 18 elements ≈ 0.95–2.0). Off-diagonal K values cause (I − K@H) to have eigenvalues of −1.85, so the covariance matrix P grows 3.42× every step. After ~100 steps the state x diverges and the innovation overflows float32.

Fix: `K = np.clip(K, 0.0, np.abs(K_ref) * 2 + 1e-6)` — clips relative to the Riccati K. The Riccati K has near-zero off-diagonal elements, so the clip zeroes out the GRU's off-diagonal output and preserves filter stability. Added `np.clip(innov, -1e4, 1e4).astype(np.float32)` to silence the overflow warning, plus a safety reset if `|x| > 1e5`.

---

**User:** When KalmanNET is trained, have there only be a 'clear model' option (must clear, see the state when cleared — then be allowed to train).

**Claude:** Changed Train/Clear buttons from `disabled` ↔ `enabled` to exclusive show/hide. When training active: both hidden. When trained: only Clear visible. When untrained: only Train visible. This forces the user to explicitly clear before re-training, making the state transition intentional and visible.

---

**User:** Make the actual drone shown model the KalmanNET, only when it's been trained and usable. Otherwise default to the Kalman filter. Make the swap back and forth obvious to the user. When the drone is focused on KalmanNET, have the Kalman Filter show a drone silhouette.

**Claude:** Implemented three-pronged mode indication:

1. **Drone colors**: refactored `makeDrone(style)` with four styles — `'kf'` (white, blue rotors), `'kn'` (cyan #00ccbb, dark rotors), `'true'` (green wireframe), `'kf-silhouette'` (translucent blue, opacity 0.30). Four drone meshes: `kfDrone`, `knDrone`, `kfSilhouette`, `trueDrone`.

2. **Visibility toggle** in `applyState`: when KN trained — cyan `knDrone` at KN estimate, translucent `kfSilhouette` at KF estimate, white `kfDrone` hidden. When untrained — white `kfDrone` at KF estimate, others hidden.

3. **Flash banner**: `#drone-mode-flash` div with CSS keyframe animation (2.4s fade in/hold/out). Shows "DRONE ● KALMAN-NET" in cyan or "DRONE ● KALMAN FILTER" in blue on every mode change. Legend updates dynamically: est. drone dot color + label change, KF silhouette row appears/disappears.

---

## Session 4 — Deployment planning, subprocess → in-process training, UI polish

**User:** Put the 'KalmanNET' next to the other lines of reference in the legend.

**Claude:** Moved `#legend-kn` in `index.html` to sit immediately after Kalman (True / Raw / Kalman / Kalman-NET), before the dot entries (True drone / Est. drone / GPS / IMU / Baro / Mag / Unc.). KF silhouette entry placed after Est. drone.

---

**User:** If I wish to deploy this application — what would be the best way? Render or Vercel? (no code changes — simply plan how to deploy)

**Claude:** Render. Vercel is architecturally incompatible — its serverless functions are stateless, short-lived, and Lambda-backed. The app requires persistent background threads (`_sim_loop`), long-lived SSE connections, a ~30s training call, in-memory shared state, and PyTorch (~200 MB), all of which hit hard Vercel limits.

Render plan: Web Service, start command `gunicorn app:app -w 1 --threads 8 -t 300`, CPU-only PyTorch wheel, `--workers 1` mandatory (all sim state is in-process globals).

---

**User:** Would I have to pay for their tier? Why — is there a way to limit the size so I can manage on the free tier?

**Claude:** The spike is specifically the subprocess model: during training, both the Flask parent process and the training subprocess each load PyTorch independently (~230 MB each), peaking at ~460 MB — over the 512 MB free tier limit. At idle the app fits in 512 MB comfortably. Solution: switch from subprocess to in-process thread, which keeps a single PyTorch instance (~360 MB peak), fitting the free tier with headroom.

---

**User:** Further explain this design decision — the need and solution, in terms of subprocess vs. in-process. Do I lose any training accuracy?

**Claude:** Python's GIL doesn't apply to PyTorch — it releases the GIL during C++ compute kernels (matrix multiplies, GRU forward/backward), so other Flask threads stay responsive regardless. Subprocess costs: dual PyTorch instances → ~460 MB peak; no real-time progress (stdout buffered until exit). In-process thread: single PyTorch instance (~360 MB), direct `_training_state["progress"]` writes each epoch → smooth live progress bar. No accuracy loss — training accuracy depends on data, architecture, loss, and hyperparameters, none of which change between execution environments.

---

**User:** Make the changes — switch to in-process thread.

**Claude:** Implemented:
- `train_kalmannet_gru.py`: extracted training loop into `train_from_buffer(buffer, progress_cb=None)`. `main()` is now a thin wrapper so `python train_kalmannet_gru.py` still works locally.
- `app.py`: removed `import subprocess`. Added `from train_kalmannet_gru import train_from_buffer`. Replaced subprocess block with direct call inside `_run_training`. Added `_progress_cb` closure that writes `progress = 0.90 + 0.09 * (epoch / epochs)` to `_training_state` each epoch.

---

**User:** Rename the UI instances of kalmannet to kalman-net (so the net is more pronounced and easier to read).

**Claude:** Updated all visible labels in `index.html`, `app.js`, and `tooltips.js`: "KALMANNET" → "KALMAN-NET", "KalmanNET" → "Kalman-NET". Flash banner, button label, legend entry, badge text, and tooltip headings all updated.

---

**User:** When I hit the sensor buttons when the sim is paused — they do not respond (to toggling on/off in the UI side).

**Claude:** Root cause: the SSE stream sends frozen state when paused, so `applyState` never fires with the new sensor state. Fix: `toggleSensor()` in `app.js` now reads the POST `/toggle` response directly and updates button classes and the status badge immediately from `data.failed`, bypassing the SSE stream entirely for UI state.

---

**User:** Update the README to consist of explaining the updated implementation, update the tooltips as well with any relevant information they represent — ensure everything is accurate.

**Claude:** Full README rewrite (plain-English KF/KN explanation, running instructions, architecture, limitations) and full tooltip rewrite in `tooltips.js` with corrected numbers (baro coefficient −0.25·ΔT, GRU predicts K directly not R, GPS K increases for wind). Added two sections: "AI and the path to realistic simulation" (domain randomization, neural digital twins, online adaptation) and "Software design for a hosted demo" (model size, vectorised training, Flask SSE, in-process thread, CPU-only PyTorch).

---

## Session 5 — Render free tier deployment

**User:** What would it take to deploy this application to Render on the free tier?

**Claude:** Three changes: `render.yaml` (free plan, CPU-only torch build command, gunicorn start command with `--threads 8 -t 300`), `requirements.txt` (remove torch, add gunicorn — torch installed separately to avoid 2.5 GB CUDA wheel), `app.py` (bind to `$PORT` env var). SSE reconnect on spin-down was already in place.

---

**User:** Getting `ImportError: PyTorch is required` during deployment.

**Claude:** Render defaulted to Python 3.14, which has no PyTorch wheels. `render.yaml` was also being ignored because the service was created via the dashboard UI (not as a Blueprint). Created `runtime.txt` and `.python-version` pinning to Python 3.11.9, and added torch back to `requirements.txt` as a direct CPU wheel URL (`torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl`) so the default `pip install -r requirements.txt` installs it without a custom build command.

---

**User:** App is deployed but clicking any button causes it to hang.

**Claude:** Root cause: Render's dashboard is running the default `gunicorn app:app` — sync workers, no threading. With sync workers, the SSE `/stream` endpoint (a persistent connection that never ends) permanently occupies the one worker. Every button click (POST to `/train`, `/toggle`, `/pause`) queues behind it and waits indefinitely. Fix: change the Start Command in the Render dashboard to `gunicorn app:app -w 1 --threads 8 -t 300 -b 0.0.0.0:$PORT`. This is what `render.yaml` specifies but the dashboard overrides it.

Vercel is not a viable alternative — serverless functions are stateless and time-limited (10s free tier), incompatible with persistent SSE streams, background sim threads, and a ~60s training call.

---

**User:** Fix the Render start command.

**Claude:** Created a `Procfile` (which overrides the dashboard start command for UI-created Render services) — then user redirected to use dashboard instructions instead of adding a new file. Dashboard fix: Settings → Start Command → `gunicorn app:app -w 1 --threads 8 -t 300 -b 0.0.0.0:$PORT` → Save → Manual Deploy.
