# BROWSER_PORT_PLAN.md — client-side port (UPGRADE_PLAN task 5.3, Stage B option ii)

Written 2026-07-09, after the deployment discussion logged in `CLAUDE_CHAT_LOG.md`
(Session 5). This is the dedicated plan file 5.3 asks for before Stage B coding.

## Decisions (made with the human, 2026-07-09)

1. **Option (ii)** — the whole sim moves client-side; `web/` is static-hostable
   (Vercel/GitHub Pages/Cloudflare Pages). The Flask app stays in the repo untouched
   as stage one of the server → browser → MCU portfolio arc.
2. **Hand-rolled JS inference instead of onnxruntime-web.** The trained model is
   ~18k parameters / 75 KB (`kalmannet_model.pt` = 76,367 bytes). One GRU step is
   ~36k multiply-adds — plain JS is thousands of times faster than the 20 Hz
   requirement, and a WASM runtime would be the only build/vendor complexity in an
   otherwise dependency-free port. ONNX export (5.3 Stage A) remains open as an
   independent portfolio artifact.
3. **Inference-first ("Path 1"): ship pretrained weights.** The TRAIN button still
   flies the real calibration flight and collects a real buffer, then **loads
   pretrained weights** (exported from the repo's `kalmannet_model.pt`) with honest
   labeling — the browser build does not run the optimizer. In-browser training
   (tfjs) or a scale-to-zero training endpoint are explicitly deferred (Session 5
   log has the full option analysis).

## Architecture: shim, don't fork

`static/app.js` touches the backend in exactly 7 places: `new EventSource('/stream')`
and `fetch()` to `/toggle`, `/environment`, `/reset`, `/pause`, `/clear-model`,
`/train`. So `web/index.html` loads the **unmodified** shared frontend
(`../static/app.js`, `../static/tooltips.js`, `../static/style.css`,
`../static/vendor/*`) and two new layers in front of it:

- **Engine** (`web/js/engine.js`) — owns a JS `DroneSim`, replicates `app.py`'s
  loop semantics: 50 ms tick, step only when unpaused, 10 steps/tick during the
  calibration flight (= Python's 200 Hz fast mode), emits the state dict (same keys
  as `DroneSim.step()` + `training` + `paused`) as a JSON string at 20 Hz.
- **Shim** (`web/js/shim.js`) — replaces `window.EventSource` and wraps
  `window.fetch`: the 6 API routes go to the engine, everything else (e.g.
  `weights.json`) passes through to the network.

Because the engine emits the exact SSE dict shape, the CLAUDE.md invariant
("SSE state keys are consumed directly in `applyState`") holds by construction.

**File map** (new files only):

```
web/
  index.html          adapted copy of static/index.html (relative asset paths,
                      engine/shim script tags, "100% IN-BROWSER" header badge)
  weights.json        exported GRU + head weights (generated, committed)
  js/mat.js           small dense-matrix helpers (no library)
  js/kf.js            KalmanFilter port
  js/kalmannet.js     KalmanNET port (GRU forward, K-clip envelope, Joseph update)
  js/sim.js           DroneSim port
  js/engine.js        local backend (loop, training flight, routes)
  js/shim.js          EventSource/fetch interception + honest tooltip/tour overrides
  test/parity.mjs     node test harness (no framework)
  test/reference_*.json  PyTorch-generated reference vectors (generated, committed)
export_web_weights.py  exports weights.json + reference vectors (run in .venv)
vercel.json            redirect / → /web/
```

## Numerics notes

- **PyTorch GRU gate math** (must match exactly; weight rows ordered r|z|n):
  `r = σ(W_ir·x + b_ir + W_hr·h + b_hr)`, `z = σ(...)`,
  `n = tanh(W_in·x + b_in + r∘(W_hn·h + b_hn))`, `h' = (1−z)∘n + z∘h`, `h₀ = 0`.
- Heads: `K = 2·σ(W_head·h' + b_head)` reshaped row-major to (6, m).
- The safety envelope ports verbatim: element-wise clip to `[0, 2·|K_Riccati|+1e-6]`
  (K_Riccati recomputed each step from live P), Joseph-form update with calibration
  R, symmetrization, divergence guard (reset x/P/h before innovations), and exact
  classical fallback when untrained. **Never remove the clip** (CLAUDE.md invariant).
- torch computes in float32; JS in float64. Bit-exactness is impossible, so parity
  is tolerance-based: single GRU steps ~1e-6, end-to-end trajectories bounded well
  below sensor-noise scale (structural bugs show up as 0.1–10 m deviations).
- Sim RNG differs from numpy by construction — sim behavior is validated
  statistically (untrained error at calm; trained-beats-KF under +30 °C bias),
  filter math is validated against recorded input→output reference vectors.

## Honest-labeling changes

- `shim.js` overrides `TOOLTIPS['btn-train'|'btn-clear'|'panel-kn-title']` at script
  eval (before tooltips.js's DOMContentLoaded applies them) and tour step 5 after
  app.js loads: the browser build **loads pretrained weights** after the flight;
  the collected buffer is real (`Engine.downloadBuffer()` saves it for offline
  training with `train_kalmannet_gru.py`); trained state persists in localStorage.
- One additive edit to shared `static/app.js`: a `step === 'loading'` case
  ("LOADING WEIGHTS" label + "LOADING PRETRAINED WEIGHTS" phase row) so the UI
  never claims OPTIMISING. The Flask backend never sends `'loading'`, so the
  server version is unaffected.

## Verification

1. `node web/test/parity.mjs` — KF parity (float64 vs float64, <1e-9), GRU parity
   vs `reference_gru.json`, end-to-end trained KalmanNET vs `reference_knstep.json`,
   plus the behavioral checks above.
2. Serve the repo root (`python -m http.server`), open `/web/`, exercise: play,
   sensor failures, env sliders, reset-while-paused, TRAIN flow end-to-end, CLEAR
   MODEL, reload-with-localStorage.

## Deferred (future stages)

- In-browser training (tfjs) or scale-to-zero training endpoint — Session 6 log.
- ONNX export + onnxruntime parity (5.3 Stage A) — still worthwhile as a toolchain
  exhibit even though the web build doesn't need the runtime.
- int8 quantization (5.4) and MCU port (5.6) reuse `export_web_weights.py`'s
  flattened-weights groundwork.
