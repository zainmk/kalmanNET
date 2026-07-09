# KalmanNET — Development Transcript

A curated transcript of the conversation behind this project — the design
questions, diagnoses, and decisions, kept in dialogue. Routine back-and-forth is
trimmed; the technical substance is preserved.

---

## Session 1 — Building the demo & diagnosing the learned filter

**User:** What's the best way to see Kalman-NET in action after training?

**Claude:** The clearest showcase is temperature stress. At 50 °C the barometer reads ~7.5 m low — in warm air the pressure at a given altitude is higher than the standard-atmosphere model assumes, so the altimeter indicates low. The fixed-R Kalman filter can't tell that bias from noise, trusts it, and drifts 6–7 m off truth. The trained Kalman-NET detects the sustained, one-sided error, down-weights the barometer, and holds altitude through GPS — roughly 1.4 m error versus 7 m.

---

**User:** The Kalman filter and Kalman-NET lines are still close, and both off. Can you diagnose why, and how to get Kalman-NET tracking the true trajectory more closely?

**Claude:** Ran a live benchmark and found Kalman-NET *losing* under wind while winning under temperature:

| Scenario | KF | KN |
|---|---|---|
| calm | 0.29 m | 0.33 m |
| wind | 0.47 m | 0.79 m |
| heat 50 °C | 7.0 m | 1.4 m |

Root cause: wind was displacing the *true* position. GPS measured the displaced position correctly, but the (early) R-prediction network inflated R_GPS — it "thought" GPS had gone noisy — and forced reliance on a constant-velocity model that didn't know about the wind. The network was ignoring the one sensor with the right answer.

---

**User:** So capturing wind in Kalman-NET would need the larger GRU from the paper? Help me understand Kalman-NET's use case better (no code changes).

**Claude:** Correct, and the reason is structural. An R-prediction network can only *inflate* R — trust a sensor *less*:

- **Temperature bias** → the sensor is wrong → trust it less → fits the architecture.
- **Wind displacement** → the model is wrong (the drone really is there) → you need to trust GPS *more* → the architecture can't express it.

The paper's Kalman-NET instead predicts the gain K directly with a GRU. K can move above *or* below the Riccati-optimal value; the recurrent state carries temporal context (learning "sustained GPS+IMU scatter = wind"); and it trains end-to-end on position error with no oracle.

---

**User:** Any public libraries for the GRU? And explain how this kind of network is trained.

**Claude:** Standard PyTorch — `nn.GRU`, per-sensor `nn.Linear` heads, `Adam`, `CosineAnnealingLR`, `clip_grad_norm_`; the CPU-only wheel is ~200 MB. Training is backprop-through-time: each step, predict x̂⁻ = F·x̂, form the innovations, run the GRU → K per sensor, apply x̂ = x̂ + K·ε; loss = Σ‖x̂[:3] − x_true[:3]‖². Gradients flow back through the K → GRU → hidden-state chain, truncated into ~100-step chunks with detached boundaries to keep gradients from exploding. For this demo it's ~15–30 s on CPU.

---

**User:** How practical would deploying this on a microcontroller be?

**Claude:** Very. ~70 KB of weights fit in flash on a Cortex-M7; one fusion step is ~14 k multiply-adds, about 60 µs at 480 MHz. Export the weights as flat C arrays and hand-write the GRU recurrence in ~50 lines — on small MCUs the Kalman covariance inversion is actually the heavier cost. The genuine difficulty is never the compute; it's the sim-to-real gap.

---

**User:** So the trained result is only as trustworthy as the simulation matches the real drone. Are simulations accurate enough to deploy?

**Claude:** That's exactly the bottleneck. The sim captures Gaussian noise and a linear temperature–pressure relationship, but real GPS multipath is heavy-tailed and spatially correlated, IMU bias drifts on 10–1000 s timescales, and propwash and magnetic interference aren't modelled at all. The standard mitigation is domain randomisation — train over widely varied noise parameters so the network becomes robust to distribution shift rather than tuned to one model. This sim is faithful enough to demonstrate the *mechanism*, not to fly on real hardware without retraining on flight data against a ground-truth reference (RTK/differential GPS or motion capture).

---

**User:** Implement the full PyTorch GRU approach — separate training script, predict K directly with the ratio display, and restore wind displacement to the true trajectory.

**Claude:** Implemented: a shared GRU (hidden = 64) over a 9-D combined-innovation vector with per-sensor K heads (`2·sigmoid`) and a Joseph-form covariance update; a standalone `train_kalmannet_gru.py` (BPTT, Adam + cosine LR) that writes a ~70 KB `kalmannet_model.pt` and auto-loads on start; one-click training that collects the calibration buffer inside the running sim; and a UI showing the K-gain ratio (↑ trusting a sensor more, ↓ less, versus the standard filter).

---

## Session 2 — Stability and hosting

**User:** After training, hitting *play* throws `RuntimeWarning: overflow encountered in cast`.

**Claude:** The GRU emits a dense K (all 18 elements near 1–2). Its off-diagonal terms push the eigenvalues of (I − K·H) outside the unit circle, so the covariance grows every step until the state diverges and overflows float32. Fix: clip K element-wise to `[0, 2·|K_Riccati|]`. The Riccati K has near-zero off-diagonals, so the clip suppresses exactly the destabilising terms while still letting the network trust a sensor up to twice as much, or ignore it entirely. A divergence reset backs it up. This clip is the filter's runtime stability envelope.

---

**User:** Should I deploy on Render or Vercel? (plan only)

**Claude:** Render. Vercel's serverless model is stateless and short-lived — incompatible with the persistent background sim thread, the long-lived SSE stream, the ~30 s training call, and the in-memory shared state. Render runs it as one process: `gunicorn app:app -w 1 --threads 8 -t 300`. The single worker is mandatory because all state lives in process globals; multiple workers would each run an independent simulation.

---

**User:** Can it run on the free tier? And explain subprocess vs in-process training — do I lose accuracy?

**Claude:** It fits the 512 MB free tier if training runs in-process. A subprocess loads PyTorch twice (~230 MB each, ~460 MB peak — over the limit); an in-process thread keeps one interpreter at ~360 MB. PyTorch releases the GIL during its C++ compute, so the SSE stream stays responsive throughout, and progress writes directly to shared state each epoch (no stdout parsing). No accuracy change — that depends only on data, architecture, loss, and hyperparameters.

---

**User:** Deployed, but clicking any button hangs the app.

**Claude:** The host was running default *sync* workers. With a sync worker, the never-ending SSE `/stream` connection permanently occupies the single worker and every other request queues behind it forever. Fix: run threaded — `gunicorn app:app -w 1 --threads 8 -t 300 -b 0.0.0.0:$PORT` — so threads serve the concurrent SSE streams and API calls while one process still owns the shared state.

---

## Session 3 — Assessment, correctness, and UX

**User:** Assess the codebase — bugs, UX, tooltip accuracy, README — and give portfolio direction for a fullstack developer moving into edge AI.

**Claude:** Full read-through. Key findings: the core KF-vs-KN error metric was computed but shown nowhere; the uncertainty sphere displayed the wrong filter's covariance; the training thread had no exception guard, so a mid-run failure could lock the UI permanently; a /train double-click could spawn two training threads; reset-while-paused desynced the browser. Several tooltips were physically wrong — the barometer mechanism was stated backwards, and "constant-velocity helix" is a contradiction the demo should teach rather than gloss over. Portfolio direction: frame the K-clip plus classical fallback as a *runtime safety envelope* (runtime assurance for learned components in control loops), surface the comparison with a live error chart and honest benchmark numbers, and push toward on-device inference — server → browser → microcontroller — as the narrative.

---

**User:** Implement the fixes — correctness first, then the UX and content-accuracy work.

**Claude:** Correctness: wrapped the training thread so a crash always unlocks the UI; made /train claim its slot atomically; fixed reset-while-paused to push a fresh frame; reordered the divergence guard before innovations; guarded the sim loop against a bad step. Content accuracy: corrected the barometer physics and the constant-velocity framing, and added the missing Kalman-NET panel tooltips. UX: a live position-error chart of both filters, a non-destructive A/B toggle between them, a first-visit guided tour, a "10× calibration flight" indicator, debounced environment controls, an SSE-reconnect badge, and vendored Three.js so the app works offline.

---

## Session 4 — When is a learned filter better? Bias vs noise

**User:** Under max wind with no temperature change, Kalman-NET barely differs from the Kalman filter — and looks more erratic and slightly worse. Is that right? Correct my understanding if I'm off.

**Claude:** Your reading is right, and it's the crux of the whole project. A Kalman filter is the provably optimal (minimum-variance) estimator when errors are zero-mean noise. Wind only inflates GPS/IMU noise — the mean stays correct, and because GPS reports position every step, the filter simply follows it; the motion model never has to extrapolate far. So a learned gain has almost nothing to add, and its denser K chases the scatter, coming out slightly worse. Temperature is different: it introduces a *bias* — a non-zero-mean offset no averaging removes and a fixed R is structurally blind to — and that's the regime the learned filter wins.

I also tested whether stronger wind could change this: even at ~30 m of displacement the Kalman filter still won. The only thing that flipped the wind result was faking a directional GPS bias plus an unrealistically clean magnetometer to cross-check it against.

---

**User:** Keep the physics honest — don't fake a wind win. Document the distinction between the two filters as part of the demo's teaching.

**Claude:** Added a "bias vs noise" section to the README and reconciled the tooltips: the Kalman filter is the right tool against noise; Kalman-NET earns its place against bias — systematic error a fixed noise model can't see. Removed the earlier overclaims that implied the learned filter wins under wind, and stated plainly where and why it loses. The lesson the demo now teaches: know when the simple, provably-optimal estimator already suffices, and reach for the learned one only against a real, structured failure. More model complexity is not automatically more accuracy.

---

## Session 5 — Public deployment strategy and the browser port

**User:** I'd like to deploy this publicly. Render's free tier wasn't stable, and neither was my NAS-with-Tailscale attempt. What's the best way?

**Claude:** Render free is structurally incompatible: 15-min spin-down kills the 20 Hz sim thread (cold start re-imports torch), 0.1 shared vCPU can't sustain the sim plus ~32 s CPU-bound training, and 512 MB RAM is marginal against the ~360 MB torch-loaded baseline. Recommended the NAS as the right home: the app needs one always-on process with ~1 core / ~1 GB, which the existing Dockerfile/compose already provides. Key fixes: (1) plain Tailscale is private, not public — use Tailscale Funnel (free `.ts.net` URL) or Cloudflare Tunnel (custom domain) instead of port-forwarding, and specifically avoid the NAS built-in reverse proxy, whose ~60 s idle timeout and buffering kill the persistent SSE stream (the likely "instability"); (2) build the Docker image on the PC and ship it — installing the 180 MB torch wheel during a build can OOM a NAS, and the requirements.txt pin is x86_64-only so it fails outright on ARM; (3) add a container memory limit. Fallback: a $4–6/mo VPS with the same compose file. Long-term the roadmap's browser-side inference makes hosting a static page. Also flagged: public visitors share one global sim and anyone can trigger /train (30 s CPU burn + sim reset for everyone) — suggested rate-limiting it before going public.

---

**User:** Would this still keep the sim as shared state between all users?

**Claude:** Yes — that's a property of the code, not the host: all state is in module globals, one thread steps one DroneSim, and every SSE connection streams the same `_state`, so any visitor's pause/reset/slider/train affects everyone (the single-worker rule exists to protect exactly this). Two paths to per-user isolation: (1) a per-session registry server-side — each session gets its own sim, cheap to step (6-state KF + 64-hidden GRU), but needs session eviction and a queue for the ~32 s CPU-bound training runs; (2) port the sim + KN inference to the browser (ONNX Runtime Web or hand-rolled — the GRU is tiny), giving per-user sims by construction, zero server state, and static hosting on GitHub Pages. Recommended skipping option 1: shared state is fine for an occasionally-shared demo link, and the browser port — already the roadmap's "server → browser" milestone — makes the refactor obsolete and is the stronger portfolio story.

---

**User:** Explain in detail what moving to the browser means and does — especially browser memory/computation limits and the methods for moving 'large' models local.

**Claude:** The port inverts the architecture: today the browser is a display for server-computed SSE frames; after the port, the sim step, 6-state KF, and KalmanNET GRU inference all run in JS in each visitor's tab, and the entire server layer (Flask, gunicorn, SSE, sim thread, locks, single-worker rule) disappears — the app becomes static files on GitHub Pages. Need/reason: hosting cost and instability vanish (no server exists), per-user isolation is automatic, the loop runs at display rate with zero latency, and the portfolio story upgrades from "served PyTorch from Flask" to the edge-AI export→optimize→deploy pipeline. Browser limits: JS heap ~2–4 GB, WASM linear memory capped at 4 GB (Memory64 still uneven), WebGPU buffer limits device-dependent, and weights are a download before they're a computation. Compute tiers: plain JS → WASM (+SIMD, +threads needing cross-origin-isolation headers) → WebGL → WebGPU compute shaders. The project's model is ~18k params / 75 KB on disk (verified: kalmannet_model.pt = 76,367 bytes; the training buffer is 20× bigger) — ~36k MACs per step, so plain hand-rolled JS (~80 lines for the GRU forward pass + heads + Joseph update + K-clip) is thousands of times faster than the 20 Hz requirement; no ONNX/tfjs/WebGPU needed. The hard part is training, not inference: either reproduce the 200-epoch Adam run in TensorFlow.js or ship pretrained weights and scope the demo to inference. Large-model methods (the toolkit this model doesn't need): quantization (fp16/int8/4-bit GPTQ/AWQ/GGUF — 8 GB-class LLMs fit tabs at 4-bit), distillation/pruning, weight sharding + Cache API/IndexedDB caching (WebLLM-style), runtimes (ONNX Runtime Web, tfjs, transformers.js, WebLLM/MLC), and native local targets (Core ML, LiteRT, ExecuTorch, llama.cpp). Tie-in to roadmap: int8-quantized the model is ~18 KB — MCU-sized — so the same pipeline continues to TFLite-Micro/CMSIS-NN, giving the "same 75 KB filter running in server, browser, and bare metal" portfolio arc.

---

**User:** Could I deploy the app via Vercel if all in browser?

**Claude:** Yes — Vercel was only ruled out earlier because of the server (stateless short-lived serverless functions can't host the sim thread, SSE stream, or 30 s training call). Fully in-browser, the app is static files, which Vercel's free Hobby tier serves perfectly: CDN, HTTPS, deploy-on-push, preview URLs. GitHub Pages / Cloudflare Pages / Netlify are equally viable. Footnotes: Vercel Hobby has a non-commercial clause (portfolio use is fine); if in-browser training with WASM threads is ever added, SharedArrayBuffer needs cross-origin-isolation headers — settable on Vercel/Cloudflare Pages but not GitHub Pages.

---

**User:** What are the pros and cons of the browser port?

**Claude:** Pros: deployment problem solved permanently (static files — no spin-down, RAM budget, SSE timeouts, tunnels, or cost); per-user isolation by construction; no abuse surface (/train can't be hit by strangers); zero-latency loop; the edge-AI export→validate→deploy portfolio story and stepping stone to the MCU milestone. Cons: a real rewrite of several hundred lines of numerics (days, not hours); silent-wrongness risk — a buggy hand-rolled GRU/Joseph update still yields plausible tracks, so PyTorch-exported parity test vectors are required and the project has no test infra yet; the training demo is the genuine dilemma (drop it / reimplement in tfjs / keep a training server — each costs something, and "watch it learn" is the demo's centerpiece); two implementations to sync or the Flask one freezes as legacy (on top of the existing 3-place sensor-noise prose sync); deleting the server deletes the backend-engineering exhibit (mitigate by keeping both as stages of the server→browser→MCU arc); no control of the client runtime (needs fixed-timestep accumulator vs variable rAF rates, background-tab throttling). Not cons: size/memory/compute — 75 KB and ~36k MACs/step is far below any browser limit. Verdict: pros win for the portfolio goal; scope as an inference-first parallel static/ variant with shipped pretrained weights and parity vectors, keep Flask in-repo as stage one, decide on in-browser training later.

---

**User:** Elaborate on the training-in-browser feature loss — how does the port affect it, how can I remedy it later, can the user still see the training phase and collect data, and how does the browser version get the trained weights?

**Claude:** Training today is three phases: (1) the 10× calibration flight collecting {measurements, x_true} into the buffer, (2) the 200-epoch Adam optimization in PyTorch (~32 s), (3) the reveal (weights load, cyan drone, A/B). Phases 1 and 3 port trivially — the flight is just the sim stepping while an array records (and the browser lifts the 10× cap), and the reveal is loading 75 KB of numbers. Only phase 2 is lost, because training needs autodiff + an optimizer (framework machinery PyTorch provides natively but browsers lack), whereas inference is plain arithmetic. So yes: the user still sees the flight and collects data in all scenarios. Three ways the browser gets trained weights: Path 1 — ship pretrained weights as a static asset (train offline in Python, export state-dict to JSON/binary, fetch()); keeps the flight and reveal but breaks causality ("your data trained this") — label honestly, don't fake a training bar. Path 2 — reimplement the training loop in TensorFlow.js (GRU/Adam/autodiff exist, but the custom teacher-forced graph of GRU→K heads→filter updates→position loss must be rebuilt with tfjs gradient APIs; ~1–2 MB dep; overhead-dominated workload so same order as 32 s, likely slower on WASM; acceptance test is behavioral — beats KF under temp bias); bonus: persist per-user weights in localStorage across visits. Path 3 — static site + transient cloud training endpoint (Modal/HF Space, scale-to-zero) running the unmodified train_from_buffer(): upload 1.5 MB buffer, get 75 KB weights back — full causality, zero rewrite, but reintroduces cold starts/abuse surface/second provider (Vercel functions can't fit torch). Recommended sequence: Path 1 first (its artifacts are prerequisites for the others), Path 2 as its own milestone if the causality loss matters (it's the strongest story), Path 3 as escape hatch if tfjs fights.

---

**User:** Build Path 1.

**Claude:** Built the full inference-first browser port (UPGRADE_PLAN 5.3 Stage B, option ii — plan file `BROWSER_PORT_PLAN.md` written first per the 5.3 ground rule; deviation from spec: hand-rolled JS instead of onnxruntime-web, justified by the 18k-param/75 KB model). Architecture: **shim, don't fork** — `static/app.js` touches the backend in exactly 7 places (EventSource('/stream') + six fetch routes), so `web/index.html` loads the unmodified shared frontend (`../static/app.js`, tooltips, CSS, vendored three.js) with two new layers in front: `web/js/engine.js` (a JS `DroneSim` replicating `app.py`'s loop — 50 ms tick, 10 steps/tick fast-mode, emits the exact SSE dict shape + training/paused) and `web/js/shim.js` (reroutes EventSource/fetch to the engine; passes weights.json through). New numerics ports: `mat.js` (Gauss-Jordan inverse, dense helpers), `kf.js`, `kalmannet.js` (PyTorch-exact GRU gate math r|z|n, 2·sigmoid heads, K-clip envelope, Joseph update, divergence guard, exact classical fallback), `sim.js` (same state-dict keys — the SSE contract holds by construction; Box-Muller noise, injectable RNG for seeded tests). `export_web_weights.py` dumps the trained state-dict to `web/weights.json` (389 KB JSON) plus three PyTorch reference-vector files. **Parity results (node web/test/parity.mjs, all 9 pass):** KF vs numpy max |Δx| = 2.7e-15; GRU vs torch = 1.7e-7 (float32/float64 floor); trained KalmanNET end-to-end over 300 steps = 8.4e-7 m; untrained fallback ≡ classical KF at 0.0; seeded behavioral runs reproduce the demo claims (calm 0.30 m; at 50 °C KN 3.53 m vs KF 7.05 m). Honest labeling: TRAIN flies the real calibration flight and collects a real buffer (`Engine.downloadBuffer()` exports it), then shows "LOADING PRETRAINED WEIGHTS" — a small additive `step === 'loading'` case in shared app.js (Flask never sends it); shim overrides the btn-train/btn-clear/panel-kn-title tooltips and tour step 5; trained state persists per-browser in localStorage; new "100% IN-BROWSER" header badge. Docs: README gains a "Browser build (no server — static hosting)" deployment section and the shared-sim note now points to web/; `vercel.json` redirects / → /web/; CLAUDE.md architecture + invariants updated (sensor-noise sync is now 4 places incl. web/js/sim.js; filter-math changes must be mirrored + parity rerun; _TRAIN_PHASES triplicated across app.py/engine.js/KN_PHASES); UPGRADE_PLAN 5.3 annotated (Stage B shipped, ONNX Stage A still open). Verified: all 13 static asset paths 200 via http.server, `node --check` clean, parity suite green, Flask app boots and serves the edited app.js. Not committed — awaiting review. Run: `python -m http.server` from the repo root → http://localhost:8000/web/ (must be served over HTTP from the root — the page fetches weights.json and pulls the shared frontend from ../static/).

---

**User:** Are there files I no longer need? And curate this log — keep only exchanges with project-purpose/portfolio substance, no filler.

**Claude:** Curated the log (removed the how-to-run/tool-explanation entries and the CSS-readability session; policy codified in CLAUDE.md: substance only). File audit: recommended dropping `demo.mp4` from git (README embeds a GitHub-hosted copy — UPGRADE_PLAN 4.7's pending decision), untracking the regenerated-artifact `training_buffer.json` (gitignore it), and deleting the Render-specific `render.yaml`/`runtime.txt`/`.python-version` if Render is abandoned (pruning the README Render section to match); keep `kalmannet_model.pt` (source of truth for weights.json), the Docker files (NAS/self-host stage), and `benchmark.py` (README Results). Reminder: the new `web/`, `vercel.json`, `export_web_weights.py`, and `BROWSER_PORT_PLAN.md` are untracked and need `git add` on the next commit.

---

**User:** I deleted some of the files — update the README and any other references to Render.

**Claude:** Render is now fully retired from the living docs (deleted: `render.yaml`, `runtime.txt`, `.python-version`, `demo.mp4`; `training_buffer.json` kept). Removed the "Render (or similar PaaS)" section from README's Deployment (Docker/NAS and the browser build remain); requirements.txt pin comment now says Docker only; CLAUDE.md no longer names Render and notes the browser build sidesteps the single-process constraint entirely; `.dockerignore` rebuilt — dropped the deleted demo.mp4, excluded `web/`, `vercel.json`, `export_web_weights.py`, and `BROWSER_PORT_PLAN.md` from the Flask image (the container doesn't need the browser build). UPGRADE_PLAN forward-looking items updated: 4.6's live-demo link now targets the deployed browser-build URL instead of a Render URL, 4.7 (demo.mp4) marked done, and the Phase 6 "re-verify Render start command" item marked obsolete. Historical references (completed-task text, this transcript) intentionally preserved. The deployment story is now two-track: Docker/NAS for the full server version, static hosting for the public demo.
