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

## Session 5 — Readability

**User:** Increase the size of everything — the text and dashboard are almost too small to read.

**Claude:** Scaled the interface up in CSS — font sizes ~1.3× across the board, a wider sidebar, larger controls and chart, single-line header badges — verified against layout overflow, with the simulation logic left untouched.
