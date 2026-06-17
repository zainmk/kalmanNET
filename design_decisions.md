# KalmanNET — Design Decisions & Conversation History

This document captures the full Q&A process of building and evolving this project, including the reasoning behind every major decision and the tradeoffs for real deployment.

---

## 1. Initial demo question — "what is the best way to see KalmanNET in action after training?"

**Question:** After training, what environment settings make the difference between KF and KN most visible?

**Answer given:** The best showcase was temperature stress. At 50°C (dT = +30°C), the barometer reads ~7.5 m low due to hot-air density bias. The fixed-R KF trusts this biased reading and its altitude estimate drifts 6–7 m off truth. The trained KN inflates R_baro by up to 500× and essentially ignores the barometer, holding altitude using only GPS cross-covariance. KN error ≈ 1.2 m vs KF error ≈ 7 m. The R̂ display shows BARO ×157–500, directly explaining the mechanism.

---

## 2. Diagnosis — "the KF and KN lines are still relatively close and both off"

**Question:** Both lines were similar in error and neither tracked truth well. Diagnose the issue.

**Benchmark results (after initial debugging):**

| Scenario | KF error | KN error | KN advantage |
|---|---|---|---|
| CALM | 0.29 m | 0.33 m | −0.04 m (KN slightly worse) |
| WIND 10 | 0.47 m | 0.79 m | −0.33 m (KN worse) |
| WIND 20 | 0.58 m | 0.65 m | −0.08 m (KN worse) |
| TEMP 30°C | 2.45 m | 0.63 m | **+1.81 m** |
| TEMP 40°C | 4.52 m | 1.33 m | **+3.19 m** |
| TEMP 50°C | 7.02 m | 1.38 m | **+5.64 m** |
| TEMP −10°C | 6.90 m | 2.57 m | **+4.33 m** |
| COMBINED | 4.44 m | 1.64 m | **+2.80 m** |

**Root cause:** Wind was originally added to the *true trajectory* (physically displacing the drone). GPS tracks the wind-displaced drone correctly. But KN inflated R_GPS (thinking GPS is noisy), which forced the filter to rely on the constant-velocity motion model — which doesn't know about wind. Result: KN drifted *worse* than KF in wind because it was ignoring the one sensor (GPS) that had the right answer.

**Fix applied:**
- Removed wind displacement from truth (wind only affects sensor noise variance, not position)
- Increased baro_bias coefficient 0.12 → 0.25 (more dramatic temperature showcase)
- Reverted wind noise coefficients to original (GPS 0.12, IMU 0.04)
- Increased process_noise 0.5 → 1.5 (helix centripetal acceleration is ~1.8 m/s²; low Q caused P to collapse, making K_IMU → 0, causing velocity drift)

---

## 3. Why R-prediction fails for wind — "so KalmanNET can't capture wind patterns?"

**Question:** If KN was worse in wind, does that mean it can't handle wind at all?

**Architectural explanation given:**

The R-prediction architecture has a fundamental limitation: it can only *decrease* trust in a sensor (inflate R → reduce K). It cannot *increase* trust beyond the Riccati-optimal level.

For temperature bias, this is correct. The barometer reads 7.5 m off — you want to trust it less. R_baro ↑ → K_baro ↓ → filter ignores the bad reading. Architecture fits the problem.

For wind displacement, the drone is physically *at* the GPS-measured position (the wind really moved it there). The KF's constant-velocity model doesn't know about wind, so it predicts the drone will be somewhere else. The correct response is to trust GPS *more* (increase K_GPS), so the estimate tracks the wind-displaced truth. But R-prediction can only reduce K_GPS — the opposite of what's needed.

This is not a tuning problem. It is a structural mismatch between the architecture and the task.

**Key insight:** R-prediction works when the sensor is wrong (bias, interference). It fails when the *model* is wrong (wind, turbulence, unmodelled dynamics). When the model is wrong, you need to trust the sensors more, not less.

---

## 4. Full KalmanNET GRU — "what would it take to capture wind patterns?"

**Question:** If R-prediction can't handle wind, what architecture can?

**GRU explanation given:**

The original KalmanNET paper (Revach et al., 2022) uses a GRU (Gated Recurrent Unit) that predicts K directly — not R. This matters for three reasons:

1. **K can go either direction.** A learned K can be larger or smaller than the Riccati-optimal K. When wind displaces the drone, the GRU can output K_GPS > K_Riccati (trust GPS more). When temperature biases the baro, it outputs K_baro < K_Riccati (trust baro less). R-prediction can only do the second.

2. **The GRU carries temporal context.** A GRU hidden state accumulates information over time. It can learn: "GPS and IMU innovations have been consistently large and correlated for 50 steps — this is probably sustained wind, not random noise. Increase K_GPS." A 20-step MLP window cannot do this.

3. **No oracle needed.** Since there is no R in the prediction, there is no need to know what the true R should be. The network is trained end-to-end on position loss: ||x̂[:3] − x_true[:3]||². It learns to minimize position error, and a good K is whatever achieves that. No simulation noise formulas needed at inference.

**What the GRU captures that R-prediction cannot:**
- Sustained wind patterns (temporal correlation in GPS/IMU innovations)
- Centripetal acceleration from curved trajectory (model mismatch)
- Cross-sensor correlations (GPS and IMU both scattered → wind; only baro off → temperature)
- Regime transitions (calm → windy → calm)

---

## 5. Training the GRU — "explain the training for this specific kind of network"

**Question:** How does BPTT work for a KalmanNET GRU? What does "1-2 days of implementation" mean?

**Training explanation given:**

**Forward pass (runtime, identical to training):**
1. Predict: x̂⁻ = F · x̂_{t−1}
2. For each active sensor: compute innovation ε = z − H · x̂⁻
3. Pack all innovations into a 9D vector (GPS=3, IMU=3, BARO=1, MAG=2)
4. GRU step: h_t, output = GRU(ε_9D, h_{t−1})
5. Per-sensor head: K_s = sigmoid_head_s(output) ∈ (0, 2) per element
6. Apply update: x̂ = x̂ + K_s · ε_s for each sensor
7. Repeat for next timestep, carrying h_t forward

**Loss:** L = Σ_t ||x̂_t[:3] − x_true_t[:3]||² (position MSE over the trajectory)

**Backpropagation Through Time (BPTT):**
The gradient must flow backwards through the sequence: loss at step t → K_t → GRU_output_t → h_t → h_{t−1} → ... This is the "unrolled" computation graph.

**Truncated BPTT (used here):** Processing a 5000-step trajectory at once creates a graph 5000 layers deep — gradients explode or vanish. Instead, split into 100-step chunks. At each chunk boundary, detach h and x̂ from the graph. Backprop only within the chunk. This trades gradient accuracy for training stability.

**What "1-2 days of implementation" meant:** The network itself is 10–20 lines of PyTorch. The complexity is in:
- The training loop correctly handling sequential filter state (x̂ must be propagated through K applications, not just the GRU)
- Making the training loop match inference exactly (same sensor ordering, same innovation computation)
- Hyperparameter tuning (learning rate, hidden size, chunk length, gradient clip)
- Verifying that K values remain in physically sensible ranges

**Training time for this demo:** ~15–30 seconds on CPU for 1000 epochs over a ~5000-step buffer.

---

## 6. Available libraries — "are there public APIs or libraries?"

**Question:** What PyTorch components does this use?

**Answer given:**

- `torch.nn.GRU` — the recurrent core. Takes a sequence of inputs (innovations), maintains hidden state h across steps, outputs a context vector at each step.
- `torch.nn.Linear` — per-sensor output heads. Maps the 64D GRU output → (6 × sensor_dim) K matrix elements.
- `torch.optim.Adam` — optimizer. Adaptive moment estimation; handles sparse/varying gradients well.
- `torch.optim.lr_scheduler.CosineAnnealingLR` — decays learning rate from 1e-3 → 1e-4 across training, helping convergence.
- `torch.nn.utils.clip_grad_norm_` — gradient clipping to norm 5.0, preventing exploding gradients in BPTT.

Install (CPU only, ~200 MB): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 7. Real deployment on microcontroller — "how practical would this be on a microcontroller?"

**Question:** Could a trained KalmanNET GRU run on a flight controller?

**Analysis given:**

**Model size:** A GRU with hidden=64, input=9, plus four linear heads, has roughly 30–50k parameters × 4 bytes = ~200 KB. Fits comfortably in flash on a Cortex-M7 (e.g., STM32H7 has 2 MB flash).

**Inference cost per step (50 ms interval at 20 Hz):**
- GRU: ~2 × 64 × (64 + 9) matrix-vector ops = ~9,300 multiply-adds
- 4 heads: ~4 × 64 × 18 = ~4,600 multiply-adds
- Total: ~14,000 MACs per step

A Cortex-M7 at 480 MHz does ~240M MACs/s. This inference takes ~60 µs — entirely feasible.

**Real challenges for deployment:**

1. **No float64.** Microcontrollers use float32. The KF and training should use float32 throughout (PyTorch uses float32 by default; numpy uses float64). This is handled by keeping the training in float32.

2. **No PyTorch at runtime.** Deploy the trained weights as flat arrays and implement the GRU equations manually in C:
   ```
   h_t = tanh(W_h · [ε_t, h_{t−1}] + b_h)   (simplified; real GRU has reset/update gates)
   K_s = 2 · sigmoid(W_s · h_t + b_s)
   ```
   This is ~50 lines of C with a preloaded weight table.

3. **The real bottleneck is the Kalman filter P matrix:** The 6×6 matrix inversion and multiplication at every update step is ~216 multiply-adds per sensor. More expensive than the GRU on very small MCUs.

4. **Sim-to-real gap** (the actual hard problem — see section 8).

---

## 8. Sim-to-real gap — "how accurate must the simulation be?"

**Question:** Training in simulation and deploying on real hardware — how big is the gap, and is it bridgeable?

**Analysis given:**

The trained network learns to recognize patterns in innovations that signal environmental conditions. If the simulation's sensor model diverges from the real sensor, the patterns differ and the network misidentifies conditions.

**What the simulation models accurately:**
- Gaussian additive noise (reasonable for GPS/IMU at slow update rates)
- Linear temperature-pressure relationship for barometer (well-established physics)
- Wind-as-constant-velocity-offset (reasonable for sustained crosswind in steady state)

**What the simulation gets wrong:**

| Real phenomenon | Simulation model | Gap |
|---|---|---|
| GPS multipath (reflections off buildings) | Pure Gaussian | Multipath produces heavy-tailed errors and correlated bursts — very different innovation signatures |
| IMU Allan deviation (bias instability, random walk) | Gaussian scaled by temperature | Real IMU bias drifts on 10–1000 second timescales; simulation treats it as white noise |
| Propeller wash on barometer | Temperature scalar only | Pressure disturbance from rotors adds ~0.3–1 m offset depending on throttle; not modelled |
| Magnetic interference from motors | Constant Gaussian | Motor current creates variable magnetic fields correlated with throttle |
| GPS fix dropout in urban canyons | No dropout modelled | Complete loss of fix is common in real flight |

**Industry approach: domain randomization.** Rather than modelling these effects precisely, randomize the simulation parameters widely during training:
- GPS noise std: uniform(0.5, 8.0) m instead of a fixed formula
- IMU bias: random walk process with random diffusion coefficient
- Baro offset: random step changes at random intervals

The network is forced to be robust to wide variation rather than perfectly calibrated to one simulation. This approach (from sim-to-real robotics literature) significantly reduces the transfer gap.

**Practical conclusion for this demo:** The simulation is accurate enough to demonstrate the *mechanism* (baro bias → KN ignores baro; wind displacement → GRU increases K_GPS). It is not accurate enough that the trained weights would work on a real drone without re-training on flight data. For production, you would collect 20–50 flights with differential GPS ground truth and train on that.

---

## 9. Architecture decision — full GRU vs. approximating R

**Question (confirmed):** Upgrade from numpy MLP R-prediction to full PyTorch GRU K-prediction.

**Tradeoffs discussed:**

| | R-prediction (current) | GRU K-prediction (new) |
|---|---|---|
| PyTorch dependency | No (~numpy only) | Yes (~200 MB CPU install) |
| Training time | ~2s | ~15–30s |
| Interpretability | R̂ values shown directly | K-ratio (vs Riccati K) shown |
| Wind handling | Fails (inflates R_GPS, wrong direction) | Correct (can increase K_GPS) |
| Temperature handling | Works well (inflates R_baro) | Works well |
| Training target | Oracle R (from known noise formula) | Position MSE (end-to-end, no oracle needed) |
| Runtime cost | ~1K multiply-adds per step | ~14K multiply-adds per step |
| Faithfulness to paper | Approximate (R-adaptive wrapper) | Faithful (Revach et al. architecture) |

**Decisions confirmed:**
- Separate training script (`train_kalmannet_gru.py`) — keeps `app.py` clean; script is inspectable and re-runnable
- Architecture Option A: shared GRU (hidden=64, input=9) + 4 per-sensor linear heads — shared GRU learns cross-sensor correlations (GPS+IMU both scattered → wind); per-sensor heads maintain sensor-specific K structure
- K-gain ratio display replacing R̂ grid — ratio > 1 means GRU is trusting sensor more than Riccati would (wind detection); ratio < 1 means GRU is trusting less (bias detection)
- Restore wind displacement to true trajectory — GRU can now correctly handle it by increasing K_GPS

---

## 10. Workflow after this change

1. Start app: `python app.py`
2. In browser: click **TRAIN KALMANNET** — runs calibration flight (~29s), saves `training_buffer.json`, runs `train_kalmannet_gru.py` via subprocess (~15–30s), saves `kalmannet_model.pt`
3. ACTIVE dot appears. KN trail (cyan) now uses GRU K-prediction
4. Set wind 15–20 m/s: watch KN track truth better than KF (GRU increases K_GPS to follow wind-displaced drone)
5. Set temp 50°C: watch KN hold altitude better than KF (GRU decreases K_baro to ignore biased reading)
6. K-GAIN RATIO panel shows ↑ for GPS in wind, ↓ for BARO in heat — the GRU's reasoning made visible

Re-training at any time: click **RETRAIN** — new buffer collected, script re-runs, model.pt overwritten, new weights loaded live.

The `kalmannet_model.pt` file persists between app restarts — the GRU resumes trained on next launch without re-running the calibration flight.
