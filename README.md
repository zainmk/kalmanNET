# KalmanNET

A simulation showing where classical sensor fusion breaks down in changing environments — and how[ Kalman-NET](https://pure.tue.nl/ws/files/201346390/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics.pdf), a recurrent neural network, learns to compensate.



https://github.com/user-attachments/assets/6e2a2b0b-b7e2-479f-b9fd-45d4e3c08ee4


---

## Kalman Filters

A flying drone knows where it is by reading from several sensors at once — GPS for absolute 3-D position, an IMU for velocity, a barometer for altitude, a magnetometer for horizontal backup. Each sensor is noisy: every reading is slightly wrong in an unpredictable direction.

A **Kalman filter** fuses these readings into a single estimate more accurate than any individual sensor. It does this in two steps every timestep:

1. **Predict** — extrapolate the drone's state forward using a physics model (constant velocity)
2. **Update** — correct using each active sensor reading, weighted by how trustworthy that sensor is

The amount of correction applied is controlled by the **Kalman gain K**:

```
K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹
```

**R** is the measurement noise covariance — it encodes how noisy each sensor is. Large R → small K → sensor barely shifts the estimate. Small R → large K → sensor dominates. The filter was calibrated at calm conditions (0 m/s wind, 20°C) and R is fixed at those values.

**The problem.** When the environment changes — wind raises GPS noise, temperature biases the barometer — the filter does not know. R stays fixed. The filter keeps trusting sensors at the wrong weights and the estimate drifts.

---

## Kalman-NET

Kalman-NET replaces the gain formula above with a **GRU** (Gated Recurrent Unit) neural network. Instead of computing K from a fixed R, the GRU predicts K directly by observing the history of recent **innovations** — the gap between what sensors measured and what the filter expected.

**Why direct-K matters.** R-inflation (the alternative approach) can only push K *down* — it cannot trust a sensor more than the formula allows. Direct-K can move the gain either way:

| Scenario | What the filter needs | R-inflation | Direct-K |
|---|---|---|---|
| Sparse/lagging GPS, drone displaced | Trust GPS *more* to follow it | ✗ | ✓ |
| A sensor develops a systematic bias | Trust that sensor *less* to ignore it | ✓ | ✓ |

That is the *capability* difference. Whether it actually pays off depends on the failure mode — and only the second row materialises in this simulation. Because GPS here reports position every step, the filter already follows the wind-displaced trajectory without extra gain, so wind reduces to a *noise* problem the classical filter handles well; the barometer *bias* is where the learned gain earns its keep. See [When to use which: bias vs noise](#when-to-use-which-bias-vs-noise) for the full argument.

**How the GRU works.** All four sensors' innovations are combined into a single 9-dimensional vector each step. The GRU processes this and outputs a K matrix for each sensor — the shared network can exploit cross-sensor patterns (e.g. GPS and IMU both showing large innovations simultaneously signals wind, not just individual sensor noise).

**Training.** The network is trained once on a scripted calibration flight that varies wind speed and temperature across the full operating envelope. The training signal is position error — the network learns whatever K adjustments minimise how far the estimate drifts from the known true trajectory. At inference, no ground truth is needed: only live sensor data.

---

## Results

Position RMSE over a 60-second flight in each scenario, standard Kalman filter vs Kalman-NET:

| Scenario | KF RMSE (m) | KN RMSE (m) | Improvement |
|---|---|---|---|
| calm | 0.360 | 1.059 | -194.6% |
| wind | 0.506 | 1.530 | -202.1% |
| heat | 7.097 | 1.463 | +79.4% |
| combined | 4.736 | 1.552 | +67.2% |

*Numbers from `benchmark.py` with a fixed seed (`np.random.seed(42)`) on a locally trained model — retraining changes them slightly. Run `python benchmark.py` to reproduce (requires a trained `kalmannet_model.pt`).*

The story is the tradeoff, not a uniform win. **Kalman-NET dominates exactly where the fixed-R Kalman filter breaks down**: under temperature-induced barometer bias (`heat`, `combined`) it cuts position error by 67–79%, because it learns to *reduce* trust in the biased altimeter — something a fixed R cannot do. But in the easy regimes (`calm`, mild `wind`) the filter is already near-optimal at sub-metre accuracy, and the teacher-forced network carries a ~1 m error floor it does not beat there (see [*Training is teacher-forced, unlike the paper*](#training-is-teacher-forced-unlike-the-paper) below). The honest summary: this Kalman-NET trades a little accuracy in calm conditions for large robustness gains under the environmental stress it was designed for.

---

## When to use which: bias vs noise

The results above look inconsistent — Kalman-NET wins big under heat but *loses* under wind — until you see the pattern. It is not wind versus temperature. It is **bias versus noise**, and it is the single most important idea this simulation teaches.

The standard Kalman filter is **provably optimal** — the minimum-variance estimator — under one condition: every sensor's error is zero-mean noise with a known, constant covariance. That assumption is exactly what its fixed `R` encodes. Kalman-NET can only beat the filter where that assumption breaks, and environments break it in two fundamentally different ways.

**Noise (zero-mean, wrong level) — the Kalman filter wins.**
Wind raises GPS and IMU noise above their calibrated values (GPS σ = 2 + 0.12·w m). `R` is now too small and the filter slightly over-trusts those sensors — but the error is still *zero-mean*: average enough noisy readings and the truth emerges. A wrong `R` only costs a little, because against zero-mean noise the classical filter is already near-optimal by construction. And since GPS reports absolute position every step, the filter simply follows it — the constant-velocity model never has to extrapolate far, so it tracks the wind-displaced trajectory without trouble. A learned gain has almost nothing to add here, and its denser, less-stable K tends to chase the noise rather than smooth it. This is why Kalman-NET is *worse* under wind: there is no structural failure to fix, only optimal behaviour to imitate imperfectly.

**Bias (non-zero mean) — Kalman-NET wins.**
Temperature gives the barometer a persistent altitude offset (−0.25·ΔT m — about −7.5 m at 50 °C). This is *not* noise: no amount of averaging removes it, because the average itself is wrong. A fixed `R` models spread, not offset, so the filter literally cannot tell a systematic lean from random scatter — it trusts the biased reading at full weight and follows it down. Kalman-NET sees the sustained, one-directional innovations that only a bias produces, and *reduces* the barometer's gain, re-anchoring altitude on the still-clean GPS. No fixed noise model can do this at any calibration, which is why the win is large and durable.

So the rule of thumb:

> **The Kalman filter is the right tool against noise. Kalman-NET earns its place against bias** — systematic, structured error that a fixed noise model is blind to by construction.

This also explains the `calm` row, where Kalman-NET is *worse*: with no bias and correctly-calibrated noise, the classical filter is already optimal and the network can only add variance. The lesson is not "the neural filter is better" — it is **knowing when the simple, provably-optimal estimator already suffices, and reaching for the learned one only when a real, structured failure appears.** More model complexity is not automatically more accuracy.

*(A corollary worth stating: to make the wind scenario favour Kalman-NET, the honest fix is in the physics, not the network — wind would have to* bias *a sensor, e.g. GPS multipath, that a second clean sensor can cross-check, mirroring the barometer case. This simulation deliberately does not fake that, so wind stays a noise problem the classical filter rightfully wins.)*

---

## Running

```bash
pip install flask numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
python app.py
```

Open `http://localhost:5000`. Press **▶ PLAY** or `Space` to start. Press `R` to reset.

**Training:** click **TRAIN KALMAN-NET** in the sidebar. The simulation runs a ~30-second scripted calibration flight collecting sensor data, then optimises the GRU for another ~30 seconds. The trained model saves to `kalmannet_model.pt` and reloads automatically on next start.

**Suggested demo sequence:**
1. Play and observe the Kalman filter (white drone) tracking the green truth drone at calm conditions
2. Raise temperature to 50°C — watch the KF altitude drift as the barometer introduces a −7.5 m bias
3. Raise wind to 15–20 m/s — watch the KF struggle to follow the wind-displaced trajectory
4. Reset, train Kalman-NET, repeat — observe the cyan drone tracking closer to truth under the same stress

---

## Deployment

### Docker (self-hosting / NAS)

```bash
docker compose up --build
# http://localhost:5000
```

The container runs `gunicorn app:app -w 1 --threads 8 -t 300`. **The single worker is a hard requirement** — simulation state lives in process globals, so multiple workers would each run an independent simulation. Threads (not workers) serve the concurrent SSE streams.

### Render (or similar PaaS)

Works on the free tier. Python is pinned to 3.11 (`runtime.txt`, `.python-version`) because the CPU torch wheel in `requirements.txt` is cp311. If the dashboard start command overrides `render.yaml`, set it manually to `gunicorn app:app -w 1 --threads 8 -t 300 -b 0.0.0.0:$PORT`. Free instances spin down when idle — the first visit after idle takes ~30 s and the UI shows RECONNECTING… until the stream resumes.

> **Note:** all visitors to one deployment share a single simulation — pausing, training, or changing the environment affects everyone connected. This is a demo design tradeoff, not an oversight (see `UPGRADE_PLAN.md` §5.3 for the planned client-side inference that removes it).

---

## Architecture

| Layer | Technology |
|---|---|
| Simulation & filters | Python — `drone_sim.py`, `kalman_filter.py`, `kalmannet.py` |
| GRU training | PyTorch — `train_kalmannet_gru.py` |
| HTTP / SSE server | Flask (`app.py`) |
| 3-D visualisation | Three.js r128 |
| UI | Vanilla JS + CSS (`static/`) |

The simulation runs at 20 Hz in a background thread. State is pushed to the browser over a persistent SSE connection (`/stream`) as JSON; the browser applies each frame to the Three.js scene without polling.

**GRU:** shared network (input = 9 combined innovations, hidden = 64) + four per-sensor linear heads. Outputs K ∈ (0, 2) per element via 2·sigmoid. One GRU call per filter step processes all sensors together. K is clipped element-wise to `[0, 2 × |K_Riccati|]` to preserve filter stability.

**Training:** teacher-forced, fully vectorised. ~5 800 raw steps subsampled 10× → 580 GRU steps. Loss = Σ ‖x̂_t[:3] − x_true_t[:3]‖². Adam + cosine LR (1e-3 → 1e-4), 200 epochs, ~32 s on CPU.

---

## Limitations and real-world deployment

### Simulation noise vs real sensor noise

This simulation models sensor noise as additive Gaussian with analytically defined standard deviations. Real sensors are not:

| Phenomenon | This simulation | Reality |
|---|---|---|
| GPS multipath | Pure Gaussian ±(2 + 0.12·w) m | Heavy-tailed, spatially correlated bursts near structures |
| IMU bias drift | Gaussian, temperature-scaled | Allan deviation: bias drifts on 10–1 000 s timescales independent of temperature |
| Barometer propwash | Not modelled | 0.3–1 m throttle-correlated pressure offset from rotor downwash |
| Magnetometer interference | Not modelled | Correlated with motor RPM, varies with airframe flex and nearby ferrous material |

Domain randomisation — training with widely varied noise parameters rather than a single fixed formula — is the standard mitigation, producing a network robust to distributional mismatch rather than one calibrated to the simulation's exact noise model.

### Training is teacher-forced, unlike the paper

For speed, training computes innovations against truth-anchored predictions
(`x_pred = F·x_true`), so the GRU never sees the innovation distribution produced by
its own imperfect estimates — a train/inference distribution shift. The original
KalmanNet paper instead backpropagates through the filter recursion itself (BPTT),
which is slower but exposes the network to its own errors. In practice the clipped
gain keeps inference stable here, but the teacher-forced network is optimistic; a
production implementation should train through the recursion.

### Barometer bias detection is transient

The GRU reduces K for baro when temperature-induced innovations are large and persistent. Once the Kalman filter's altitude estimate has been pulled toward the biased barometer reading over many steps, innovations shrink back toward zero — the bias is now baked into the state — and the GRU loses its corrective signal. A production system would include an explicit bias state in the filter's state vector.

### MAG is unaffected by environment

Magnetometer noise is fixed at σ = 3.0 m in this simulation — wind and temperature have no effect. The GRU has nothing to adapt for MAG; K stays near the Riccati value. The meaningful adaptation is on GPS (wind) and BARO (temperature). In real deployments with motor magnetic interference, the GRU would adjust K for MAG dynamically — the architecture supports it.

### Ground truth required for training

The training signal requires knowing the true drone state during the calibration flight. The simulation provides this for free via the analytical helix. Real deployments require a ground-truth reference system (RTK GPS, differential GPS, motion capture lab) during the calibration flight. This is the operational cost of KalmanNET deployment.

### Training coverage

The scripted calibration flight covers wind 0–20 m/s and temperature −10–50°C in discrete phases. Conditions not represented — simultaneous high wind and extreme cold, rapid step-changes, GPS outages during collection — are out-of-distribution at inference. Production training data would span the full joint operating envelope over many flights.

---

## AI and the path to realistic simulation

The biggest gap between this demo and real deployment is the quality of the simulation itself — Gaussian noise formulas are a poor substitute for the real, correlated, heavy-tailed behaviour of physical sensors. This is an active area of AI research.

### Learned sensor models

Rather than hand-writing noise formulas, generative models trained on real flight logs can learn the actual distribution of GPS multipath, IMU bias drift, and barometric propwash directly from data. A GAN or diffusion model trained on thousands of real drone flights could produce sensor readings statistically indistinguishable from hardware — including the correlation structure (e.g. GPS noise spiking as the drone passes a building) that a Gaussian model cannot capture. A Kalman-NET trained on data from such a model would generalise far better to deployment than one trained on this simulation.

### Neural digital twins

A **digital twin** is a real-time virtual replica of a physical system. Modern AI-powered twins (NVIDIA Omniverse, Microsoft Azure Digital Twins) go beyond static physics models — they embed learned neural components that update continuously as the real system runs. Applied to a drone, the twin would ingest real telemetry during flight, detect where its predictions diverge from reality, and update its sensor noise model accordingly. A Kalman-NET trained against a twin that is itself learning becomes progressively more calibrated to the actual deployed environment without requiring repeated calibration flights.

### Online adaptation without ground truth

This demo requires ground truth during training (the analytical helix). In the real world, ground truth is expensive. Physics-informed neural networks and self-supervised learning offer an alternative: the network uses **multi-sensor consistency** as a proxy signal — if GPS, IMU, and barometer all agree, the estimate is probably correct; if one disagrees persistently, it is probably biased. With this signal, a network could continue adapting during operational flights, closing the sim-to-real gap over time without any external reference system.

### Foundation models for sensor fusion

Large pre-trained models have begun to appear in the robotics and navigation space — models trained across diverse sensor modalities, platforms, and environments. A foundation model for inertial navigation, fine-tuned on a brief calibration flight, could replace both the simulation and the scripted training pipeline. The calibration flight provides the task-specific signal; the pre-trained weights carry the general understanding of how sensors behave. This would dramatically reduce the ground truth and data collection burden that currently makes KalmanNET deployment expensive.

---

## Software design for a hosted demo

Running a neural network training pipeline inside a browser-accessible web application imposes constraints that a production ML system would not face. The decisions below were made specifically to keep the demo self-contained, fast to start, and hostable on a single CPU instance.

### Model size — trainable on CPU in seconds

The GRU uses `hidden=64` and produces a model file of ~70 KB. This is deliberately small. The paper's architecture uses larger hidden dimensions; scaling to `hidden=256` would increase training time from ~30 seconds to over 10 minutes on a single CPU core — far too long for an interactive demo where a user clicks a button and waits. The 64-unit GRU is expressive enough to demonstrate the adaptive gain mechanism on the simulation's noise patterns while remaining trainable within a reasonable wait.

### Training speed — vectorisation and subsampling

The raw calibration buffer contains ~5,800 steps. Processing these sequentially in a Python loop (the naive approach) produces ~23,000 individual PyTorch GRU calls per epoch — approximately 28 minutes for 200 epochs. Two optimisations bring this to ~32 seconds:

- **10× subsampling** reduces the GRU sequence length from 5,800 to 580 steps. Temporal patterns in sensor noise operate on timescales of seconds, not milliseconds, so every 10th step retains the relevant signal.
- **Teacher-forced vectorisation** replaces the sequential per-step Python loop with a single batched GRU call over all 580 steps, followed by vectorised matrix multiplications (`torch.bmm`) to compute all state corrections in parallel. PyTorch's BLAS kernels handle this in one pass rather than 580 individual calls.

### Serving — Flask SSE, single process

The simulation state is broadcast to the browser at 20 Hz over a **Server-Sent Events** (SSE) connection — a persistent HTTP response that the server writes to continuously. SSE is unidirectional (server → browser only), which is all this application needs. It requires no extra library, no handshake protocol, and no client-side connection management beyond `new EventSource()`. WebSockets would add complexity for no benefit.

The Flask application runs as a **single process with multiple threads**. This is a hard requirement: the simulation state (`_state`, `_paused`, `_training_state`) lives in Python global variables protected by threading locks. Multiple worker processes would each maintain independent copies of this state — each user would effectively be looking at a different simulation. A single process with `--threads 8` serves concurrent SSE connections and API requests without this problem.

### Training execution — in-process thread, not subprocess

Training runs in a daemon thread within the same Python process rather than a child subprocess. The practical reason is memory: a subprocess would start a second Python interpreter, load PyTorch a second time (~230 MB), and run alongside the main process — approximately 460 MB peak during training, which exceeds the free tier of most hosting platforms. A thread shares the parent process's PyTorch runtime, keeping peak memory at ~360 MB. PyTorch releases the GIL during its C++ compute operations, so the SSE stream and Flask routes remain responsive throughout training. Progress updates write directly to `_training_state` each epoch — no stdout parsing needed.

### Dependencies — CPU-only PyTorch, no build step

The GPU-enabled PyTorch wheel is ~2.5 GB. The CPU-only wheel is ~200 MB. For a 70 KB model running 580-step sequences, there is no performance case for GPU — the data simply does not saturate a GPU's parallelism. The CPU-only install keeps the deployment footprint small and eliminates CUDA driver dependencies.

The frontend uses **Three.js loaded from CDN** with vanilla JavaScript and CSS — no bundler, no Node.js, no build step. The entire application starts with `python app.py`. This makes the demo trivially deployable on any machine with Python and avoids the maintenance burden of a JavaScript build pipeline for what is fundamentally a Python project.

---

## References

Revach, G., Shlezinger, N., Ni, X., Escoriza, A. L., van Sloun, R. J. G., & Eldar, Y. C. (2022). *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics.* IEEE Transactions on Signal Processing.
https://pure.tue.nl/ws/files/201346390/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics.pdf
