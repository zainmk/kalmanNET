# Kalman-NET Demo

A simulation showing where classical sensor fusion breaks down in changing environments — and how Kalman-NET, a recurrent neural network, learns to compensate.



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

**Why direct-K matters.** R-inflation (the alternative approach) can only push K *down* — it cannot trust a sensor more than the formula allows. But environments create both problems:

| Scenario | What the filter needs | R-inflation | Direct-K |
|---|---|---|---|
| Wind displaces the drone | Trust GPS *more* to follow it | ✗ | ✓ |
| Temperature biases the barometer | Trust baro *less* to ignore it | ✓ | ✓ |

**How the GRU works.** All four sensors' innovations are combined into a single 9-dimensional vector each step. The GRU processes this and outputs a K matrix for each sensor — the shared network can exploit cross-sensor patterns (e.g. GPS and IMU both showing large innovations simultaneously signals wind, not just individual sensor noise).

**Training.** The network is trained once on a scripted calibration flight that varies wind speed and temperature across the full operating envelope. The training signal is position error — the network learns whatever K adjustments minimise how far the estimate drifts from the known true trajectory. At inference, no ground truth is needed: only live sensor data.

---

## Running

```bash
pip install flask numpy torch --index-url https://download.pytorch.org/whl/cpu
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

## Design decisions

### Why a GRU predicting K directly

An earlier version predicted R per-sensor using an MLP. Two problems motivated the switch:

**K can go in either direction.** R-inflation reduces K — the filter trusts the sensor less. Wind displacement requires the filter to trust GPS *more* to track the displaced true position. This is impossible with R-inflation alone. Direct-K prediction has no such constraint.

**Temporal context is load-bearing.** An MLP operating on a fixed-size innovation window cannot reliably distinguish transient noise spikes from sustained environmental drift. The GRU carries hidden state across the full flight, allowing it to recognise that GPS and IMU innovations growing *together* over many steps means wind — not simultaneous sensor failure — and respond with a sustained K adjustment.

### Why in-process training, not a subprocess

Training runs in the `_run_training` daemon thread rather than a child process. PyTorch releases the GIL during its C++ compute operations, so Flask's SSE stream and route handlers remain responsive throughout. The simulation is already paused during the optimisation phase, so there is no competing compute load. A subprocess would instantiate two independent Python runtimes, each loading the full PyTorch library (~230 MB), doubling peak memory during training. The in-process approach also enables real-time epoch progress updates via direct state mutation — no stdout parsing needed.

### Why simulation-based calibration training

The network trains on a scripted flight that covers the full operating envelope in controlled conditions. This mirrors a real KalmanNET deployment: a calibration flight is performed where ground truth is available (differential GPS, motion capture), the network trains on that data, and inference runs on subsequent operational flights without ground truth. In this simulation ground truth is always available (the analytical helix), so the calibration flight is free.

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
