# KalmanNET Demo

A fullstack Flask + Three.js simulation that motivates the KalmanNET paper
(Revach et al., 2022) by showing — in real time — where a fixed-R Kalman filter
breaks down and where an adaptive R-prediction network recovers.

---

## Running

```
pip install flask numpy
python app.py
```

Open `http://localhost:5000`.  Press **▶ PLAY** or `Space` to start the
simulation.  Press `R` to reset.

---

## Architecture

| Layer | Technology |
|---|---|
| Simulation & filter | Python — `drone_sim.py`, `kalman_filter.py`, `kalmannet.py` |
| HTTP / SSE server | Flask (`app.py`) |
| 3-D visualisation | Three.js r128 |
| UI | Vanilla JS + CSS (`static/`) |

The simulation runs at 20 Hz in a background thread.  State is pushed to the
browser over a persistent SSE connection (`/stream`) as JSON.  The browser
applies each frame to the Three.js scene without polling.

---

## KalmanNET — design decisions

### Why R-prediction instead of direct gain prediction

The original KalmanNET paper (Revach et al.) replaces the Kalman gain
`K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹` with a GRU that outputs K directly — R never
appears at inference time.

This demo instead predicts **log(R_diagonal)** per sensor from the recent
innovation window, then feeds the adapted R into the standard gain formula.
Reasons:

1. **Interpretability.** The R̂ values are shown live in the sidebar.  The user
   can watch GPS R̂ climb from 4 → ~14 as wind rises and see *why* the KN
   estimate is more stable.
2. **Narrative continuity.** Every tooltip and the KF badge already explain R
   as the key tunable quantity.  Predicting R is a natural extension of that
   framing rather than a black-box gain replacement.
3. **Simpler training.** R has a concrete oracle target (the known simulation
   noise formula) so training is a straightforward supervised regression.
   Deriving a oracle K target requires solving a more complex optimisation.

The trade-off: this is not a strict implementation of the paper.  It is better
described as an *R-adaptive Kalman filter trained by a neural network*, which
demonstrates the same motivating insight.

### Why simulation-based calibration training

The network is trained on a scripted flight that varies wind speed and
temperature across the full operating envelope.  This mirrors how a real
KalmanNET deployment would work: a **calibration flight** is performed in a
controlled environment where ground truth is available (differential GPS, motion
capture, etc.), the network trains on that data, and inference runs on
subsequent operational flights without ground truth.

In this simulation ground truth is always available (the analytical helix), so
the calibration flight is free.  In production that is the hard part.

### Why a numpy MLP instead of PyTorch / TensorFlow

A two-layer MLP with ~3 000 total parameters trains in under 15 seconds on CPU
with vectorised numpy.  Adding PyTorch would introduce a ~500 MB dependency,
a framework import overhead, and CUDA/CPU fallback logic — none of which buys
anything at this scale.  The manual backward pass (`_MLP.step`) is fewer than
20 lines and is easy to audit.

### Training oracle targets

During the calibration flight, the simulation exposes the exact noise formula
for each sensor:

| Sensor | Effective σ |
|---|---|
| GPS | `2.0 + 0.12·wind` m |
| IMU | `0.5 + 0.04·wind + 0.008·\|ΔT\|` m/s |
| BARO | `√( (0.5 + 0.008·\|ΔT\|)² + (ΔT·0.12)² )` m (noise + bias) |
| MAG | `3.0` m (constant) |

`R_target = diag(σ²)`.  The network is trained to predict `log(R_target)` from
the preceding 20 innovations (1 second of history at 20 Hz).

---

## Limitations at demo scale

### Network expressiveness
A 2-layer MLP with 16 hidden units is sufficient for the smooth, parametric
noise curves in this simulation.  Real sensor noise is non-stationary,
multimodal, and correlated across axes.  A deeper or recurrent architecture
(the GRU in the original paper) would be required in practice.

### Training coverage
The scripted 2.5-minute calibration flight covers wind 0–20 m/s and temperature
−10–50°C in isolation and in one combined-stress scenario.  Any operating point
not covered risks out-of-distribution predictions.  In production, training data
would be collected over many flights spanning the full joint operating envelope.

### MAG is unaffected by environment
Magnetometer noise is constant in this simulation (`σ = 3.0 m` regardless of
wind or temperature).  The MAG network therefore learns to output a constant
R̂ ≈ diag(9, 9).  The interesting adaptation happens on GPS, IMU, and — most
visibly — BARO under temperature stress.

### No cross-sensor architecture
Each sensor has an independent MLP.  The original KalmanNET uses a single
recurrent network that sees all innovations together, allowing it to learn
cross-sensor correlations (e.g., "GPS and IMU both show high scatter → must be
wind").  The independent-sensor approach here cannot exploit those correlations.

### Baro bias detection is transient
When temperature changes, baro innovations spike (large positive or negative
mean) and the network inflates R_baro, correctly resisting the bias.  But once
the Kalman filter's altitude estimate has been pulled toward the biased reading
over many steps, innovations shrink back toward zero and the network loses its
signal.  A production system would use a bias estimator in the state vector or a
cross-sensor consistency check to detect persistent offsets.

### Ground truth required for training
This is not a limitation of the demo specifically but of KalmanNET as a concept.
The supervised training signal requires knowing the true state during
calibration.  The demo has this for free; real deployments do not.

---

## References

Revach, G., Shlezinger, N., Ni, X., Escoriza, A. L., van Sloun, R. J. G., &
Eldar, Y. C. (2022). *KalmanNet: Neural Network Aided Kalman Filtering for
Partially Known Dynamics.*  IEEE Transactions on Signal Processing.
https://pure.tue.nl/ws/files/201346390/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics.pdf
