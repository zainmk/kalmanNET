# KalmanNET Implementation — Considerations

## What KalmanNET replaces

The standard Kalman filter computes its gain analytically:

```
K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹
```

This requires fixed, pre-calibrated R (measurement noise) and Q (process noise) matrices.
KalmanNET replaces this closed-form gain with a GRU that outputs K directly from the
history of innovations `ε = z − Hx̂`. The motion model F and observation model H remain;
only the gain computation is learned.

---

## What the GRU needs to learn

The GRU reads the innovation sequence over time and implicitly infers:

- When sensor noise has increased (wind → GPS/IMU R should rise → K should shrink)
- When a sensor has a systematic bias (temperature → baro consistently wrong in one direction)
- When a sensor has failed (innovations from that sensor disappear; others shift)
- Combinations of the above

It does not need to be told what the environment is — it infers from innovation patterns.

---

## Training protocol

For the GRU to generalise at test time, training must expose it to the innovation
signatures of every condition it will later encounter. If a condition was not in training,
KalmanNET will not outperform the standard filter on it.

| Phase | Duration | Conditions | Innovation signature learned |
|---|---|---|---|
| Baseline | 10 s | All sensors, 0 wind, 20°C | Clean per-sensor statistics |
| Wind stress | 30 s | 8–12 m/s wind, all sensors | GPS/IMU noise inflation |
| Temperature stress | 20 s | 40–45°C, all sensors | Persistent baro bias direction |
| Single failure | 15 s | GPS off, moderate wind | Remaining sensor shift on dropout |
| Combined stress | 15 s | Wind + temperature + one failure | Compounded degradation |

**Minimum: ~90 s. Recommended: ~120 s.**

The distribution constraint is strict: train at wind 10 m/s, test at 15 m/s — partial
generalisation. Train at 0 wind, test at 15 m/s — no advantage over standard Kalman.
Set sliders to target stress levels *before* training starts.

---

## Test conditions that show the largest gap

After training under the above protocol, these scenarios produce the sharpest divergence
between KalmanNET and the standard Kalman-only filter:

**1. High wind → GPS failure**
Standard filter over-trusted GPS while it was noisy (K too large). On GPS dropout it snaps
to a bad prior. KalmanNET already reduced GPS weight → smoother transition, smaller error spike.

**2. Sustained temperature deviation → barometer failure**
Standard filter followed the biased baro reading throughout. On baro failure its altitude
estimate visibly corrects — revealing accumulated error. KalmanNET learned the baro was
consistently directionally wrong and discounted it → altitude was already more accurate.

**3. Wind + temperature simultaneously, then fail two sensors**
Standard filter compounds two miscalibrated R matrices with sudden data loss.
KalmanNET adapted both gains during training → handles degraded sensor suite from a
better-calibrated prior.

---

## Architecture notes

- **Input**: innovation sequence `ε₁, ε₂, …, εₜ` (one vector per active sensor per step)
- **Output**: Kalman gain K (replaces the analytic formula entirely)
- **Why GRU over LSTM**: Simpler (2 gates vs 3), trains faster, performs comparably for the
  sequence lengths involved here (tens to hundreds of steps). KalmanNET paper uses GRU.
- **Training**: offline via backpropagation through time (BPTT) on recorded flight data.
  Inference runs at real-time speed with no additional warmup.
- **Sensor failures during inference**: the GRU input should include a failure mask so the
  network knows which sensors are active, rather than inferring absence from zero innovations.

---

## Current sim parameters (what the standard Kalman assumes — forever)

| Sensor | Fixed σ (calm) | Fixed R | Actual σ at max stress |
|---|---|---|---|
| GPS | 2.0 m | diag(4, 4, 4) | ~4.4 m at wind 20 m/s |
| IMU | 0.5 m/s | diag(0.25, 0.25, 0.25) | ~1.3 m/s at wind 20 m/s + 30°C dev |
| Baro | 0.5 m | [0.25] | ~0.74 m + 3.6 m bias at 50°C |
| Mag | 3.0 m | diag(9, 9) | unchanged (not modelled as env-sensitive) |

Calibration temperature: **20°C**. Baro bias formula: `0.12 × (T − 20)` metres.
