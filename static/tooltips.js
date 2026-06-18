'use strict';

const TOOLTIPS = {

  // ── Header badges ────────────────────────────────────────────────────────────

  'badge-helix': `The drone's true programmed path is a rising helix with radius 20 m, angular speed 0.3 rad/s, and vertical climb rate 1 m/s. This is the analytic ground truth — in real flight it would be unknown. Shown here only to benchmark filter accuracy at every timestep.<br><br>
    <b>Wind model:</b> wind accumulates as an exponential moving average of force applied each step (decay 3% per step), displacing the true trajectory horizontally. The autopilot does not counteract it — the drone is physically pushed. Vertical gusts are not modelled.`,

  'badge-kn': `<b>Kalman-NET</b> — Neural Network-Aided Kalman Filtering<br><br>
    Replaces the Kalman gain formula K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ with a GRU network that predicts K directly from the recent history of innovations — the gap between what sensors measured and what the filter expected.<br><br>
    Unlike R-inflation, the GRU can <i>increase</i> K above the Riccati value — allowing the filter to trust a sensor <i>more</i> than the standard formula, not just less. This is necessary when wind displaces the true drone position: the filter must increase trust in GPS to track the displaced trajectory, not reduce it.<br><br>
    Trained once on a scripted calibration flight where ground truth is known. At inference only live sensor data is needed — no explicit environment state required.<br><br>
    Click to open the original paper (Revach et al., 2022).`,

  'badge-kf': `<b>Purpose</b><br>
    The Linear Kalman Filter optimally fuses data from multiple noisy sensors into a single estimate — the minimum-variance solution when several sensors observe the same underlying state.<br><br>
    <b>Predict</b> — advance state using the motion model:<br>
    &nbsp;x̂ ← F·x̂ &nbsp;&nbsp; (where the drone is expected to be)<br>
    &nbsp;P ← F·P·Fᵀ + Q &nbsp;&nbsp; (uncertainty grows without corrections)<br><br>
    <b>Update</b> — correct using each active sensor reading z:<br>
    <span style="display:block;margin:6px 0 2px;padding:5px 9px;background:rgba(30,55,130,0.45);border-left:2px solid #4477dd;border-radius:3px;font-family:monospace;color:#aaccff;text-align:center">K = P·Hᵀ·(H·P·Hᵀ + <b style="color:#fff">R</b>)⁻¹ &nbsp;<span style="color:#5577aa;font-size:9px">← Kalman gain</span></span>
    <span style="display:block;margin:2px 0 6px;padding:5px 9px;background:rgba(30,55,130,0.45);border-left:2px solid #4477dd;border-radius:3px;font-family:monospace;color:#aaccff;text-align:center">x̂ ← x̂ + K·(z − H·x̂) &nbsp;<span style="color:#5577aa;font-size:9px">← state correction</span></span>
    <b>R — Measurement Noise Covariance</b><br>
    Encodes how much the filter trusts each sensor. Set once at calibration and never changed. Large R → small K → sensor barely shifts the estimate. Small R → large K → sensor dominates. When the environment degrades sensor quality, R is wrong — and so is K.<br><br>
    <b>Linearity assumptions:</b><br>
    The motion model is linear (constant-velocity helix). The sensor models are linear (each sensor observes a fixed subset of the 6-D state). Extensions like EKF and UKF handle non-linearities analytically — Kalman-NET addresses them by replacing the gain formula with a learned network.`,


  // ── Sensor control panel ─────────────────────────────────────────────────────

  'panel-sensor-control': `No single sensor is sufficient. GPS drifts without velocity context. IMU accumulates position error without an anchor. Barometer only sees altitude. Magnetometer is too noisy to rely on alone. The Kalman filter fuses these values into a complete, robust position estimate.<br><br>
    Disabling sensors degrades the estimate but does not cause catastrophic failure — the filter falls back to dead-reckoning on the remaining sensors. Watch how the estimate degrades as you disable each sensor in turn.`,

  'btn-gps': `
    <b>GPS — absolute position [x, y, z]</b><br><br>
    <b><u>Purpose:</u></b> The only drift-free absolute 3-D fix. The filter's primary position anchor. Without it, horizontal position falls back to the far noisier magnetometer.<br><br>
    Each fix scatters ±2 m. IMU velocity bridges between fixes. Barometer independently validates the altitude component.<br><br>
    <b><u>Kalman Filter:</u></b><br>
    &nbsp;H = [I₃ | 0₃] &nbsp; selects position rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×3 gain matrix<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; all 6 states corrected via cross-covariance in P<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;Calibration: σ = 2.0 m &nbsp;|&nbsp; R = diag(4, 4, 4)<br>
    Wind physically displaces the true drone position (the green line shifts). GPS correctly measures the displaced position, but the Kalman filter's motion model predicts the undisturbed helix — so GPS innovations grow with every step. With fixed R and the standard K, the filter under-corrects, always lagging behind where the drone actually is.<br><br>
    Kalman-NET sees these growing, sustained innovations and <i>increases</i> K for GPS above the Riccati value — pulling the estimate aggressively toward GPS to track the displaced trajectory. This is the scenario R-inflation cannot handle: the filter needs to trust GPS <i>more</i>, not less.`,

  'btn-imu': `
    <b>IMU — velocity [vx, vy, vz]</b><br><br>
    <b><u>Purpose:</u></b> Fast, continuous updates every step. Bridges the gaps between GPS fixes so position doesn't drift in between. Measures velocity, not position — velocity errors integrate into unbounded position error over time without GPS to correct them.<br><br>
    <b><u>Kalman Filter:</u></b><br>
    &nbsp;H = [0₃ | I₃] &nbsp; selects velocity rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×3 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; position states also shift via off-diagonal terms in P<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;Calibration: σ = 0.5 m/s &nbsp;|&nbsp; R = diag(0.25, 0.25, 0.25)<br>
    Wind creates two competing effects on IMU. Noise increases (σ = 0.5 + 0.04·w m/s) suggesting the filter should reduce K. But wind also shifts the true velocity — IMU carries real information about the wind-displaced motion. Temperature additionally raises thermal drift (σ = 0.5 + 0.008·|ΔT| m/s).<br><br>
    Kalman-NET learns the balance: the GRU adjusts K for IMU based on the innovation pattern it has observed, tuning trust in velocity readings to minimise overall position error — something a fixed R cannot adapt to.`,

  'btn-baro': `
    <b>Barometer — altitude [z only]</b><br><br>
    <b><u>Purpose:</u></b> The cheapest and most reliable sensor for altitude. Anchors z independently of GPS — even when GPS fails, BARO + IMU together keep vertical estimates tight. Zero XY awareness; contributes nothing to horizontal position.<br><br>
    <b><u>Kalman Filter:</u></b><br>
    &nbsp;H = [0, 0, 1, 0, 0, 0] &nbsp; selects altitude (z) row of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; scalar innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×1 gain vector<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; scalar correction propagated to all 6 states via K<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;Calibration: σ = 0.5 m &nbsp;|&nbsp; R = [0.25] &nbsp;|&nbsp; calibrated at 20°C, zero bias<br>
    Temperature introduces two compounding errors. At 50°C (dT = +30): noise σ ≈ 0.74 m <i>and</i> a persistent <b>−7.5 m altitude bias</b> (hot air is less dense — lower pressure — baro reads drone as lower than it is). At −10°C (dT = −30): <b>+7.5 m bias</b> (cold air reads high).<br><br>
    The filter trusts this biased reading at full weight because R is small and fixed — it cannot distinguish a systematic offset from random noise. Kalman-NET detects the sustained, one-directional innovations and <i>reduces</i> K for baro, down-weighting the faulty altimeter and anchoring altitude through GPS instead.`,

  'btn-mag': `
    <b>Magnetometer — horizontal position [x, y]</b><br><br>
    <b><u>Purpose:</u></b> Independent XY fix that doesn't rely on satellite signal. When GPS fails, MAG is the only sensor preventing horizontal position from going unbounded — even if noisily. High R means the filter assigns it low Kalman gain, so GPS naturally dominates when healthy. Zero altitude awareness.<br><br>
    <b><u>Kalman Filter:</u></b><br>
    &nbsp;H = [1,0,0,0,0,0; 0,1,0,0,0,0] &nbsp; selects x and y rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 2×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×2 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; gentle correction — large R keeps gain small<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;Calibration: σ = 3.0 m &nbsp;|&nbsp; R = diag(9, 9)<br>
    MAG noise is fixed in this simulation — wind and temperature do not affect it — so innovations stay stationary and the GRU has nothing to adapt. K stays near the Riccati value.<br><br>
    In real deployments, motor magnetic interference, nearby ferrous structures, and dynamic environments cause effective noise to vary widely with RPM and airframe orientation. The GRU architecture supports adapting K for MAG dynamically — this simulation simply lacks the excitation to demonstrate it.`,


  // ── Environment overlay ──────────────────────────────────────────────────────

  'env-title': `Environmental conditions that push the sensor suite beyond the Kalman filter's calm-condition assumptions. The filter's R matrices and gain K are calibrated at 0 m/s wind, 20°C — any deviation creates a mismatch between assumed and actual sensor behaviour that Kalman-NET learns to correct.`,

  'env-reset-btn': `Reset to Kalman calibration defaults: 0 m/s wind, 0° heading, 20°C. These are the exact conditions the filter's R matrices were tuned for — where it performs optimally.`,

  'env-wind-speed-label': `Physically deflects the true trajectory (green line). Wind force accumulates each step and decays at 3% per step — the drone is pushed, not corrected. Also raises GPS noise via multipath and turbulence (σ = 2 + 0.12·w m) and IMU noise via vibration (σ = 0.5 + 0.04·w m/s). The Kalman filter always assumes calm-condition R — it does not know wind has increased.`,

  'env-wind-heading-label': `Direction the wind blows, in degrees clockwise from East. Only has an effect when Wind Speed > 0.`,

  'env-temp-label': `Sensors are calibrated at 20°C. Deviations cause IMU thermal drift (σ = 0.5 + 0.008·|ΔT| m/s) and a barometer altitude bias (offset = −0.25·ΔT m — hot air has lower pressure, so the baro reads the drone as lower than it actually is; cold air reads high). The Kalman filter assumes 20°C at all times and cannot detect either effect.`,


  // ── Sidebar panels ───────────────────────────────────────────────────────────

  'panel-sensor-readings': `Raw: the noisy measurement received from the sensor this timestep. KF Est: the Kalman filter's current best estimate of that quantity, fused from all active sensors. KN Est: the Kalman-NET estimate (shown only after training).`,


  // ── Bottom legend ────────────────────────────────────────────────────────────

  'legend-true':       `The analytically computed ground-truth helix (plus any wind displacement). In real flight this is unknown — shown here only to benchmark filter accuracy.`,
  'legend-raw':        `Raw sensor position — direct readings with no filtering. GPS x/y/z when available, falling back to magnetometer (XY) and barometer (Z). Shows exactly how noisy unfiltered sensing is. IMU is excluded because it measures velocity, not position.`,
  'legend-kalman':     `The Kalman filter's fused position estimate as a trail. Compare its path to the green truth line to see where the filter lags or drifts under environmental stress.`,
  'legend-kn':         `The Kalman-NET estimate trail (cyan). Visible after training. Compare to the blue Kalman trail to see where the learned gain produces a tighter estimate.`,
  'legend-true-drone': `The green wireframe shows the drone's true position at this timestep. The gap between this and the estimate drone is the current position error.`,
  'legend-est-drone':  `The estimate drone: white when tracking the Kalman filter, cyan when tracking Kalman-NET (after training). When Kalman-NET is active, a translucent blue silhouette marks where the Kalman filter would be.`,
  'legend-gps':        `Each yellow dot is a raw GPS fix. The scatter represents ±2 m sensor noise the filter must average out. Scatter increases with wind speed.`,
  'legend-imu':        `The orange arrow shows the IMU's measured velocity vector. Direction is heading; length scales with speed. Disappears when IMU is failed.`,
  'legend-baro':       `The pink dot and horizontal ring mark the barometer's measured altitude plane. The ring stays flat — barometers have no XY awareness. Under temperature stress the ring drifts above or below the true altitude.`,
  'legend-mag':        `The purple dot shows the magnetometer's raw XY position reading. It is noisier than GPS (±3 m) — notice the wider scatter. MAG provides XY backup when GPS fails.`,
  'legend-unc':        `The translucent blue sphere represents 1σ position uncertainty from the covariance matrix P. Inflates as sensors fail; shrinks when they recover.`,

};

document.addEventListener('DOMContentLoaded', () => {
  Object.entries(TOOLTIPS).forEach(([id, content]) => {
    const el = document.getElementById(id);
    if (el) el.dataset.tooltip = content.trim();
  });
});
