'use strict';

const TOOLTIPS = {

  // Headers ───────────────────────────────────────────────────────────

  'badge-helix': `The drone's true programmed path is a rising helix with radius 20 m, angular speed 0.3 rad/s, and vertical climb of 1 m/s. This serves as the analytic ground truth and lets us compare error accumulation in the filter at every timestep.<br><br><b>Assumption:</b> the autopilot recovers from wind push at a fixed 3% per step regardless of wind magnitude — a real autopilot would fight harder at low speeds and risk being overwhelmed at high speeds. Wind only displaces the drone horizontally; vertical gusts are not modelled.`,

  'badge-kn': `<b>KalmanNET</b> — Neural Network-Aided Kalman Filtering<br><br>
    Replaces the fixed R matrix with a per-sensor MLP that predicts the effective measurement noise covariance from a sliding window of recent innovations. Trained on a scripted calibration flight where ground truth is known; at inference only the live innovation sequence is used — no explicit environment state required.<br><br>
    Click to open the original paper (Revach et al., 2022).`,

  'badge-kf': `<b>Purpose</b><br>
    The Linear Kalman Filter is used to optimally fuse the data from multiple noisy sensors into a single estimate — the minimum-variance solution to when multiple sensors are observing the same underlying state and accuracy of data is required.<br><br>
    <b>Predict</b> — advance state using the motion model:<br>
    &nbsp;x̂ ← F·x̂ &nbsp;&nbsp; (where the drone is expected to be)<br>
    &nbsp;P ← F·P·Fᵀ + Q &nbsp;&nbsp; (uncertainty grows without corrections)<br><br>
    <b>Update</b> — correct using each active sensor reading z:<br>
    <span style="display:block;margin:6px 0 2px;padding:5px 9px;background:rgba(30,55,130,0.45);border-left:2px solid #4477dd;border-radius:3px;font-family:monospace;color:#aaccff;text-align:center">K = P·Hᵀ·(H·P·Hᵀ + <b style="color:#fff">R</b>)⁻¹ &nbsp;<span style="color:#5577aa;font-size:9px">← Kalman gain</span></span>
    <span style="display:block;margin:2px 0 6px;padding:5px 9px;background:rgba(30,55,130,0.45);border-left:2px solid #4477dd;border-radius:3px;font-family:monospace;color:#aaccff;text-align:center">x̂ ← x̂ + K·(z − H·x̂) &nbsp;<span style="color:#5577aa;font-size:9px">← state correction</span></span>
    <b>R — Measurement Noise Covariance</b><br>
    Encodes how much the kalman filter trusts each sensor. It is set once at calibration but never changed. Large R → sensor barely shifts the estimate. Small R → sensor dominates. Unaccountable environmental noise can lead to an inaccurate 'R' and degrade the estimate - <i> KalmanNET learns from its environment to assign 'R' appropriately, for a more accurate representation<i>.<br><br>
    <b>Assumptions of Linearity: </b><br> 
    We assume the following two assumptions; the motion model is linear - the drone flies in a constant velocity helix, and the sensor models are linear - each sensor observes a fixed subset of the state. 
    Extensions of the algorithm (EKF, UKF) exist to address non-linearities analytically - <i> KalmanNET addresses it by replacing this gain formula with a learned network instead </i>.`,



  // Sensor control panel ─────────────────────────────────────────────────────

  'panel-sensor-control': `No single sensor is sufficient. GPS drifts without velocity context. IMU accumulates position error without an anchor. Barometer only sees altitude. Magnetometer is too noisy to rely on alone. The Kalman filter 'fuses' these values to form a complete and robust position estimate. <br><br> The kalman filter suffers when the sensors are disabled, but it doesn't fail catastrophically — it falls back to dead-reckoning on the remaining sensors. Watch how the estimate degrades as you disable each sensor in turn.`,

  'btn-gps': `
    <b>GPS — absolute position [x, y, z]</b><br><br>
    <b><u>Purpose:</u></b> The only drift-free absolute 3D fix — the filter's primary position anchor. Without it, horizontal position falls back to the far noisier magnetometer.<br><br>
    Each fix scatters ±2 m. IMU velocity bridges between fixes so the estimate doesn't jump. Barometer independently validates the altitude component.<br><br>
    <b><u>Kalman-Filter:</u></b><br>
    &nbsp;H = [I₃ | 0₃] &nbsp; selects position rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×3 gain matrix<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; all 6 states corrected via cross-covariance in P<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;σ = 2.0 m &nbsp;|&nbsp; R = diag(4, 4, 4)<br>
    Environmental factors can cause the actual noise to differ from the assumed noise. Under wind 15 m/s: actual σ ≈ 3.8 m — filter still uses <b> R = diag(4, 4, 4) </b>.<br>
    The gain K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ is computed with the wrong R, so K is too large and the filter over-weights a now-noisier GPS fix — pulling the estimate off-course more than it should. The Kalman-NET learns to adjust R dynamically.<br><br>   
  `,
  'btn-imu': `
    <b>IMU — velocity [vx, vy, vz]</b><br><br>
    <b><u>Purpose:</u></b> Fast, continuous updates every step. Bridges the gaps between slow GPS fixes so position doesn't drift in between.<br><br>
    Measures velocity not position. Velocity errors integrate into unbounded position error over time. GPS corrects the accumulated drift. Without GPS, position grows unconstrained.<br><br>
    <b><u>Kalman-Filter:</u></b><br>
    &nbsp;H = [0₃ | I₃] &nbsp; selects velocity rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×3 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; position states also shift via off-diagonal terms in P<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;σ = 0.5 m/s &nbsp;|&nbsp; R = diag(0.25, 0.25, 0.25)<br>
    Wind 15 m/s: actual σ ≈ 1.1 m/s. Temp ±30°C: actual σ ≈ 0.74 m/s.<br>
    The wind speed affects the actual noise, and the Kalman-NET would detect the growing innovations and raise its effective gain as trained.`,

  'btn-baro': `
    <b>Barometer — altitude [z only]</b><br><br>
    <b><u>Purpose:</u></b> Cheapest and most reliable sensor for altitude. Anchors z independently of GPS — even when GPS fails, BARO + IMU together keep vertical estimates tight. Zero XY awareness; contributes nothing to horizontal position, which depends entirely on GPS and magnetometer.<br><br>
    <b><u>Kalman-Filter:</u></b><br>
    &nbsp;H = [0, 0, 1, 0, 0, 0] &nbsp; selects altitude (z) row of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; scalar innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×1 gain vector<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; scalar correction propagated to all 6 states via K<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;σ = 0.5 m &nbsp;|&nbsp; R = [0.25] &nbsp;|&nbsp; calibrated at 20°C, zero bias<br>
    Temperature introduces two compounding errors. At 50°C (dT = +30): actual σ ≈ 0.74 m <i>and</i> a persistent −3.6 m altitude bias (hot air has lower pressure — baro reads drone as lower than it is). At −10°C (dT = −30): +3.6 m bias (cold air reads high). The filter trusts the biased reading at full weight because R is small and fixed — it cannot distinguish systematic offset from random noise. KalmanNET learns the relationship between temperature and effective baro error, inflating R when temperature deviates from calibration so the filter down-weights a persistently wrong measurement rather than anchoring altitude to it.`,

  'btn-mag': `
    <b>Magnetometer — horizontal position [x, y]</b><br><br>
    <b><u>Purpose:</u></b> Independent XY fix that doesn't rely on satellite signal. When GPS fails, MAG is the only sensor preventing horizontal position from going unbounded — even if noisily. High R means the filter assigns it low Kalman gain, so GPS naturally dominates when healthy. Zero altitude awareness; barometer covers the z-axis entirely.<br><br>
    <b><u>Kalman-Filter:</u></b><br>
    &nbsp;H = [1,0,0,0,0,0; 0,1,0,0,0,0] &nbsp; selects x and y position rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 2×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp; 6×2 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; gentle correction — large R keeps gain small, so each fix barely moves the estimate<br><br>
    <b><u>Kalman-NET:</u></b><br>
    &nbsp;σ = 3.0 m &nbsp;|&nbsp; R = diag(9, 9)<br>
    In this simulation MAG noise is fixed — wind and temperature do not affect it, so there is no R mismatch to observe here. In real deployments, magnetic interference from motors, nearby structures, and dynamic environments causes effective noise to vary widely. KalmanNET would learn to adjust R in response to detected interference patterns — tightening gain when MAG is momentarily reliable, loosening it when interference spikes, rather than holding R constant at a conservative worst-case value.`,

  // ── Environment overlay ──────────────────────────────────────────────────────

  'env-title': `Environmental conditions that stress the sensor suite beyond the Kalman filter's calm-condition assumptions. The filter's R matrices never update — the growing mismatch between assumed and actual noise is exactly what KalmanNET would learn to correct.`,

  'env-reset-btn': `Reset to Kalman calibration defaults: 0 m/s wind, 0° heading, 20°C. These are the exact conditions the filter's R matrices were tuned for — where it performs optimally.`,

  'env-wind-speed-label': `Physically deflects the true trajectory (green line). Also raises GPS noise via multipath and turbulence (actual σ = 2 + 0.12·w m) and IMU noise via vibration (actual σ = 0.5 + 0.04·w m/s). Kalman filter always assumes calm-condition R — it does not know wind has increased.`,

  'env-wind-heading-label': `Direction the wind blows, in degrees clockwise from East. Only has an effect when Wind Speed > 0. The drone's autopilot partially counteracts the push — the true path bows rather than drifts without bound.`,

  'env-temp-label': `Sensors are calibrated at 20°C. Deviations cause IMU thermal drift (actual σ = 0.5 + 0.008·|ΔT| m/s) and a barometer altitude bias (offset = −0.12·ΔT m — hot air has lower pressure, so the baro reads the drone as lower than it is). Kalman filter assumes 20°C at all times.`,




  // ── Sidebar panels ───────────────────────────────────────────────────────────

  'panel-sensor-readings': `Raw: the noisy measurement received from the sensor this timestep. KF Est: the Kalman filter's current best estimate of the same quantity, fused from all active sensors.`,

  'panel-error-title': `Plots position error (distance between Kalman estimate and ground truth) over the last 200 timesteps. Colour shifts green → amber → red as error grows. Watch it spike when sensors fail and the filter falls back to dead-reckoning.`,



  
  // ── Bottom legend ────────────────────────────────────────────────────────────

  'legend-true':       `The analytically computed ground-truth helix. In real flight this is unknown — shown here only to benchmark the filter's accuracy.`,
  'legend-raw':        `Raw sensor position — direct readings with no filtering. GPS x/y/z when available, falling back to magnetometer (XY) and barometer (Z). Shows exactly how noisy unfiltered sensing is. IMU is excluded because it measures velocity, not position.`,
  'legend-kalman':     `The Kalman filter's fused position estimate drawn as a trail. Compare its path to the green truth line to see where the filter lags or deviates.`,
  'legend-true-drone': `The semi-transparent green wireframe shows the drone's true position. The gap between this and the white drone is the current position error.`,
  'legend-est-drone':  `The white drone mesh shows the Kalman filter's current best estimate of position. Its rotors spin at the estimated angular rate.`,
  'legend-gps':        `Each yellow dot is a raw GPS fix. The scatter represents ±2 m sensor noise the filter must average out.`,
  'legend-imu':        `The orange arrow shows the IMU's measured velocity vector. Arrow direction is heading; length scales with speed. Disappears when the IMU is failed.`,
  'legend-baro':       `The pink dot and horizontal ring mark the barometer's measured altitude plane. The ring stays flat — barometers have no XY awareness.`,
  'legend-mag':        `The purple dot shows the magnetometer's raw XY position reading. It is noisier than GPS (±3 m) — notice the wider scatter around the true path.`,
  'legend-unc':        `The translucent blue sphere represents 1σ position uncertainty from the covariance matrix P. Inflates as sensors fail; shrinks when they recover.`,

};

document.addEventListener('DOMContentLoaded', () => {
  Object.entries(TOOLTIPS).forEach(([id, content]) => {
    const el = document.getElementById(id);
    if (el) el.dataset.tooltip = content.trim();
  });
});
