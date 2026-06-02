'use strict';

const TOOLTIPS = {

  // Headers ───────────────────────────────────────────────────────────

  'badge-helix': `The drone's true programmed path is a rising helix with radius 20 m, angular speed 0.3 rad/s, and vertical climb of 1 m/s. This serves as the analytic ground truth and lets us compare error accumulation in the filter at every timestep.<br><br><b>Assumption:</b> the autopilot recovers from wind push at a fixed 3% per step regardless of wind magnitude — a real autopilot would fight harder at low speeds and risk being overwhelmed at high speeds. Wind only displaces the drone horizontally; vertical gusts are not modelled.`,

  'badge-kf': `<b>Purpose</b><br>
    Optimally fuses multiple noisy sensors into a single state estimate — the minimum-variance solution to "what is the drone's true position given all these imperfect readings?" Used in GPS navigation, aerospace, and robotics wherever multiple sensors observe the same underlying state.<br><br>
    <b>Predict</b> — advance state using the motion model:<br>
    &nbsp;x̂ ← F·x̂ &nbsp;&nbsp; (where the drone is expected to be)<br>
    &nbsp;P ← F·P·Fᵀ + Q &nbsp;&nbsp; (uncertainty grows without corrections)<br><br>
    <b>Update</b> — correct using each active sensor reading z:<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + <b>R</b>)⁻¹ &nbsp;&nbsp; (Kalman gain)<br>
    &nbsp;x̂ ← x̂ + K·(z − H·x̂) &nbsp;&nbsp; (nudge estimate toward reading)<br><br>
    <b>R — measurement noise covariance</b><br>
    Encodes how much the filter trusts each sensor, set once at calibration and never changed. Large R → small K → sensor barely shifts the estimate. Small R → large K → sensor dominates. If real noise exceeds R (e.g. wind raises GPS noise beyond its calibrated value), K is wrong and the estimate degrades — this is precisely the problem KalmanNET solves by learning R implicitly.<br><br>
    <b>The "Linear" constraint</b><br>
    The filter is provably optimal only when two assumptions hold:<br>
    1. <b>Motion model is linear</b> — F is a fixed matrix (constant velocity; no turns or acceleration modelled)<br>
    2. <b>Sensor models are linear</b> — H is a fixed matrix (each sensor selects a fixed subset of state dimensions)<br>
    When either breaks — non-linear dynamics, bearing-angle sensors, atmospheric distortion — the filter is only an approximation. Extensions (EKF, UKF) address this analytically; KalmanNET addresses it by replacing the gain formula with a learned function.`,



  // Sensor control panel ─────────────────────────────────────────────────────

  'panel-sensor-control': `No single sensor is sufficient. GPS drifts without velocity context. IMU accumulates position error without an anchor. Barometer only sees altitude. Magnetometer is too noisy to rely on alone. The Kalman filter 'fuses' these values to form a complete and robust position estimate. <br><br> The kalman filter suffers when the sensors are disabled, but it doesn't fail catastrophically — it falls back to dead-reckoning on the remaining sensors. Watch how the estimate degrades as you disable each sensor in turn.`,

  'btn-gps': `
    <b>GPS — absolute position [x, y, z]</b><br><br>
    <b>Role:</b> The only drift-free absolute 3D fix — the filter's primary position anchor. Without it, horizontal position falls back to the far noisier magnetometer.<br><br>
    <b>Its gap, filled by others:</b> Each fix scatters ±2 m. IMU velocity bridges between fixes so the estimate doesn't jump. Barometer independently validates the altitude component.<br><br>
    <b>Kalman update:</b><br>
    &nbsp;H = [I₃ | 0₃] &nbsp; selects position rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ &nbsp; 6×3 gain matrix<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; all 6 states corrected via cross-covariance in P<br><br>
    <b>Calm-condition assumptions (fixed — never updated):</b><br>
    &nbsp;σ = 2.0 m &nbsp;|&nbsp; R = diag(4, 4, 4)<br>
    Under wind 15 m/s: actual σ ≈ 3.8 m — filter still uses R = 4.<br>
    This mismatch causes the gain K to be too small; sensor is over-trusted.`,

  'btn-imu': `
    <b>IMU — velocity [vx, vy, vz]</b><br><br>
    <b>Role:</b> Fast, continuous updates every step. Bridges the gaps between slow GPS fixes so position doesn't drift in between.<br><br>
    <b>Its gap, filled by others:</b> Measures velocity not position. Velocity errors integrate into unbounded position error over time. GPS corrects the accumulated drift. Without GPS, position grows unconstrained.<br><br>
    <b>Kalman update:</b><br>
    &nbsp;H = [0₃ | I₃] &nbsp; selects velocity rows of the 6-D state<br>
    &nbsp;ε = z − H·x̂ &nbsp; 3×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ &nbsp; 6×3 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; position states also shift via off-diagonal terms in P<br><br>
    <b>Calm-condition assumptions (fixed — never updated):</b><br>
    &nbsp;σ = 0.5 m/s &nbsp;|&nbsp; R = diag(0.25, 0.25, 0.25)<br>
    Wind 15 m/s: actual σ ≈ 1.1 m/s. Temp ±30°C: actual σ ≈ 0.74 m/s.<br>
    KalmanNET would detect the growing innovations and raise its effective gain.`,

  'btn-baro': `
    <b>Barometer — altitude [z only]</b><br><br>
    <b>Role:</b> Cheapest sensor, single axis, highly reliable. Anchors altitude independently of GPS — even when GPS fails, BARO + IMU together keep vertical estimates tight.<br><br>
    <b>Its gap, filled by others:</b> Zero XY awareness. Contributes nothing to horizontal position, which depends entirely on GPS and magnetometer.<br><br>
    <b>Kalman update:</b><br>
    &nbsp;H = [0, 0, 1, 0, 0, 0] &nbsp; altitude (z) row only<br>
    &nbsp;ε = z − H·x̂ &nbsp; scalar innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ &nbsp; 6×1 gain vector<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; scalar correction propagated to all states via K<br><br>
    <b>Calm-condition assumptions (fixed — never updated):</b><br>
    &nbsp;σ = 0.5 m &nbsp;|&nbsp; R = [0.25] &nbsp;|&nbsp; calibrated at 20°C, zero bias<br>
    At 50°C: actual σ ≈ 0.74 m plus a +3.6 m altitude bias (hot air reads drone as lower).<br>
    At −10°C: −3.6 m bias (cold air reads drone as higher). Filter sees neither.`,

  'btn-mag': `
    <b>Magnetometer — horizontal position [x, y]</b><br><br>
    <b>Role:</b> Independent XY fix that doesn't rely on satellite signal. When GPS fails, MAG is the only sensor preventing horizontal position from going unbounded — even if it does so noisily.<br><br>
    <b>Its gap, filled by others:</b> Coarsest sensor at ±3 m. High R means the filter assigns it low gain, so GPS overrides its influence when healthy. Barometer covers the altitude axis this sensor ignores.<br><br>
    <b>Kalman update:</b><br>
    &nbsp;H = [1,0,0,0,0,0; 0,1,0,0,0,0] &nbsp; x and y position rows<br>
    &nbsp;ε = z − H·x̂ &nbsp; 2×1 innovation<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ &nbsp; 6×2 gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp; gentle correction due to large R<br><br>
    <b>Calm-condition assumptions (fixed — never updated):</b><br>
    &nbsp;σ = 3.0 m &nbsp;|&nbsp; R = diag(9, 9)<br>
    Not significantly affected by wind or temperature in this model — its noise is already large enough that environmental variation is secondary.`,

  // ── Environment overlay ──────────────────────────────────────────────────────

  'env-title': `Environmental conditions that stress the sensor suite beyond the Kalman filter's calm-condition assumptions. The filter's R matrices never update — the growing mismatch between assumed and actual noise is exactly what KalmanNET would learn to correct.`,

  'env-reset-btn': `Reset to Kalman calibration defaults: 0 m/s wind, 0° heading, 20°C. These are the exact conditions the filter's R matrices were tuned for — where it performs optimally.`,

  'env-wind-speed-label': `Physically deflects the true trajectory (green line). Also raises GPS noise via multipath and turbulence (actual σ = 2 + 0.12·w m) and IMU noise via vibration (actual σ = 0.5 + 0.04·w m/s). Kalman filter always assumes calm-condition R — it does not know wind has increased.`,

  'env-wind-heading-label': `Direction the wind blows, in degrees clockwise from East. Only has an effect when Wind Speed > 0. The drone's autopilot partially counteracts the push — the true path bows rather than drifts without bound.`,

  'env-temp-label': `Sensors are calibrated at 20°C. Deviations cause IMU thermal drift (actual σ = 0.5 + 0.008·|ΔT| m/s) and a barometer altitude bias (offset = 0.12·ΔT m — hot air has lower pressure, so the baro reads the drone as lower than it is). Kalman filter assumes 20°C at all times.`,




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
