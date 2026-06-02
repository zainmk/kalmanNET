'use strict';

// Coordinate mapping: sim [x,y,z] → three.js [x, z_sim→y, y_sim→z]
function s2t(pt) { return [pt[0], pt[2], pt[1]]; }

// ── Renderer ──────────────────────────────────────────────────────────────────
const canvas   = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x04040f);
scene.fog        = new THREE.FogExp2(0x04040f, 0.0025);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
camera.position.set(60, 38, 60);

const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 8, 0);
controls.update();

// ── Lighting ──────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x1a2060, 1.2));
const sunLight = new THREE.DirectionalLight(0xffffff, 1.1);
sunLight.position.set(60, 120, 40);
scene.add(sunLight);
const blueLight = new THREE.PointLight(0x0044ff, 0.8, 180);
blueLight.position.set(-40, 20, -30);
scene.add(blueLight);

// ── Stars ─────────────────────────────────────────────────────────────────────
(function addStars() {
  const N = 3000;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const th  = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r   = 500 + Math.random() * 400;
    pos[i*3]   = r * Math.sin(phi) * Math.cos(th);
    pos[i*3+1] = r * Math.cos(phi);
    pos[i*3+2] = r * Math.sin(phi) * Math.sin(th);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.9, sizeAttenuation: true })));
})();

// ── Ground grid ───────────────────────────────────────────────────────────────
scene.add(new THREE.GridHelper(140, 28, 0x0e1f66, 0x080f33));

// ── Drone factory ─────────────────────────────────────────────────────────────
function makeDrone(ghost) {
  const g = new THREE.Group();

  function mat(color, opacity) {
    return new THREE.MeshPhongMaterial({ color, transparent: opacity < 1, opacity, wireframe: ghost });
  }

  const bodyMat = mat(ghost ? 0x00ee66 : 0xffffff, ghost ? 0.28 : 1);
  g.add(new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.55, 2.2), bodyMat));
  g.add(new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.22, 0.38), bodyMat.clone()));
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.22, 7.6), bodyMat.clone()));

  const rotMat = new THREE.MeshPhongMaterial({
    color: ghost ? 0x00ee66 : 0x44aaff,
    transparent: true, opacity: ghost ? 0.18 : 0.82,
  });
  const rotors = [];
  [[3.8, 0.32, 0], [-3.8, 0.32, 0], [0, 0.32, 3.8], [0, 0.32, -3.8]].forEach(([x, y, z]) => {
    const r = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 0.09, 20), rotMat);
    r.position.set(x, y, z);
    g.add(r);
    rotors.push(r);
  });

  if (!ghost) {
    [[3.8, 0.42, 0, 0xff2200], [-3.8, 0.42, 0, 0xff2200],
     [0, 0.42, 3.8, 0x00ff44], [0, 0.42, -3.8, 0x00ff44]].forEach(([x, y, z, c]) => {
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({ color: c }));
      led.position.set(x, y, z);
      g.add(led);
    });
    const gimbal = new THREE.Mesh(new THREE.SphereGeometry(0.3, 12, 12), new THREE.MeshPhongMaterial({ color: 0x222222 }));
    gimbal.position.set(0, -0.45, 0);
    g.add(gimbal);
  }

  g.userData.rotors = rotors;
  return g;
}

const drone = makeDrone(false);
const ghost  = makeDrone(true);
ghost.scale.setScalar(0.9);
scene.add(drone);
scene.add(ghost);

// Error line between true and estimated position
const errLineGeo = new THREE.BufferGeometry();
const errLinePos = new Float32Array(6);
errLineGeo.setAttribute('position', new THREE.BufferAttribute(errLinePos, 3));
const errLine = new THREE.Line(errLineGeo, new THREE.LineBasicMaterial({ color: 0xff8800, transparent: true, opacity: 0.55 }));
scene.add(errLine);

// Uncertainty sphere
const uncSphere = new THREE.Mesh(
  new THREE.SphereGeometry(1, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0x0033ff, transparent: true, opacity: 0.055 })
);
scene.add(uncSphere);

// ── Trails ────────────────────────────────────────────────────────────────────
const MAX_TRAIL = 100;

function makeTrailLine(color) {
  const positions = new Float32Array(MAX_TRAIL * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
  scene.add(line);
  return line;
}

const trueTrail = makeTrailLine(0x00ff44);
const estTrail  = makeTrailLine(0x00aaff);
const rawTrail  = makeTrailLine(0xff3300);

function updateTrail(line, trail) {
  const attr = line.geometry.attributes.position;
  const n = Math.min(trail.length, MAX_TRAIL);
  for (let i = 0; i < n; i++) {
    const [tx, ty, tz] = s2t(trail[i]);
    attr.setXYZ(i, tx, ty, tz);
  }
  line.geometry.setDrawRange(0, n);
  attr.needsUpdate = true;
}

// ── Sensor visualisers ────────────────────────────────────────────────────────
function makeDot(color, radius = 0.5) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 10, 10),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 })
  );
  scene.add(m);
  return m;
}

const gpsDot  = makeDot(0xffdd00, 0.45);
const baroDot = makeDot(0xff44aa, 0.4);
const magDot  = makeDot(0xbb44ff, 0.4);

const baroRing = new THREE.Mesh(
  new THREE.TorusGeometry(3, 0.06, 6, 40),
  new THREE.MeshBasicMaterial({ color: 0xff44aa, transparent: true, opacity: 0.4 })
);
baroRing.rotation.x = Math.PI / 2;
scene.add(baroRing);

const imuArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 5, 0xff8800, 1.1, 0.65);
scene.add(imuArrow);

// ── Error chart ───────────────────────────────────────────────────────────────
const errCanvas  = document.getElementById('err-chart');
const errCtx     = errCanvas.getContext('2d');
const errHistory = [];
const MAX_ERR    = 200;

function drawErrChart() {
  const w = errCanvas.clientWidth || 240;
  const h = errCanvas.clientHeight || 64;
  if (errCanvas.width !== w)  errCanvas.width  = w;
  if (errCanvas.height !== h) errCanvas.height = h;
  errCtx.clearRect(0, 0, w, h);
  if (errHistory.length < 2) return;

  const peak = Math.max(10, ...errHistory) * 1.15;

  errCtx.strokeStyle = '#0c1433';
  errCtx.lineWidth = 1;
  [5, 10, 15, 20].forEach(v => {
    const y = h - (v / peak) * h;
    if (y > 0 && y < h) { errCtx.beginPath(); errCtx.moveTo(0, y); errCtx.lineTo(w, y); errCtx.stroke(); }
  });

  errCtx.beginPath();
  errHistory.forEach((val, i) => {
    const x = (i / (MAX_ERR - 1)) * w;
    const y = h - (val / peak) * h;
    i === 0 ? errCtx.moveTo(x, y) : errCtx.lineTo(x, y);
  });
  const maxErr = Math.max(...errHistory);
  errCtx.strokeStyle = maxErr > 10 ? '#ff4422' : maxErr > 4 ? '#ffaa22' : '#00ee66';
  errCtx.lineWidth = 1.5;
  errCtx.stroke();

  errCtx.strokeStyle = '#0e1840';
  errCtx.lineWidth = 1;
  errCtx.beginPath(); errCtx.moveTo(0, h - 0.5); errCtx.lineTo(w, h - 0.5); errCtx.stroke();
}

// ── State application ─────────────────────────────────────────────────────────
function fmt(v, d = 1) { return (v >= 0 ? '+' : '') + v.toFixed(d); }

function setSR(base, rawVal, kfVal, d) {
  const rawEl = document.getElementById(base + '-raw');
  const kfEl  = document.getElementById(base + '-kf');
  if (rawVal === null) {
    rawEl.textContent = 'OFFLINE';
    rawEl.className = 'sr-raw offline';
  } else {
    rawEl.textContent = fmt(rawVal, d);
    rawEl.className = 'sr-raw';
  }
  kfEl.textContent = fmt(kfVal, d);
}

let latestState = null;

function applyState(s) {
  latestState = s;

  const [ex, ey, ez] = s2t(s.est);
  const [tx, ty, tz] = s2t(s.true);

  drone.position.set(ex, ey, ez);
  ghost.position.set(tx, ty, tz);

  errLinePos[0] = ex; errLinePos[1] = ey; errLinePos[2] = ez;
  errLinePos[3] = tx; errLinePos[4] = ty; errLinePos[5] = tz;
  errLineGeo.attributes.position.needsUpdate = true;

  uncSphere.position.set(ex, ey, ez);
  uncSphere.scale.setScalar(Math.max(0.4, s.uncertainty));

  updateTrail(trueTrail, s.true_trail);
  updateTrail(estTrail,  s.est_trail);
  updateTrail(rawTrail,  s.raw_trail);

  const r = s.readings;
  if (r.gps) {
    const [gx, gy, gz] = s2t(r.gps);
    gpsDot.position.set(gx, gy, gz);
    gpsDot.visible = true;
  } else {
    gpsDot.visible = false;
  }

  if (r.baro) {
    baroDot.position.set(ex, r.baro[0], ez);
    baroRing.position.set(ex, r.baro[0], ez);
    baroDot.visible = true;
    baroRing.visible = true;
  } else {
    baroDot.visible = false;
    baroRing.visible = false;
  }

  if (r.mag) {
    magDot.position.set(r.mag[0], ey, r.mag[1]);
    magDot.visible = true;
  } else {
    magDot.visible = false;
  }

  if (r.imu) {
    const vel = new THREE.Vector3(r.imu[0], r.imu[2], r.imu[1]);
    const speed = vel.length();
    if (speed > 0.05) {
      imuArrow.setDirection(vel.normalize());
      imuArrow.setLength(Math.min(speed * 1.6, 10), 1.1, 0.65);
      imuArrow.position.set(ex, ey, ez);
      imuArrow.visible = true;
    }
  } else {
    imuArrow.visible = false;
  }

  ['gps', 'imu', 'baro', 'mag'].forEach(name => {
    document.getElementById(`btn-${name}`).className = 'sensor-btn ' + (s.failed[name] ? 'failed' : 'active');
  });

  const k = s.est;
  setSR('sr-gps-x',  r.gps  ? r.gps[0]  : null, k[0], 1);
  setSR('sr-gps-y',  r.gps  ? r.gps[1]  : null, k[1], 1);
  setSR('sr-gps-z',  r.gps  ? r.gps[2]  : null, k[2], 1);
  setSR('sr-imu-vx', r.imu  ? r.imu[0]  : null, k[3], 2);
  setSR('sr-imu-vy', r.imu  ? r.imu[1]  : null, k[4], 2);
  setSR('sr-imu-vz', r.imu  ? r.imu[2]  : null, k[5], 2);
  setSR('sr-baro-z', r.baro ? r.baro[0] : null, k[2], 1);
  setSR('sr-mag-x',  r.mag  ? r.mag[0]  : null, k[0], 1);
  setSR('sr-mag-y',  r.mag  ? r.mag[1]  : null, k[1], 1);

  const badge = document.getElementById('badge-status');
  if (s.active === 0) {
    badge.textContent = 'DEAD RECKONING ONLY';
    badge.style.cssText = 'border-color:#881100;background:rgba(50,10,10,0.6);color:#ff4422';
  } else if (s.active < 4) {
    badge.textContent = `DEGRADED MODE  (${s.active}/4)`;
    badge.style.cssText = 'border-color:#886600;background:rgba(40,30,5,0.6);color:#ffaa33';
  } else {
    badge.textContent = 'ALL SENSORS NOMINAL';
    badge.style.cssText = 'border-color:#004422;background:rgba(0,30,15,0.5);color:#00ee66';
  }

  errHistory.push(s.error);
  if (errHistory.length > MAX_ERR) errHistory.shift();
  drawErrChart();

  controls.target.y += (ey - 8 - controls.target.y) * 0.008;
}

// ── Environment controls ──────────────────────────────────────────────────────
function resetEnv() {
  document.getElementById('env-wind-speed').value   = 0;
  document.getElementById('env-wind-heading').value = 0;
  document.getElementById('env-temp').value         = 20;
  onEnvChange();
}
window.resetEnv = resetEnv;

function onEnvChange() {
  const speed   = +document.getElementById('env-wind-speed').value;
  const heading = +document.getElementById('env-wind-heading').value;
  const temp    = +document.getElementById('env-temp').value;

  document.getElementById('env-wind-speed-val').textContent  = speed.toFixed(1) + ' m/s';
  document.getElementById('env-wind-heading-val').textContent = heading + '°';
  const dT = temp - 20;
  document.getElementById('env-temp-val').textContent =
    temp + '°C' + (dT !== 0 ? ' (' + (dT > 0 ? '+' : '') + dT + ')' : '');

  fetch('/environment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      wind_speed:   speed,
      wind_heading: heading * Math.PI / 180,
      temperature:  temp,
    }),
  });
}

// ── SSE connection ────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/stream');
  es.onmessage = e => { try { applyState(JSON.parse(e.data)); } catch (_) {} };
  es.onerror   = () => { es.close(); setTimeout(connectSSE, 2000); };
}
connectSSE();

// ── Control endpoints ─────────────────────────────────────────────────────────
async function toggleSensor(name) {
  await fetch('/toggle', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sensor: name }) });
}

async function resetSim() {
  await fetch('/reset', { method: 'POST' });
  errHistory.length = 0;
  trueTrail.geometry.setDrawRange(0, 0);
  estTrail.geometry.setDrawRange(0, 0);
  rawTrail.geometry.setDrawRange(0, 0);
  camera.position.set(60, 38, 60);
  controls.target.set(0, 8, 0);
  controls.update();
}

// ── Tooltips ──────────────────────────────────────────────────────────────────
const tooltipEl = document.getElementById('tooltip');

function positionTooltip(e) {
  const pad = 14;
  const tw  = tooltipEl.offsetWidth;
  const th  = tooltipEl.offsetHeight;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  if (x + tw > window.innerWidth  - pad) x = e.clientX - tw - pad;
  if (y + th > window.innerHeight - pad) y = e.clientY - th - pad;
  tooltipEl.style.left = x + 'px';
  tooltipEl.style.top  = y + 'px';
}

// Rich HTML content for tooltips that need formatting or math notation.
// Keyed by element id; applied below so the HTML stays clean.
const TOOLTIP_RICH = {
  'badge-kf': `
    <b>Linear Kalman Filter — predict → update</b><br><br>
    <b>Predict</b> (every timestep, no sensors needed):<br>
    &nbsp;x̂ ← F·x̂ &nbsp;&nbsp; propagate state via constant-velocity model<br>
    &nbsp;P ← F·P·Fᵀ + Q &nbsp;&nbsp; uncertainty grows with time<br><br>
    <b>Update</b> (once per active sensor per step):<br>
    &nbsp;ε = z − H·x̂ &nbsp;&nbsp; innovation: what sensor saw vs filter predicted<br>
    &nbsp;K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹ &nbsp;&nbsp; Kalman gain<br>
    &nbsp;x̂ ← x̂ + K·ε &nbsp;&nbsp; correct the state<br>
    &nbsp;P ← (I − K·H)·P &nbsp;&nbsp; reduce uncertainty<br><br>
    R is per-sensor noise covariance. Noisier sensor → larger R → smaller K → smaller correction to the state.`,

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
};

// Apply rich tooltips by element id, then wire all [data-tooltip] elements.
Object.entries(TOOLTIP_RICH).forEach(([id, html]) => {
  const el = document.getElementById(id);
  if (el) el.dataset.tooltip = html.trim();
});

document.querySelectorAll('[data-tooltip]').forEach(el => {
  el.addEventListener('mouseenter', e => {
    tooltipEl.innerHTML = el.dataset.tooltip;   // innerHTML: our own controlled content
    tooltipEl.classList.add('visible');
    positionTooltip(e);
  });
  el.addEventListener('mousemove', positionTooltip);
  el.addEventListener('mouseleave', () => tooltipEl.classList.remove('visible'));
});

// ── Resize ────────────────────────────────────────────────────────────────────
function onResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
window.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); resetSim(); } });
onResize();

// ── Animation loop ────────────────────────────────────────────────────────────
let prevNow = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - prevNow) / 1000, 0.1);
  prevNow = now;

  [drone, ghost].forEach(d => {
    d.userData.rotors.forEach((r, i) => { r.rotation.y += (i % 2 === 0 ? 1 : -1) * 18 * dt; });
  });

  controls.update();
  renderer.render(scene, camera);
}

requestAnimationFrame(animate);
