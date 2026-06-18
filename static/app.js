'use strict';

let simPaused  = true;
let isTraining = false;

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
// style: 'kf' (white solid), 'kn' (cyan solid), 'true' (green wireframe),
//        'kf-silhouette' (translucent blue solid)
function makeDrone(style) {
  const g = new THREE.Group();
  const wireframe = style === 'true';
  const translucent = style === 'kf-silhouette';

  const bodyColor = { kf: 0xffffff, kn: 0x00ccbb, true: 0x00ee66, 'kf-silhouette': 0x1155cc }[style];
  const rotorColor = { kf: 0x44aaff, kn: 0x009988, true: 0x00ee66, 'kf-silhouette': 0x1155cc }[style];
  const bodyOpacity = translucent ? 0.30 : (wireframe ? 0.28 : 1);
  const rotorOpacity = translucent ? 0.18 : (wireframe ? 0.18 : 0.82);

  function mat(color, opacity) {
    return new THREE.MeshPhongMaterial({ color, transparent: opacity < 1, opacity, wireframe });
  }

  const bodyMat = mat(bodyColor, bodyOpacity);
  g.add(new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.55, 2.2), bodyMat));
  g.add(new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.22, 0.38), bodyMat.clone()));
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.22, 7.6), bodyMat.clone()));

  const rotMat = new THREE.MeshPhongMaterial({
    color: rotorColor, transparent: true, opacity: rotorOpacity, wireframe,
  });
  const rotors = [];
  [[3.8, 0.32, 0], [-3.8, 0.32, 0], [0, 0.32, 3.8], [0, 0.32, -3.8]].forEach(([x, y, z]) => {
    const r = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 0.09, 20), rotMat);
    r.position.set(x, y, z);
    g.add(r);
    rotors.push(r);
  });

  if (style === 'kf' || style === 'kn') {
    const ledA = style === 'kn' ? 0x00ffee : 0xff2200;
    const ledB = style === 'kn' ? 0x00ffee : 0x00ff44;
    [[3.8, 0.42, 0, ledA], [-3.8, 0.42, 0, ledA],
     [0, 0.42, 3.8, ledB], [0, 0.42, -3.8, ledB]].forEach(([x, y, z, c]) => {
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({ color: c }));
      led.position.set(x, y, z);
      g.add(led);
    });
    const gimbal = new THREE.Mesh(new THREE.SphereGeometry(0.3, 12, 12),
      new THREE.MeshPhongMaterial({ color: style === 'kn' ? 0x003333 : 0x222222 }));
    gimbal.position.set(0, -0.45, 0);
    g.add(gimbal);
  }

  g.userData.rotors = rotors;
  return g;
}

// kfDrone  — white solid, follows KF estimate (active when KN untrained)
// knDrone  — cyan solid, follows KN estimate (active when KN trained)
// kfSilhouette — translucent blue, follows KF estimate (visible alongside knDrone)
// trueDrone — green wireframe, always follows true position
const kfDrone      = makeDrone('kf');
const knDrone      = makeDrone('kn');
const kfSilhouette = makeDrone('kf-silhouette');
const trueDrone    = makeDrone('true');
// legacy alias used in animation loop
const drone = kfDrone;
const ghost = trueDrone;

trueDrone.scale.setScalar(0.9);
kfSilhouette.scale.setScalar(0.92);

scene.add(kfDrone);
scene.add(knDrone);
scene.add(kfSilhouette);
scene.add(trueDrone);

knDrone.visible      = false;
kfSilhouette.visible = false;

// Helix t=0: x=RADIUS·cos(0)=20, y=RADIUS·sin(0)=0, z=0 → s2t = (20, 0, 0)
[kfDrone, knDrone, kfSilhouette, trueDrone].forEach(d => d.position.set(20, 0, 0));

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
const knTrail   = makeTrailLine(0x00ccbb);

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

// ── Badge tooltip — dynamic based on current sensor failures ─────────────────
function buildBadgeTooltip(failed) {
  if (!failed.gps && !failed.imu && !failed.baro && !failed.mag) {
    return 'All 4 sensors active — filter operating at calibration accuracy. GPS anchors 3D position, IMU bridges velocity between fixes, Baro constrains altitude precisely, Mag provides XY backup. Position error typically &lt; 3 m.';
  }
  if (failed.gps && failed.imu && failed.baro && failed.mag) {
    return 'All sensors offline — pure dead-reckoning. The filter propagates using only its constant-velocity motion model. Uncertainty grows unboundedly. The estimate diverges from truth with no path to recovery until a sensor is restored.';
  }

  const parts = [];

  if (failed.gps)
    parts.push('<b>GPS offline</b> — primary position anchor lost. XY now relies solely on magnetometer (σ = 3 m). The filter dead-reckons from IMU velocity with no absolute position correction — error accumulates every step.');
  if (failed.imu)
    parts.push('<b>IMU offline</b> — no velocity corrections between GPS fixes. The filter assumes constant velocity but the drone curves, so predictions degrade between fixes. Error shows a sawtooth pattern: spikes between fixes, partial recovery on each GPS update.');
  if (failed.baro)
    parts.push('<b>Barometer offline</b> — altitude now constrained only by GPS (σ = 2 m vs baro\'s σ = 0.5 m). XY is unaffected. Altitude estimates become noisier. Most significant when temperature was causing a baro bias the filter was following.');
  if (failed.mag)
    parts.push('<b>Magnetometer offline</b> — minimal impact while GPS is active (MAG already has low Kalman gain due to R = diag(9,9)). Critical if GPS also fails: without mag, horizontal position becomes fully unbounded.');

  if (failed.gps && failed.imu)
    parts.push('<b>⚠ GPS + IMU together</b> — the worst combination. No position anchor and no velocity corrections. Error grows rapidly on all horizontal axes.');

  return parts.join('<br><br>');
}

// ── State application ─────────────────────────────────────────────────────────
function fmt(v, d = 1) { return (v >= 0 ? '+' : '') + v.toFixed(d); }

function setSR(base, rawVal, kfVal, knVal, d) {
  const rawEl = document.getElementById(base + '-raw');
  const kfEl  = document.getElementById(base + '-kf');
  const knEl  = document.getElementById(base + '-kn');
  if (rawVal === null) {
    rawEl.textContent = 'OFFLINE';
    rawEl.className = 'sr-raw offline';
  } else {
    rawEl.textContent = fmt(rawVal, d);
    rawEl.className = 'sr-raw';
  }
  kfEl.textContent = fmt(kfVal, d);
  if (knEl) {
    if (knVal !== null) {
      knEl.textContent = fmt(knVal, d);
      knEl.className = 'sr-kn trained';
    } else {
      knEl.textContent = '—';
      knEl.className = 'sr-kn';
    }
  }
}

let latestState  = null;
let prevKNTrained = false;

// ── Drone mode flash banner ────────────────────────────────────────────────────
function flashDroneMode(isKN) {
  const el = document.getElementById('drone-mode-flash');
  el.className = '';
  void el.offsetWidth; // force reflow to restart animation
  el.textContent = isKN ? 'DRONE  ●  KALMANNET' : 'DRONE  ●  KALMAN FILTER';
  el.className   = 'drone-mode-flash ' + (isKN ? 'kn' : 'kf') + ' visible';
}

// Update the bottom legend to reflect which drone is the "estimate" drone
function updateLegendForMode(isKN) {
  const estRow = document.getElementById('legend-est-drone');
  estRow.querySelector('.leg-dot').style.background = isKN ? '#00ccbb' : '#ffffff';
  // lastChild is the plain text node "Est. drone" / "KN drone"
  const txt = estRow.lastChild;
  if (txt && txt.nodeType === Node.TEXT_NODE) txt.nodeValue = isKN ? 'KN drone' : 'Est. drone';

  const silRow = document.getElementById('legend-kf-sil');
  if (silRow) silRow.style.display = isKN ? '' : 'none';
}

function applyState(s) {
  latestState = s;

  const [ex, ey, ez] = s2t(s.est);
  const [tx, ty, tz] = s2t(s.true);
  const knActive     = !!(s.kn_trained && s.kn_est);

  // ── Drone visibility & position ──────────────────────────────────────────────
  if (knActive) {
    const [kx, ky, kz] = s2t(s.kn_est);
    knDrone.position.set(kx, ky, kz);
    kfSilhouette.position.set(ex, ey, ez);
    kfDrone.visible      = false;
    knDrone.visible      = true;
    kfSilhouette.visible = true;

    errLinePos[0] = kx; errLinePos[1] = ky; errLinePos[2] = kz;
    errLinePos[3] = tx; errLinePos[4] = ty; errLinePos[5] = tz;
    uncSphere.position.set(kx, ky, kz);
    controls.target.y += (ky - 8 - controls.target.y) * 0.008;
  } else {
    kfDrone.position.set(ex, ey, ez);
    kfDrone.visible      = true;
    knDrone.visible      = false;
    kfSilhouette.visible = false;

    errLinePos[0] = ex; errLinePos[1] = ey; errLinePos[2] = ez;
    errLinePos[3] = tx; errLinePos[4] = ty; errLinePos[5] = tz;
    uncSphere.position.set(ex, ey, ez);
    controls.target.y += (ey - 8 - controls.target.y) * 0.008;
  }

  trueDrone.position.set(tx, ty, tz);
  errLineGeo.attributes.position.needsUpdate = true;
  uncSphere.scale.setScalar(Math.max(0.4, s.uncertainty));

  // Flash + legend update on mode change
  if (knActive !== prevKNTrained) {
    prevKNTrained = knActive;
    flashDroneMode(knActive);
    updateLegendForMode(knActive);
  }

  updateTrail(trueTrail, s.true_trail);
  updateTrail(estTrail,  s.est_trail);
  updateTrail(rawTrail,  s.raw_trail);
  updateTrail(knTrail,   s.kn_trail || []);

  const r = s.readings;
  if (r.gps) {
    const [gx, gy, gz] = s2t(r.gps);
    gpsDot.position.set(gx, gy, gz);
    gpsDot.visible = true;
  } else {
    gpsDot.visible = false;
  }

  // Sensor visualisers anchor to whichever drone is currently active
  const [ax, ay, az] = knActive ? s2t(s.kn_est) : [ex, ey, ez];

  if (r.baro) {
    baroDot.position.set(ax, r.baro[0], az);
    baroRing.position.set(ax, r.baro[0], az);
    baroDot.visible = true;
    baroRing.visible = true;
  } else {
    baroDot.visible = false;
    baroRing.visible = false;
  }

  if (r.mag) {
    magDot.position.set(r.mag[0], ay, r.mag[1]);
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
      imuArrow.position.set(ax, ay, az);
      imuArrow.visible = true;
    }
  } else {
    imuArrow.visible = false;
  }

  ['gps', 'imu', 'baro', 'mag'].forEach(name => {
    document.getElementById(`btn-${name}`).className = 'sensor-btn ' + (s.failed[name] ? 'failed' : 'active');
  });

  const k  = s.est;
  const kn = s.kn_trained ? (s.kn_est || []) : null;
  setSR('sr-gps-x',  r.gps  ? r.gps[0]  : null, k[0], kn ? kn[0] : null, 1);
  setSR('sr-gps-y',  r.gps  ? r.gps[1]  : null, k[1], kn ? kn[1] : null, 1);
  setSR('sr-gps-z',  r.gps  ? r.gps[2]  : null, k[2], kn ? kn[2] : null, 1);
  setSR('sr-imu-vx', r.imu  ? r.imu[0]  : null, k[3], kn ? kn[3] : null, 2);
  setSR('sr-imu-vy', r.imu  ? r.imu[1]  : null, k[4], kn ? kn[4] : null, 2);
  setSR('sr-imu-vz', r.imu  ? r.imu[2]  : null, k[5], kn ? kn[5] : null, 2);
  setSR('sr-baro-z', r.baro ? r.baro[0] : null, k[2], kn ? kn[2] : null, 1);
  setSR('sr-mag-x',  r.mag  ? r.mag[0]  : null, k[0], kn ? kn[0] : null, 1);
  setSR('sr-mag-y',  r.mag  ? r.mag[1]  : null, k[1], kn ? kn[1] : null, 1);

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
  badge.dataset.tooltip = buildBadgeTooltip(s.failed);

  // KN K-ratio display + training lock
  updateKNPanel(s);

  // Sync play button if backend paused state differs (e.g. after training auto-reset)
  if ('paused' in s && s.paused !== simPaused) {
    simPaused = s.paused;
    const btn = document.getElementById('btn-play-pause');
    btn.innerHTML = simPaused ? '&#9654; PLAY' : '&#9646;&#9646; PAUSE';
    btn.classList.toggle('playing', !simPaused);
  }

}

// ── KalmanNET panel ───────────────────────────────────────────────────────────

const KN_PHASES = [
  'CALM BASELINE',
  'WIND  10 m/s',
  'WIND  20 m/s',
  'WIND   5 m/s',
  'CALM',
  'TEMP  40°C',
  'TEMP  50°C',
  'TEMP -10°C',
  'COMBINED STRESS',
  'RETURN TO CALM',
];

function renderPhaseList(phaseIdx, step) {
  const el = document.getElementById('kn-phase-list');
  const optimising = step === 'training';
  el.innerHTML = KN_PHASES.map((name, i) => {
    let cls, icon;
    if (optimising || i < phaseIdx) {
      cls = 'kn-pl-done'; icon = '✓';
    } else if (i === phaseIdx) {
      cls = 'kn-pl-active'; icon = '▶';
    } else {
      cls = 'kn-pl-pending'; icon = '·';
    }
    return `<div class="kn-pl-row ${cls}"><span class="kn-pl-icon">${icon}</span><span>${name.trim()}</span></div>`;
  }).join('');
  if (optimising) {
    el.innerHTML += `<div class="kn-pl-row kn-pl-active"><span class="kn-pl-icon">▶</span><span>OPTIMISING NETWORK</span></div>`;
  }
}

const _LOCK_IDS = [
  'btn-play-pause', 'btn-reset-header',
  'env-wind-speed', 'env-wind-heading', 'env-temp', 'env-reset-btn',
  'btn-gps', 'btn-imu', 'btn-baro', 'btn-mag',
  'btn-clear',
];

function setTrainingLock(locked) {
  isTraining = locked;
  _LOCK_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = locked;
  });
}

function updateKNPanel(s) {
  const dot      = document.getElementById('kn-status-dot');
  const label    = document.getElementById('kn-status-label');
  const info     = document.getElementById('kn-training-info');
  const btn      = document.getElementById('btn-train');
  const clearBtn = document.getElementById('btn-clear');
  const rhat     = document.getElementById('kn-rhat');
  const t        = s.training || {};

  if (t.active) {
    setTrainingLock(true);
    dot.className   = 'kn-status-dot training';
    label.textContent = t.step === 'training' ? 'OPTIMISING' : 'COLLECTING DATA';
    info.style.display = '';
    document.getElementById('kn-bar-fill').style.width = ((t.progress || 0) * 100).toFixed(1) + '%';
    renderPhaseList(t.phase_idx ?? 0, t.step);
    btn.style.display      = 'none';
    clearBtn.style.display = 'none';
    rhat.style.display = 'none';
    if (t.wind !== undefined) {
      document.getElementById('env-wind-speed').value = t.wind;
      document.getElementById('env-wind-speed-val').textContent = t.wind.toFixed(1) + ' m/s';
    }
    if (t.temp !== undefined) {
      document.getElementById('env-temp').value = t.temp;
      document.getElementById('env-temp-val').textContent = t.temp.toFixed(0) + '°C';
    }
  } else if (s.kn_trained) {
    setTrainingLock(false);
    dot.className   = 'kn-status-dot active';
    label.textContent = 'ACTIVE';
    info.style.display = 'none';
    clearBtn.style.display = '';
    clearBtn.disabled      = false;
    btn.style.display      = 'none';
    rhat.style.display = '';
    renderKratio(s.kn_k_ratio || {});
  } else {
    setTrainingLock(false);
    dot.className   = 'kn-status-dot';
    label.textContent = 'UNTRAINED';
    info.style.display = 'none';
    clearBtn.style.display = 'none';
    btn.style.display      = '';
    btn.disabled           = false;
    btn.innerHTML = '&#9654; &nbsp;TRAIN KALMANNET';
    rhat.style.display = 'none';
  }
}

function renderKratio(kratio) {
  const grid = document.getElementById('kn-rhat-grid');
  const rows = [
    ['GPS',  kratio.gps],
    ['IMU',  kratio.imu],
    ['BARO', kratio.baro],
    ['MAG',  kratio.mag],
  ];
  grid.innerHTML = rows.map(([name, vals]) => {
    if (!vals || !vals.length) return '';
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const dev  = Math.abs(mean - 1.0);
    const cls  = dev > 2.0 ? 'rhat-high' : dev > 0.8 ? 'rhat-mid' : 'rhat-ok';
    const str  = vals.map(v => v.toFixed(2)).join(', ');
    const dir  = mean > 1.15 ? '↑' : mean < 0.85 ? '↓' : '~';
    return `<span class="rhat-name">${name}</span>` +
           `<span class="rhat-val ${cls}">${str}</span>` +
           `<span class="rhat-ratio ${cls}">${dir}×${mean.toFixed(1)}</span>`;
  }).join('');
}

async function clearModel() {
  await fetch('/clear-model', { method: 'POST' });
}

async function startTraining() {
  await fetch('/train', { method: 'POST' });
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

async function togglePause() {
  const res  = await fetch('/pause', { method: 'POST' });
  const data = await res.json();
  simPaused  = data.paused;
  const btn  = document.getElementById('btn-play-pause');
  btn.innerHTML  = simPaused ? '&#9654; PLAY' : '&#9646;&#9646; PAUSE';
  btn.classList.toggle('playing', !simPaused);
}

async function resetSim() {
  await fetch('/reset', { method: 'POST' });
  trueTrail.geometry.setDrawRange(0, 0);
  estTrail.geometry.setDrawRange(0, 0);
  rawTrail.geometry.setDrawRange(0, 0);
  knTrail.geometry.setDrawRange(0, 0);
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

// Tooltip content lives in tooltips.js — applied before this script runs.
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
window.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); if (!isTraining) togglePause(); }
  if (e.code === 'KeyR')  { e.preventDefault(); if (!isTraining) resetSim(); }
});
onResize();

// ── Animation loop ────────────────────────────────────────────────────────────
let prevNow = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - prevNow) / 1000, 0.1);
  prevNow = now;

  if (!simPaused) {
    [kfDrone, knDrone, trueDrone, kfSilhouette].forEach(d => {
      d.userData.rotors.forEach((r, i) => { r.rotation.y += (i % 2 === 0 ? 1 : -1) * 18 * dt; });
    });
  }

  controls.update();
  renderer.render(scene, camera);
}

requestAnimationFrame(animate);
