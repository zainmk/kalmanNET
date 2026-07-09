'use strict';

// Direct port of drone_sim.py. The returned state dict uses the exact key
// names applyState() in static/app.js consumes (the SSE contract) — renaming
// anything here breaks the shared frontend.
//
// Sensor-noise numbers are duplicated from drone_sim.py and quoted in prose in
// static/tooltips.js and README.md — keep all in sync if any value changes.

class DroneSim {
  constructor(dt = 0.05, opts = {}) {
    this.dt = dt;
    this.t  = 0.0;
    this._random = opts.random || Math.random;   // injectable for seeded tests
    this._spare  = null;                          // Box-Muller cache

    this.sensorFailed = { gps: false, imu: false, baro: false, mag: false };
    this.trueTrail = [];
    this.estTrail  = [];
    this.rawTrail  = [];
    this.knTrail   = [];
    const h0 = this._trueHelix(0.0);
    this.rawPos = [h0[0], h0[1], h0[2]];

    // Environmental state (user-controlled)
    this.windSpeed   = 0.0;    // m/s, 0–20
    this.windHeading = 0.0;    // radians
    this.temperature = 20.0;   // °C absolute; calibration point is 20 °C
    this.windDisp    = [0.0, 0.0];

    // Training data collection (buffer is real; the browser build ships
    // inference only — download it for offline training)
    this.collectingData = false;
    this.trainingBuffer = [];

    this._initFilters();
  }

  static get RADIUS() { return 20.0; }
  static get OMEGA()  { return 0.3; }
  static get VZ()     { return 1.0; }

  _randn(std) {
    // Box-Muller with spare caching
    if (this._spare !== null) {
      const v = this._spare;
      this._spare = null;
      return v * std;
    }
    let u = 0, v = 0;
    while (u === 0) u = this._random();
    v = this._random();
    const r  = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * v;
    this._spare = r * Math.sin(th);
    return r * Math.cos(th) * std;
  }

  _initFilters() {
    const h0 = this._trueHelix(0.0);
    this.kf = new KalmanFilter(this.dt);
    this.kf.x = h0.slice();
    // Preserve the trained network across resets — only reset filter state
    if (this.kn && this.kn.trained) {
      this.kn.reset(h0);
    } else {
      this.kn = new KalmanNET(this.dt);
      this.kn.reset(h0);
    }
  }

  _trueHelix(t) {
    const R = DroneSim.RADIUS, w = DroneSim.OMEGA, vz = DroneSim.VZ;
    return [
      R * Math.cos(w * t),      R * Math.sin(w * t),      vz * t,
      -R * w * Math.sin(w * t), R * w * Math.cos(w * t),  vz,
    ];
  }

  toggleSensor(name) {
    if (name in this.sensorFailed) this.sensorFailed[name] = !this.sensorFailed[name];
  }

  setEnvironment(windSpeed, windHeading, temperature) {
    this.windSpeed   = Math.max(0.0, Math.min(20.0, windSpeed));
    this.windHeading = windHeading;   // radians
    this.temperature = Math.max(-10.0, Math.min(50.0, temperature));
  }

  step() {
    this.t += this.dt;

    // ── Wind drift (EMA, 3 % decay per step) ──────────────────────────────────
    const prevDisp = this.windDisp.slice();
    let gustX = 0.0, gustY = 0.0;
    if (this.windSpeed > 0) {
      gustX = this.windSpeed * Math.cos(this.windHeading) + this._randn(this.windSpeed * 0.1);
      gustY = this.windSpeed * Math.sin(this.windHeading) + this._randn(this.windSpeed * 0.1);
    }
    this.windDisp[0] = 0.97 * this.windDisp[0] + gustX * this.dt * 0.08;
    this.windDisp[1] = 0.97 * this.windDisp[1] + gustY * this.dt * 0.08;
    const windVelX = (this.windDisp[0] - prevDisp[0]) / this.dt;
    const windVelY = (this.windDisp[1] - prevDisp[1]) / this.dt;

    // ── True state (helix + wind displacement) ────────────────────────────────
    const helix = this._trueHelix(this.t);
    const truth = [
      helix[0] + this.windDisp[0],
      helix[1] + this.windDisp[1],
      helix[2],
      helix[3] + windVelX,
      helix[4] + windVelY,
      helix[5],
    ];

    // ── Effective sensor noise (environment-scaled) ───────────────────────────
    const dT = this.temperature - 20.0;
    const gpsStd  = 2.0 + 0.12 * this.windSpeed;
    const imuStd  = 0.5 + 0.04 * this.windSpeed + 0.008 * Math.abs(dT);
    const effStd = {
      gps:  [gpsStd, gpsStd, gpsStd],
      imu:  [imuStd, imuStd, imuStd],
      baro: [0.5 + 0.008 * Math.abs(dT)],
      mag:  [3.0, 3.0],
    };
    // Warm air: pressure at altitude is higher than the standard-atmosphere
    // model assumes → altimeter reads low. Hot reads low, cold reads high.
    const baroBias = -dT * 0.25;

    // ── Generate sensor readings ──────────────────────────────────────────────
    const readings = {};
    const zValues  = {};
    for (const name of Object.keys(this.sensorFailed)) {
      if (!this.sensorFailed[name]) {
        const H = this.kf.sensors[name].H;
        const z = Mat.mulVec(H, truth).map((v, i) => v + this._randn(effStd[name][i]));
        if (name === 'baro') z[0] += baroBias;
        zValues[name]  = z;
        readings[name] = z.slice();
      } else {
        readings[name] = null;
      }
    }

    // ── Update filters ────────────────────────────────────────────────────────
    this.kf.predict();
    for (const name of Object.keys(zValues)) this.kf.update(name, zValues[name]);

    this.kn.step(zValues);   // predict + all-sensor update in one GRU call

    const est   = this.kf.x.slice();
    const knEst = this.kn._kf.x.slice();

    // ── Collect training data (all sensors must be active) ───────────────────
    const anyFailed = Object.values(this.sensorFailed).some(v => v);
    if (this.collectingData && !anyFailed) {
      const meas = {};
      for (const name of Object.keys(zValues)) meas[name] = zValues[name].slice();
      this.trainingBuffer.push({ measurements: meas, x_true: truth.slice() });
    }

    // ── Raw position estimate ─────────────────────────────────────────────────
    let raw = this.rawPos.slice();
    if (readings.gps) {
      raw = readings.gps.slice(0, 3);
    } else {
      if (readings.mag) {
        raw[0] = readings.mag[0];
        raw[1] = readings.mag[1];
      }
      if (readings.baro) raw[2] = readings.baro[0];
    }
    this.rawPos = raw;

    // ── Rolling trail buffers (last 150 steps) ────────────────────────────────
    this.trueTrail.push(truth.slice(0, 3));
    this.estTrail.push(est.slice(0, 3));
    this.rawTrail.push(raw.slice());
    this.knTrail.push(knEst.slice(0, 3));
    if (this.trueTrail.length > 150) {
      this.trueTrail.shift();
      this.estTrail.shift();
      this.rawTrail.shift();
      this.knTrail.shift();
    }

    const active = Object.values(this.sensorFailed).filter(v => !v).length;
    const err   = Math.hypot(truth[0] - est[0],   truth[1] - est[1],   truth[2] - est[2]);
    const knErr = Math.hypot(truth[0] - knEst[0], truth[1] - knEst[1], truth[2] - knEst[2]);
    const kRatio = {};
    for (const s of Object.keys(this.kn._kRatio)) kRatio[s] = this.kn._kRatio[s].slice();

    return {
      t:           Math.round(this.t * 100) / 100,
      true:        truth.slice(),
      est:         est,
      readings:    readings,
      failed:      Object.assign({}, this.sensorFailed),
      raw:         this.rawPos.slice(),
      true_trail:  this.trueTrail.slice(-80),
      est_trail:   this.estTrail.slice(-80),
      raw_trail:   this.rawTrail.slice(-80),
      kn_est:      knEst,
      kn_trail:    this.knTrail.slice(-80),
      kn_error:    knErr,
      kn_k_ratio:  kRatio,
      kn_trained:  this.kn.trained,
      error:       err,
      uncertainty: this.kf.positionUncertainty(),
      kn_uncertainty: this.kn.positionUncertainty(),
      active:      active,
      env: {
        wind_speed:   Math.round(this.windSpeed * 10) / 10,
        wind_heading: Math.round(this.windHeading * 180 / Math.PI * 10) / 10,
        temperature:  Math.round(this.temperature * 10) / 10,
      },
    };
  }

  reset() {
    this.t = 0.0;
    this.trueTrail.length = 0;
    this.estTrail.length  = 0;
    this.rawTrail.length  = 0;
    this.knTrail.length   = 0;
    for (const k of Object.keys(this.sensorFailed)) this.sensorFailed[k] = false;
    this.windDisp = [0.0, 0.0];
    const h0 = this._trueHelix(0.0);
    this.rawPos = [h0[0], h0[1], h0[2]];
    this._initFilters();
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { DroneSim };
