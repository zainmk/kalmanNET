'use strict';

// Direct port of kalmannet.py (inference only — training stays in PyTorch).
// The GRU forward pass replicates torch.nn.GRU exactly: gate rows ordered
// r|z|n, h₀ = 0, h' = (1−z)∘n + z∘h. Weights come from web/weights.json
// (exported by export_web_weights.py); web/test/parity.mjs holds this port
// to PyTorch reference outputs.

const SENSOR_ORDER   = ['gps', 'imu', 'baro', 'mag'];
const SENSOR_DIMS    = { gps: 3, imu: 3, baro: 1, mag: 2 };
const SENSOR_OFFSETS = { gps: 0, imu: 3, baro: 6, mag: 7 };
const TOTAL_MEAS_DIM = 9;
const STATE_DIM      = 6;

function _sigmoid(v) { return 1 / (1 + Math.exp(-v)); }

class KalmanNET {
  constructor(dt = 0.05) {
    this.dt      = dt;
    this.trained = false;
    this.weights = null;
    this._h      = null;
    this._kf     = new KalmanFilter(dt);
    this._kRatio = {};
    for (const s of SENSOR_ORDER) this._kRatio[s] = new Array(SENSOR_DIMS[s]).fill(1.0);
  }

  // weights: parsed weights.json — {format, input, hidden, gru:{w_ih,w_hh,b_ih,b_hh}, heads:{s:{w,b}}}
  loadWeights(weights) {
    if (!weights || weights.format !== 'gru-kalmannet-v1')
      throw new Error('KalmanNET.loadWeights: unexpected weights format');
    this.weights = weights;
    this._h      = null;
    this.trained = true;
  }

  reset(h0) {
    this._kf   = new KalmanFilter(this.dt);
    this._kf.x = h0.slice();
    this._h    = null;
    for (const s of SENSOR_ORDER) this._kRatio[s] = new Array(SENSOR_DIMS[s]).fill(1.0);
  }

  // ── Main interface: one call per sim step ──────────────────────────────────
  step(zValues) {
    this._kf.predict();

    const names = Object.keys(zValues);
    if (!names.length) return;

    if (!this.trained) {
      this._kfFallback(zValues);
      return;
    }

    // Safety: reset if state has diverged — before innovations are built, so
    // they are computed from the post-reset state (mirrors kalmannet.py).
    if (this._kf.x.some(v => !Number.isFinite(v) || Math.abs(v) > 1e5)) {
      this._kf.x = new Array(STATE_DIM).fill(0);
      this._kf.P = Mat.eye(STATE_DIM, 10.0);
      this._h    = null;
    }

    // Build 9D innovation vector (all sensors combined). Math.fround mirrors
    // the float32 cast the Python side applies before the GRU.
    const iv9 = new Array(TOTAL_MEAS_DIM).fill(0);
    const innovDict = {};
    for (const name of names) {
      const H     = this._kf.sensors[name].H;
      const innov = Mat.vecSub(zValues[name], Mat.mulVec(H, this._kf.x));
      innovDict[name] = innov;
      const o = SENSOR_OFFSETS[name];
      for (let k = 0; k < innov.length; k++)
        iv9[o + k] = Math.fround(Math.min(1e4, Math.max(-1e4, innov[k])));
    }

    const K_out = this._gruForward(iv9);

    // Apply K for each active sensor in fixed order
    for (const name of SENSOR_ORDER) {
      if (!(name in zValues)) continue;
      const H    = this._kf.sensors[name].H;
      const Rcal = this._kf.sensors[name].R;
      const Ht   = Mat.transpose(H);
      const m    = SENSOR_DIMS[name];

      // Riccati K — display reference and stability bound for the clip.
      const S_ref = Mat.add(Mat.mul(Mat.mul(H, this._kf.P), Ht), Rcal);
      const K_ref = Mat.mul(Mat.mul(this._kf.P, Ht), Mat.inv(S_ref));

      // Clip element-wise to ≤ 2 × |K_ref| — the stability envelope. K_ref has
      // near-zero off-diagonals, so this suppresses the dense off-diagonal K
      // that makes (I − K·H) eigenvalues leave ±1 and blow up P. Never remove.
      const K = K_out[name];
      for (let i = 0; i < STATE_DIM; i++)
        for (let j = 0; j < m; j++)
          K[i][j] = Math.min(Math.max(K[i][j], 0), Math.abs(K_ref[i][j]) * 2 + 1e-6);

      const ratios = new Array(m);
      for (let j = 0; j < m; j++) {
        let kn = 0, rn = 0;
        for (let i = 0; i < STATE_DIM; i++) {
          kn += K[i][j] * K[i][j];
          rn += K_ref[i][j] * K_ref[i][j];
        }
        ratios[j] = (Math.sqrt(kn) + 1e-9) / (Math.sqrt(rn) + 1e-9);
      }
      this._kRatio[name] = ratios;

      this._kf.x = Mat.vecAdd(this._kf.x, Mat.mulVec(K, innovDict[name]));
      const IKH = Mat.sub(Mat.eye(STATE_DIM), Mat.mul(K, H));
      const P   = Mat.add(
        Mat.mul(Mat.mul(IKH, this._kf.P), Mat.transpose(IKH)),
        Mat.mul(Mat.mul(K, Rcal), Mat.transpose(K))
      );
      // Symmetrize: P = 0.5·(P + Pᵀ)
      for (let i = 0; i < STATE_DIM; i++)
        for (let j = i + 1; j < STATE_DIM; j++) {
          const v = 0.5 * (P[i][j] + P[j][i]);
          P[i][j] = v;
          P[j][i] = v;
        }
      this._kf.P = P;
    }
  }

  // One GRU step + all four heads. Updates this._h, returns {sensor: K 6×m}
  // with elements in (0, 2) — pre-clip, exactly what the torch net outputs.
  _gruForward(iv9) {
    const w = this.weights;
    const H = w.hidden;
    const h = this._h || new Array(H).fill(0);
    const gi = new Array(3 * H);
    const gh = new Array(3 * H);
    for (let r = 0; r < 3 * H; r++) {
      let si = w.gru.b_ih[r];
      const wi = w.gru.w_ih[r];
      for (let j = 0; j < TOTAL_MEAS_DIM; j++) si += wi[j] * iv9[j];
      gi[r] = si;
      let sh = w.gru.b_hh[r];
      const wh = w.gru.w_hh[r];
      for (let j = 0; j < H; j++) sh += wh[j] * h[j];
      gh[r] = sh;
    }
    const hNew = new Array(H);
    for (let i = 0; i < H; i++) {
      const r = _sigmoid(gi[i] + gh[i]);
      const z = _sigmoid(gi[H + i] + gh[H + i]);
      const n = Math.tanh(gi[2 * H + i] + r * gh[2 * H + i]);
      hNew[i] = (1 - z) * n + z * h[i];
    }
    this._h = hNew;

    const K = {};
    for (const s of SENSOR_ORDER) {
      const m    = SENSOR_DIMS[s];
      const head = w.heads[s];
      const Ks   = Mat.zeros(STATE_DIM, m);
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = 0; j < m; j++) {
          const row = i * m + j;   // torch .view(-1, STATE_DIM, m) is row-major
          let acc = head.b[row];
          const wr = head.w[row];
          for (let k = 0; k < H; k++) acc += wr[k] * hNew[k];
          Ks[i][j] = 2 * _sigmoid(acc);
        }
      }
      K[s] = Ks;
    }
    return K;
  }

  // Exact classical update — must match kalman_filter.py (no Joseph form here,
  // mirroring kalmannet.py's _kf_fallback).
  _kfFallback(zValues) {
    for (const name of SENSOR_ORDER) {
      if (!(name in zValues)) continue;
      const H  = this._kf.sensors[name].H;
      const R  = this._kf.sensors[name].R;
      const Ht = Mat.transpose(H);
      const innov = Mat.vecSub(zValues[name], Mat.mulVec(H, this._kf.x));
      const S = Mat.add(Mat.mul(Mat.mul(H, this._kf.P), Ht), R);
      const K = Mat.mul(Mat.mul(this._kf.P, Ht), Mat.inv(S));
      this._kf.x = Mat.vecAdd(this._kf.x, Mat.mulVec(K, innov));
      this._kf.P = Mat.mul(Mat.sub(Mat.eye(STATE_DIM), Mat.mul(K, H)), this._kf.P);
    }
  }

  positionUncertainty() {
    return Math.sqrt(Math.max(0, this._kf.P[0][0] + this._kf.P[1][1] + this._kf.P[2][2]));
  }
}

if (typeof module !== 'undefined' && module.exports)
  module.exports = { KalmanNET, SENSOR_ORDER, SENSOR_DIMS, SENSOR_OFFSETS };
