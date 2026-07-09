'use strict';

// Direct port of kalman_filter.py. State: [x, y, z, vx, vy, vz].
// Keep the math and the numeric values in lockstep with the Python file —
// web/test/parity.mjs replays PyTorch/numpy reference vectors against this.

function _makeSensors() {
  return {
    gps: {
      H: [[1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0]],
      R: [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
      noise_std: [2.0, 2.0, 2.0],
    },
    imu: {
      H: [[0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1]],
      R: [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]],
      noise_std: [0.5, 0.5, 0.5],
    },
    baro: {
      H: [[0, 0, 1, 0, 0, 0]],
      R: [[0.25]],
      noise_std: [0.5],
    },
    mag: {
      H: [[1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0]],
      R: [[9, 0], [0, 9]],
      noise_std: [3.0, 3.0],
    },
  };
}

class KalmanFilter {
  constructor(dt = 0.05, processNoise = 1.5) {
    this.dt = dt;
    const n = 6;

    this.x = new Array(n).fill(0);
    this.P = Mat.eye(n, 100.0);

    // Constant-velocity state transition
    this.F = Mat.eye(n);
    this.F[0][3] = dt;
    this.F[1][4] = dt;
    this.F[2][5] = dt;

    // Discretised white-noise-acceleration Q (same block structure as Python)
    const q = processNoise, t2 = dt * dt, t3 = t2 * dt, t4 = t3 * dt;
    this.Q = Mat.zeros(n, n);
    for (let i = 0; i < 3; i++) {
      this.Q[i][i]         = q * t4 / 4;
      this.Q[i][i + 3]     = q * t3 / 2;
      this.Q[i + 3][i]     = q * t3 / 2;
      this.Q[i + 3][i + 3] = q * t2;
    }

    this.sensors = _makeSensors();
  }

  predict() {
    this.x = Mat.mulVec(this.F, this.x);
    this.P = Mat.add(Mat.mul(Mat.mul(this.F, this.P), Mat.transpose(this.F)), this.Q);
  }

  update(sensorName, z) {
    const H  = this.sensors[sensorName].H;
    const R  = this.sensors[sensorName].R;
    const Ht = Mat.transpose(H);
    const innov = Mat.vecSub(z, Mat.mulVec(H, this.x));
    const S = Mat.add(Mat.mul(Mat.mul(H, this.P), Ht), R);
    const K = Mat.mul(Mat.mul(this.P, Ht), Mat.inv(S));
    this.x = Mat.vecAdd(this.x, Mat.mulVec(K, innov));
    this.P = Mat.mul(Mat.sub(Mat.eye(6), Mat.mul(K, H)), this.P);
  }

  positionUncertainty() {
    return Math.sqrt(this.P[0][0] + this.P[1][1] + this.P[2][2]);
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { KalmanFilter };
