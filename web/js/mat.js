'use strict';

// Small dense-matrix helpers for the browser port. Matrices are arrays of row
// arrays (float64); vectors are plain arrays. Sizes here are tiny (≤ 6×6), so
// clarity beats cleverness — no library, no typed-array micro-optimisation.

const Mat = {

  zeros(r, c) {
    return Array.from({ length: r }, () => new Array(c).fill(0));
  },

  eye(n, s = 1) {
    const A = Mat.zeros(n, n);
    for (let i = 0; i < n; i++) A[i][i] = s;
    return A;
  },

  clone(A) {
    return A.map(row => row.slice());
  },

  transpose(A) {
    const r = A.length, c = A[0].length;
    const T = Mat.zeros(c, r);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  },

  add(A, B) {
    return A.map((row, i) => row.map((v, j) => v + B[i][j]));
  },

  sub(A, B) {
    return A.map((row, i) => row.map((v, j) => v - B[i][j]));
  },

  mul(A, B) {
    const r = A.length, n = B.length, c = B[0].length;
    const C = Mat.zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let k = 0; k < n; k++) {
        const a = A[i][k];
        if (a === 0) continue;
        for (let j = 0; j < c; j++) C[i][j] += a * B[k][j];
      }
    }
    return C;
  },

  mulVec(A, x) {
    return A.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
  },

  // Gauss-Jordan inverse with partial pivoting. Only used on S = H·P·Hᵀ + R,
  // which is ≤ 3×3, symmetric positive definite.
  inv(A) {
    const n = A.length;
    const M = A.map((row, i) => row.concat(Mat.eye(n)[i]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++)
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (M[piv][col] === 0) throw new Error('Mat.inv: singular matrix');
      if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
      const d = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map(row => row.slice(n));
  },

  vecAdd(a, b) { return a.map((v, i) => v + b[i]); },
  vecSub(a, b) { return a.map((v, i) => v - b[i]); },
};

if (typeof module !== 'undefined' && module.exports) module.exports = { Mat };
