import numpy as np

from kalman_filter import KalmanFilter


# ── Tiny two-layer MLP ────────────────────────────────────────────────────────

class _MLP:
    """
    Two-layer fully-connected network with ReLU hidden layer.
    Predicts log(R_diagonal) from a flattened innovation window.
    All maths is plain numpy — no framework dependency.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        # Input normalisation stats — set during training
        self.x_mean = np.zeros(n_in)
        self.x_std  = np.ones(n_in)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Single-sample inference.  x: (n_in,) → (n_out,)."""
        xn = (x - self.x_mean) / self.x_std
        h  = np.maximum(0.0, xn @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward_batch(self, X: np.ndarray):
        """Batch forward.  X: (N, n_in) → out (N,n_out), h (N,n_hidden), Xn (N,n_in)."""
        Xn = (X - self.x_mean) / self.x_std
        H  = np.maximum(0.0, Xn @ self.W1 + self.b1)
        return H @ self.W2 + self.b2, H, Xn

    # ── Backward (mini-batch SGD) ─────────────────────────────────────────────

    def step(self, Xn: np.ndarray, H: np.ndarray,
             Y_pred: np.ndarray, Y_true: np.ndarray, lr: float) -> float:
        """One gradient step.  Inputs are already normalised batch arrays."""
        N   = len(Xn)
        dL  = (Y_pred - Y_true) / N        # (N, n_out)
        loss = float(np.mean((Y_pred - Y_true) ** 2))

        dW2  = H.T @ dL                    # (n_hidden, n_out)
        db2  = dL.sum(axis=0)
        dh   = dL @ self.W2.T              # (N, n_hidden)
        dh  *= (H > 0)                     # ReLU mask
        dW1  = Xn.T @ dh                   # (n_in, n_hidden)
        db1  = dh.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss


# ── KalmanNET ─────────────────────────────────────────────────────────────────

class KalmanNET:
    """
    R-adaptive Kalman filter.

    Each sensor has a small MLP that maps a sliding window of recent
    innovations → log(R_diagonal).  At inference only the live innovation
    sequence is required; no explicit knowledge of the environment is used.

    Training uses oracle R targets computed from the simulation's known
    noise formulas — the supervision signal available during a calibration
    flight where ground truth exists.
    """

    WINDOW   = 20     # innovation history length  (1 s at 20 Hz)
    HIDDEN   = 16
    EPOCHS   = 250
    BATCH    = 64
    LR_INIT  = 8e-3
    LR_FINAL = 1e-3

    # Calibrated (calm-condition) R diagonals — same as KalmanFilter defaults
    R_CALIB: dict = {
        "gps":  np.array([4.0,  4.0,  4.0 ]),
        "imu":  np.array([0.25, 0.25, 0.25]),
        "baro": np.array([0.25]),
        "mag":  np.array([9.0,  9.0  ]),
    }
    _DIMS: dict = {"gps": 3, "imu": 3, "baro": 1, "mag": 2}

    def __init__(self, dt: float = 0.05):
        self.dt      = dt
        self.trained = False
        self._nets: dict  = {s: None for s in self.R_CALIB}
        self._wins: dict  = {s: []   for s in self.R_CALIB}
        self._r_hat: dict = {s: self.R_CALIB[s].tolist() for s in self.R_CALIB}
        self._reset_kf()

    def _reset_kf(self) -> None:
        self.kf = KalmanFilter(dt=self.dt)

    def reset(self, h0: list) -> None:
        self._reset_kf()
        self.kf.x = np.array(h0)
        self._wins = {s: [] for s in self.R_CALIB}

    # ── Runtime ───────────────────────────────────────────────────────────────

    def predict(self) -> None:
        self.kf.predict()

    def update(self, sensor_name: str, z: np.ndarray) -> np.ndarray:
        """Update with measurement z; returns the pre-update innovation."""
        H     = self.kf.sensors[sensor_name]["H"]
        innov = z - H @ self.kf.x

        win = self._wins[sensor_name]
        win.append(innov.copy())
        if len(win) > self.WINDOW:
            win.pop(0)

        R_eff = self._effective_R(sensor_name)
        S = H @ self.kf.P @ H.T + R_eff
        K = self.kf.P @ H.T @ np.linalg.inv(S)
        self.kf.x = self.kf.x + K @ innov
        self.kf.P = (np.eye(6) - K @ H) @ self.kf.P
        return innov

    def _effective_R(self, s: str) -> np.ndarray:
        net   = self._nets[s]
        calib = self.R_CALIB[s]
        m     = self._DIMS[s]

        if net is None:
            return np.diag(calib)

        win    = self._wins[s]
        padded = np.zeros((self.WINDOW, m))
        if win:
            padded[-len(win):] = np.array(win)

        log_r  = net.forward(padded.flatten())
        r_diag = np.exp(log_r)
        # Never trust a sensor more than calibrated (R floor = 80 % of calib)
        r_diag = np.clip(r_diag, 0.8 * calib, 500.0 * calib)
        self._r_hat[s] = r_diag.tolist()
        return np.diag(r_diag)

    def position_uncertainty(self) -> float:
        return float(np.sqrt(self.kf.P[0, 0] + self.kf.P[1, 1] + self.kf.P[2, 2]))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, buffer: list) -> None:
        """
        Train per-sensor MLPs from a collected trajectory buffer.

        buffer items: {
          "innovations": {sensor_name: [float, ...]},
          "env":         {"wind_speed": float, "temperature": float}
        }
        """
        for s in self.R_CALIB:
            m    = self._DIMS[s]
            n_in = self.WINDOW * m

            xs, ys = [], []
            for i in range(self.WINDOW, len(buffer)):
                rows, ok = [], True
                for j in range(i - self.WINDOW, i):
                    row = buffer[j]["innovations"].get(s)
                    if row is None:
                        ok = False; break
                    rows.append(row)
                if not ok:
                    continue

                x_in  = np.array(rows, dtype=float).flatten()
                r_tgt = self._oracle_R(s, buffer[i]["env"])
                xs.append(x_in)
                ys.append(np.log(r_tgt + 1e-9))

            if len(xs) < self.BATCH:
                continue

            X = np.array(xs, dtype=float)   # (N, n_in)
            Y = np.array(ys, dtype=float)   # (N, m)

            net = _MLP(n_in, self.HIDDEN, m, seed=abs(hash(s)) % 2**31)
            net.x_mean = X.mean(axis=0)
            net.x_std  = X.std(axis=0) + 1e-8

            N = len(X)
            for epoch in range(self.EPOCHS):
                t   = epoch / self.EPOCHS
                lr  = self.LR_INIT * (1 - t) + self.LR_FINAL * t
                idx = np.random.permutation(N)
                for start in range(0, N, self.BATCH):
                    b       = idx[start:start + self.BATCH]
                    out, H_act, Xn = net.forward_batch(X[b])
                    net.step(Xn, H_act, out, Y[b], lr)

            self._nets[s] = net

        self.trained = True

    @staticmethod
    def _oracle_R(sensor: str, env: dict) -> np.ndarray:
        """
        Ground-truth effective R given the simulation's known noise formulas.
        Used only during training — not available at inference time.
        """
        w  = env.get("wind_speed",  0.0)
        dT = env.get("temperature", 20.0) - 20.0
        if sensor == "gps":
            s = 2.0 + 0.12 * w
            return np.array([s*s, s*s, s*s])
        if sensor == "imu":
            s = 0.5 + 0.04 * w + 0.008 * abs(dT)
            return np.array([s*s, s*s, s*s])
        if sensor == "baro":
            s    = 0.5 + 0.008 * abs(dT)
            bias = dT * 0.12
            return np.array([s * s + bias * bias])
        # mag: static noise in this simulation
        return np.array([9.0, 9.0])
