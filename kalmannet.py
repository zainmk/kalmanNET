from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is required for KalmanNET GRU. "
        "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e

from kalman_filter import KalmanFilter

MODEL_PATH = Path(__file__).parent / "kalmannet_model.pt"

SENSOR_ORDER   = ["gps", "imu", "baro", "mag"]
SENSOR_DIMS    = {"gps": 3, "imu": 3, "baro": 1, "mag": 2}
SENSOR_OFFSETS = {"gps": 0, "imu": 3, "baro": 6, "mag": 7}
TOTAL_MEAS_DIM = 9    # 3 + 3 + 1 + 2
STATE_DIM      = 6
HIDDEN_DIM     = 64


class _GRUKalmanNET(nn.Module):
    """
    Shared GRU with per-sensor K-matrix output heads.

    Input:  9D concatenated innovation vector (zero-padded for inactive sensors)
    Hidden: 64D context — accumulates temporal patterns across filter steps
    Output: per-sensor K matrices ∈ (0, 2)^(6×m) via 2·sigmoid activation

    The sigmoid bound ensures K is always positive and caps at 2× per element,
    preventing runaway updates while allowing amplification (K > 1, e.g. GPS in wind)
    or dampening (K < 1, e.g. BARO in heat).
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(TOTAL_MEAS_DIM, HIDDEN_DIM, batch_first=False)
        self.heads = nn.ModuleDict({
            s: nn.Linear(HIDDEN_DIM, STATE_DIM * SENSOR_DIMS[s])
            for s in SENSOR_ORDER
        })

    def forward(self, innov_9d, h=None):
        """
        innov_9d : (seq_len, 1, 9) tensor  — batch dim must be 1
        h        : (1, 1, HIDDEN_DIM) or None
        Returns  : {sensor: (seq_len, STATE_DIM, sensor_dim)} K tensors, new h
        """
        out, h_new = self.gru(innov_9d, h)          # out: (T, 1, hidden)
        out = out.squeeze(1)                          # (T, hidden)
        K_out = {}
        for s in SENSOR_ORDER:
            raw   = self.heads[s](out)                # (T, state_dim * sensor_dim)
            K_out[s] = 2.0 * torch.sigmoid(raw).view(-1, STATE_DIM, SENSOR_DIMS[s])
        return K_out, h_new


class KalmanNET:
    """
    GRU-based KalmanNET runtime.

    Predicts the Kalman gain K directly from the temporal pattern of innovations,
    replacing the fixed Riccati-optimal K. Trained end-to-end on position MSE
    so K can go above the Riccati value (trust sensor more — wind) or below
    (trust sensor less — temperature bias), which R-prediction cannot do.

    Falls back to standard KF equations when no trained model is loaded.
    """

    def __init__(self, dt: float = 0.05):
        self.dt      = dt
        self.trained = False
        self._net    = _GRUKalmanNET()
        self._h      = None                       # GRU hidden state (1, 1, HIDDEN_DIM)
        self._kf     = KalmanFilter(dt=dt)
        self._k_ratio = {s: [1.0] * SENSOR_DIMS[s] for s in SENSOR_ORDER}
        if MODEL_PATH.exists():
            self._load()

    def _load(self) -> bool:
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            self._net.load_state_dict(state)
            self._net.eval()
            self.trained = True
            return True
        except Exception as exc:
            print(f"[KalmanNET] Failed to load model: {exc}")
            return False

    def reset(self, h0: list) -> None:
        self._kf   = KalmanFilter(dt=self.dt)
        self._kf.x = np.array(h0)
        self._h    = None
        self._k_ratio = {s: [1.0] * SENSOR_DIMS[s] for s in SENSOR_ORDER}

    # ── Runtime ───────────────────────────────────────────────────────────────

    def predict(self) -> None:
        self._kf.predict()

    def update(self, sensor_name: str, z: np.ndarray) -> None:
        H     = self._kf.sensors[sensor_name]["H"]
        innov = z - H @ self._kf.x

        if not self.trained:
            R = self._kf.sensors[sensor_name]["R"]
            S = H @ self._kf.P @ H.T + R
            K = self._kf.P @ H.T @ np.linalg.inv(S)
            self._kf.x = self._kf.x + K @ innov
            self._kf.P = (np.eye(STATE_DIM) - K @ H) @ self._kf.P
            return

        # ── Build 9D innovation vector (this sensor's slot only) ──────────
        iv9    = np.zeros(TOTAL_MEAS_DIM, dtype=np.float32)
        offset = SENSOR_OFFSETS[sensor_name]
        iv9[offset:offset + SENSOR_DIMS[sensor_name]] = innov.astype(np.float32)

        # ── GRU step ──────────────────────────────────────────────────────
        with torch.no_grad():
            x_in       = torch.from_numpy(iv9).unsqueeze(0).unsqueeze(0)   # (1,1,9)
            K_out, self._h = self._net(x_in, self._h)

        K = K_out[sensor_name][0].numpy()          # (STATE_DIM, sensor_dim)

        # ── K-ratio for display (GRU K vs Riccati K from current P) ───────
        R_cal  = self._kf.sensors[sensor_name]["R"]
        S_ref  = H @ self._kf.P @ H.T + R_cal
        K_ref  = self._kf.P @ H.T @ np.linalg.inv(S_ref)
        k_norm = np.linalg.norm(K,     axis=0) + 1e-9
        r_norm = np.linalg.norm(K_ref, axis=0) + 1e-9
        self._k_ratio[sensor_name] = (k_norm / r_norm).tolist()

        # ── Apply GRU K ───────────────────────────────────────────────────
        self._kf.x = self._kf.x + K @ innov

        # Joseph form ensures P stays positive semi-definite with arbitrary K
        I_KH = np.eye(STATE_DIM) - K @ H
        self._kf.P = I_KH @ self._kf.P @ I_KH.T + K @ R_cal @ K.T
        self._kf.P = 0.5 * (self._kf.P + self._kf.P.T)    # symmetrize

    def position_uncertainty(self) -> float:
        return float(np.sqrt(max(0.0, self._kf.P[0, 0] + self._kf.P[1, 1] + self._kf.P[2, 2])))
