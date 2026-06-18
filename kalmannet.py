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
TOTAL_MEAS_DIM = 9
STATE_DIM      = 6
HIDDEN_DIM     = 64


class _GRUKalmanNET(nn.Module):
    """
    Shared GRU + per-sensor K heads.
    Called once per filter step with the full 9D innovation vector
    (all active sensors combined, zeros for inactive ones).
    Outputs K ∈ (0, 2) per element via 2·sigmoid.
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
        innov_9d : (seq_len, 1, 9)
        Returns  : K dict {sensor: (seq_len, STATE_DIM, sensor_dim)}, new h
        """
        out, h_new = self.gru(innov_9d, h)
        out = out.squeeze(1)
        K_out = {
            s: 2.0 * torch.sigmoid(self.heads[s](out)).view(-1, STATE_DIM, SENSOR_DIMS[s])
            for s in SENSOR_ORDER
        }
        return K_out, h_new


class KalmanNET:
    """
    GRU-based KalmanNET. One GRU call per filter step (all active sensors
    combined into a 9D innovation vector). Falls back to standard KF when
    untrained.
    """

    def __init__(self, dt: float = 0.05):
        self.dt      = dt
        self.trained = False
        self._net    = _GRUKalmanNET()
        self._h      = None
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

    # ── Main interface: one call per sim step ─────────────────────────────────

    def step(self, z_values: dict) -> None:
        """
        Full predict + update for all active sensors in one GRU call.
        z_values: {sensor_name: np.ndarray} for every active sensor this step.
        """
        self._kf.predict()

        if not z_values:
            return

        if not self.trained:
            self._kf_fallback(z_values)
            return

        # Build 9D innovation vector (all sensors combined)
        iv9        = np.zeros(TOTAL_MEAS_DIM, dtype=np.float32)
        innov_dict = {}
        for name, z in z_values.items():
            H     = self._kf.sensors[name]["H"]
            innov = z - H @ self._kf.x
            innov_dict[name] = innov
            o     = SENSOR_OFFSETS[name]
            iv9[o:o + SENSOR_DIMS[name]] = np.clip(innov, -1e4, 1e4).astype(np.float32)

        # Safety: reset if state has diverged (prevents overflow cascade)
        if np.any(~np.isfinite(self._kf.x)) or np.any(np.abs(self._kf.x) > 1e5):
            self._kf.x = np.zeros(STATE_DIM)
            self._kf.P = np.eye(STATE_DIM) * 10.0
            self._h    = None

        # Single GRU call for this step
        with torch.no_grad():
            x_in       = torch.from_numpy(iv9).unsqueeze(0).unsqueeze(0)  # (1,1,9)
            K_out, self._h = self._net(x_in, self._h)

        # Apply K for each active sensor in fixed order
        for name in SENSOR_ORDER:
            if name not in z_values:
                continue
            H     = self._kf.sensors[name]["H"]
            innov = innov_dict[name]

            # Riccati K computed first — used both for display and as the K bound.
            R_cal  = self._kf.sensors[name]["R"]
            S_ref  = H @ self._kf.P @ H.T + R_cal
            K_ref  = self._kf.P @ H.T @ np.linalg.inv(S_ref)

            K      = K_out[name][0].numpy()              # (STATE_DIM, m), range (0, 2)
            # Clip element-wise to ≤ 2 × |K_ref|.
            # K_ref has near-zero off-diagonal elements, so this suppresses the
            # dense off-diagonal K that the GRU can produce and that causes
            # (I − K@H) to have eigenvalues outside ±1 → P blow-up.
            K      = np.clip(K, 0.0, np.abs(K_ref) * 2 + 1e-6)

            k_norm = np.linalg.norm(K,     axis=0) + 1e-9
            r_norm = np.linalg.norm(K_ref, axis=0) + 1e-9
            self._k_ratio[name] = (k_norm / r_norm).tolist()

            self._kf.x = self._kf.x + K @ innov
            I_KH = np.eye(STATE_DIM) - K @ H
            self._kf.P = I_KH @ self._kf.P @ I_KH.T + K @ R_cal @ K.T
            self._kf.P = 0.5 * (self._kf.P + self._kf.P.T)

    def _kf_fallback(self, z_values: dict) -> None:
        for name in SENSOR_ORDER:
            if name not in z_values:
                continue
            H     = self._kf.sensors[name]["H"]
            R     = self._kf.sensors[name]["R"]
            innov = z_values[name] - H @ self._kf.x
            S     = H @ self._kf.P @ H.T + R
            K     = self._kf.P @ H.T @ np.linalg.inv(S)
            self._kf.x = self._kf.x + K @ innov
            self._kf.P = (np.eye(STATE_DIM) - K @ H) @ self._kf.P

    def position_uncertainty(self) -> float:
        return float(np.sqrt(max(0.0, self._kf.P[0, 0] + self._kf.P[1, 1] + self._kf.P[2, 2])))
