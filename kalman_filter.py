import numpy as np


class KalmanFilter:
    """
    Linear Kalman filter.
    State vector: [x, y, z, vx, vy, vz] (position + velocity, 6-DOF).
    Supports dynamic sensor activation / deactivation for redundancy demo.
    """

    def __init__(self, dt: float = 0.05, process_noise: float = 1.5):
        self.dt = dt
        n = 6

        self.x = np.zeros(n)          # state estimate
        self.P = np.eye(n) * 100.0    # covariance

        # State-transition matrix (constant-velocity model)
        self.F = np.eye(n)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Process-noise matrix (discretised white-noise acceleration)
        q, t2, t3, t4 = process_noise, dt**2, dt**3, dt**4
        self.Q = q * np.block([
            [t4 / 4 * np.eye(3), t3 / 2 * np.eye(3)],
            [t3 / 2 * np.eye(3), t2       * np.eye(3)],
        ])

        # Sensor configurations
        self.sensors = {
            "gps": {
                "H":         np.array([[1,0,0,0,0,0],
                                       [0,1,0,0,0,0],
                                       [0,0,1,0,0,0]], dtype=float),
                "R":         np.diag([4.0, 4.0, 4.0]),
                "noise_std": [2.0, 2.0, 2.0],
            },
            "imu": {
                "H":         np.array([[0,0,0,1,0,0],
                                       [0,0,0,0,1,0],
                                       [0,0,0,0,0,1]], dtype=float),
                "R":         np.diag([0.25, 0.25, 0.25]),
                "noise_std": [0.5, 0.5, 0.5],
            },
            "baro": {
                "H":         np.array([[0,0,1,0,0,0]], dtype=float),
                "R":         np.array([[0.25]]),
                "noise_std": [0.5],
            },
            "mag": {
                "H":         np.array([[1,0,0,0,0,0],
                                       [0,1,0,0,0,0]], dtype=float),
                "R":         np.diag([9.0, 9.0]),
                "noise_std": [3.0, 3.0],
            },
        }

    # ── Kalman steps ──────────────────────────────────────────────────────────

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, sensor_name: str, z: np.ndarray) -> None:
        H = self.sensors[sensor_name]["H"]
        R = self.sensors[sensor_name]["R"]
        innov = z - H @ self.x
        S     = H @ self.P @ H.T + R
        K     = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ H) @ self.P

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty from covariance diagonal."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1] + self.P[2, 2]))
