import numpy as np
from kalman_filter import KalmanFilter


class DroneSim:
    """
    Simulates a drone flying a rising helix.
    Four sensors (GPS, IMU, barometer, magnetometer) can be individually
    disabled at runtime to demonstrate Kalman filter redundancy.
    """

    RADIUS = 20.0   # helix radius (m)
    OMEGA  = 0.3    # angular velocity (rad/s)
    VZ     = 1.0    # climb rate (m/s)

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.t  = 0.0
        self.sensor_failed = {k: False for k in ("gps", "imu", "baro", "mag")}
        self.true_trail = []
        self.est_trail  = []
        self._init_kf()

    # ── Kalman init ───────────────────────────────────────────────────────────

    def _init_kf(self) -> None:
        self.kf = KalmanFilter(dt=self.dt)
        self.kf.x = np.array(self._true_state(0.0))

    # ── Physics ───────────────────────────────────────────────────────────────

    def _true_state(self, t: float) -> list:
        R, w, vz = self.RADIUS, self.OMEGA, self.VZ
        return [
            R * np.cos(w * t),         R * np.sin(w * t),         vz * t,
            -R * w * np.sin(w * t),    R * w * np.cos(w * t),    vz,
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle_sensor(self, name: str) -> None:
        if name in self.sensor_failed:
            self.sensor_failed[name] = not self.sensor_failed[name]

    def set_sensor_failed(self, name: str, failed: bool) -> None:
        if name in self.sensor_failed:
            self.sensor_failed[name] = failed

    def step(self) -> dict:
        self.t += self.dt
        truth = np.array(self._true_state(self.t))

        # Predict (dead-reckoning if all sensors are off)
        self.kf.predict()

        # Update with each active sensor
        readings = {}
        for name, failed in self.sensor_failed.items():
            if not failed:
                cfg = self.kf.sensors[name]
                noise = np.array([np.random.normal(0, s) for s in cfg["noise_std"]])
                z = cfg["H"] @ truth + noise
                self.kf.update(name, z)
                readings[name] = z.tolist()
            else:
                readings[name] = None

        est = self.kf.x.tolist()

        # Rolling trail buffer (last 150 steps)
        self.true_trail.append(truth[:3].tolist())
        self.est_trail.append(est[:3])
        if len(self.true_trail) > 150:
            self.true_trail.pop(0)
            self.est_trail.pop(0)

        active = sum(1 for v in self.sensor_failed.values() if not v)

        return {
            "t":          round(self.t, 2),
            "true":       truth.tolist(),
            "est":        est,
            "readings":   readings,
            "failed":     dict(self.sensor_failed),
            "true_trail": [list(p) for p in self.true_trail[-80:]],
            "est_trail":  [list(p) for p in self.est_trail[-80:]],
            "error":      float(np.linalg.norm(truth[:3] - np.array(est[:3]))),
            "uncertainty": self.kf.position_uncertainty(),
            "active":     active,
        }

    def reset(self) -> None:
        self.t = 0.0
        self.true_trail.clear()
        self.est_trail.clear()
        for k in self.sensor_failed:
            self.sensor_failed[k] = False
        self._init_kf()
