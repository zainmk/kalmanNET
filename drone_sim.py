import math

import numpy as np

from kalman_filter import KalmanFilter
from kalmannet    import KalmanNET


class DroneSim:
    """
    Drone flying a rising helix, perturbed by controllable environmental factors.
    Wind physically deflects the true trajectory and raises GPS/IMU noise.
    Temperature biases the barometer and increases IMU thermal drift.
    The standard Kalman filter's R matrices are fixed at calm-condition values —
    the growing mismatch is exactly what KalmanNET learns to correct.
    """

    RADIUS = 20.0
    OMEGA  = 0.3
    VZ     = 1.0

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.t  = 0.0
        self.sensor_failed = {k: False for k in ("gps", "imu", "baro", "mag")}
        self.true_trail = []
        self.est_trail  = []
        self.raw_trail  = []
        self.kn_trail   = []
        h0 = self._true_helix(0.0)
        self.raw_pos = [h0[0], h0[1], h0[2]]

        # Environmental state (user-controlled)
        self.wind_speed   = 0.0   # m/s, 0–20
        self.wind_heading = 0.0   # radians
        self.temperature  = 20.0  # °C absolute; calibration point is 20°C
        self.wind_disp    = [0.0, 0.0]  # accumulated horizontal drift from wind

        # KalmanNET training data collection
        self.collecting_data  = False
        self.training_buffer  = []

        self._init_filters()

    # ── Filter init ───────────────────────────────────────────────────────────

    def _init_filters(self) -> None:
        h0 = self._true_helix(0.0)
        self.kf = KalmanFilter(dt=self.dt)
        self.kf.x = np.array(h0)
        # Preserve the trained network across resets — only reset filter state
        if hasattr(self, 'kn') and self.kn.trained:
            self.kn.reset(h0)
        else:
            self.kn = KalmanNET(dt=self.dt)
            self.kn.reset(h0)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _true_helix(self, t: float) -> list:
        """Analytical helix with no environmental perturbation."""
        R, w, vz = self.RADIUS, self.OMEGA, self.VZ
        return [
            R * np.cos(w * t),       R * np.sin(w * t),       vz * t,
            -R * w * np.sin(w * t),  R * w * np.cos(w * t),   vz,
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle_sensor(self, name: str) -> None:
        if name in self.sensor_failed:
            self.sensor_failed[name] = not self.sensor_failed[name]

    def set_sensor_failed(self, name: str, failed: bool) -> None:
        if name in self.sensor_failed:
            self.sensor_failed[name] = failed

    def set_environment(self, wind_speed: float, wind_heading: float, temperature: float) -> None:
        self.wind_speed   = max(0.0, min(20.0, wind_speed))
        self.wind_heading = wind_heading  # radians
        self.temperature  = max(-10.0, min(50.0, temperature))

    def step(self) -> dict:
        self.t += self.dt

        # ── Wind drift ────────────────────────────────────────────────────────
        prev_disp = list(self.wind_disp)
        if self.wind_speed > 0:
            gust_x = (self.wind_speed * math.cos(self.wind_heading)
                      + np.random.normal(0, self.wind_speed * 0.1))
            gust_y = (self.wind_speed * math.sin(self.wind_heading)
                      + np.random.normal(0, self.wind_speed * 0.1))
        else:
            gust_x, gust_y = 0.0, 0.0
        self.wind_disp[0] = 0.97 * self.wind_disp[0] + gust_x * self.dt * 0.08
        self.wind_disp[1] = 0.97 * self.wind_disp[1] + gust_y * self.dt * 0.08
        wind_vel_x = (self.wind_disp[0] - prev_disp[0]) / self.dt
        wind_vel_y = (self.wind_disp[1] - prev_disp[1]) / self.dt

        # ── True state (helix + wind displacement) ────────────────────────────
        helix = self._true_helix(self.t)
        truth = np.array([
            helix[0] + self.wind_disp[0],
            helix[1] + self.wind_disp[1],
            helix[2],
            helix[3] + wind_vel_x,
            helix[4] + wind_vel_y,
            helix[5],
        ])

        # ── Effective sensor noise (environment-scaled) ───────────────────────
        dT = self.temperature - 20.0
        eff_std = {
            "gps":  [2.0 + 0.12 * self.wind_speed] * 3,
            "imu":  [0.5 + 0.04 * self.wind_speed + 0.008 * abs(dT)] * 3,
            "baro": [0.5 + 0.008 * abs(dT)],
            "mag":  [3.0, 3.0],
        }
        # Temperature-induced barometer bias: in warm air, pressure at a given
        # altitude is higher than the standard-atmosphere model assumes, so the
        # altimeter converts it to a lower indicated altitude — hot reads low.
        baro_bias = -dT * 0.25

        # ── Generate sensor readings ───────────────────────────────────────────
        readings = {}
        z_values = {}

        for name, failed in self.sensor_failed.items():
            if not failed:
                cfg   = self.kf.sensors[name]
                noise = np.array([np.random.normal(0, s) for s in eff_std[name]])
                z     = cfg["H"] @ truth + noise
                if name == "baro":
                    z[0] += baro_bias
                z_values[name] = z
                readings[name] = z.tolist()
            else:
                readings[name] = None

        # ── Update filters ────────────────────────────────────────────────────
        self.kf.predict()
        for name, z in z_values.items():
            self.kf.update(name, z)

        self.kn.step(z_values)   # predict + all-sensor update in one GRU call

        est    = self.kf.x.tolist()
        kn_est = self.kn._kf.x.tolist()

        # ── Collect training data (all sensors must be active) ────────────────
        if self.collecting_data and not any(self.sensor_failed.values()):
            self.training_buffer.append({
                "measurements": {k: v.tolist() for k, v in z_values.items()},
                "x_true":       truth.tolist(),
            })

        # ── Raw position estimate ─────────────────────────────────────────────
        raw = list(self.raw_pos)
        if readings["gps"]:
            raw = readings["gps"][:3]
        else:
            if readings["mag"]:
                raw[0] = readings["mag"][0]
                raw[1] = readings["mag"][1]
            if readings["baro"]:
                raw[2] = readings["baro"][0]
        self.raw_pos = raw

        # ── Rolling trail buffers (last 150 steps) ────────────────────────────
        self.true_trail.append(truth[:3].tolist())
        self.est_trail.append(est[:3])
        self.raw_trail.append(list(raw))
        self.kn_trail.append(kn_est[:3])
        if len(self.true_trail) > 150:
            self.true_trail.pop(0)
            self.est_trail.pop(0)
            self.raw_trail.pop(0)
            self.kn_trail.pop(0)

        active = sum(1 for v in self.sensor_failed.values() if not v)

        return {
            "t":           round(self.t, 2),
            "true":        truth.tolist(),
            "est":         est,
            "readings":    readings,
            "failed":      dict(self.sensor_failed),
            "raw":         list(self.raw_pos),
            "true_trail":  [list(p) for p in self.true_trail[-80:]],
            "est_trail":   [list(p) for p in self.est_trail[-80:]],
            "raw_trail":   [list(p) for p in self.raw_trail[-80:]],
            "kn_est":      kn_est,
            "kn_trail":    [list(p) for p in self.kn_trail[-80:]],
            "kn_error":    float(np.linalg.norm(truth[:3] - np.array(kn_est[:3]))),
            "kn_k_ratio":  dict(self.kn._k_ratio),
            "kn_trained":  self.kn.trained,
            "error":       float(np.linalg.norm(truth[:3] - np.array(est[:3]))),
            "uncertainty": self.kf.position_uncertainty(),
            "kn_uncertainty": self.kn.position_uncertainty(),
            "active":      active,
            "env": {
                "wind_speed":   round(self.wind_speed, 1),
                "wind_heading": round(math.degrees(self.wind_heading), 1),
                "temperature":  round(self.temperature, 1),
            },
        }

    def reset(self) -> None:
        self.t = 0.0
        self.true_trail.clear()
        self.est_trail.clear()
        self.raw_trail.clear()
        self.kn_trail.clear()
        for k in self.sensor_failed:
            self.sensor_failed[k] = False
        self.wind_disp = [0.0, 0.0]
        h0 = self._true_helix(0.0)
        self.raw_pos = [h0[0], h0[1], h0[2]]
        self._init_filters()
