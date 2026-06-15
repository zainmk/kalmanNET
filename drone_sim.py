import math

import numpy as np

from kalman_filter import KalmanFilter


class DroneSim:
    """
    Drone flying a rising helix, perturbed by controllable environmental factors.
    Wind physically deflects the true trajectory and raises GPS/IMU noise.
    Temperature biases the barometer and increases IMU thermal drift.
    The Kalman filter's R matrices are fixed at calm-condition values — the
    growing mismatch is exactly what KalmanNET would learn to correct.
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
        h0 = self._true_helix(0.0)
        self.raw_pos = [h0[0], h0[1], h0[2]]

        # Environmental state (user-controlled)
        self.wind_speed   = 0.0   # m/s, 0–20
        self.wind_heading = 0.0   # radians
        self.temperature  = 20.0  # °C absolute; calibration point is 20°C
        self.wind_disp    = [0.0, 0.0]  # accumulated horizontal displacement (m)

        self._init_kf()

    # ── Kalman init ───────────────────────────────────────────────────────────

    def _init_kf(self) -> None:
        self.kf = KalmanFilter(dt=self.dt)
        self.kf.x = np.array(self._true_helix(0.0))

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
        # Drone autopilot fights wind but can't fully compensate.
        # 0.97 decay = partial recovery; bounded horizontal drift results.
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

        # ── True state = helix + wind perturbation ────────────────────────────
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
        # These exceed the Kalman filter's fixed R — the intentional mismatch.
        dT = self.temperature - 20.0  # deviation from 20 °C calibration point
        eff_std = {
            "gps":  [2.0 + 0.12 * self.wind_speed] * 3,
            "imu":  [0.5 + 0.04 * self.wind_speed + 0.008 * abs(dT)] * 3,
            "baro": [0.5 + 0.008 * abs(dT)],
            "mag":  [3.0, 3.0],
        }
        # Temperature-induced barometer bias: hot air = lower pressure = reads too low
        baro_bias = -dT * 0.12  # metres per °C deviation (hot → reads low, cold → reads high)

        # ── Kalman predict ────────────────────────────────────────────────────
        self.kf.predict()

        # ── Update with each active sensor ────────────────────────────────────
        readings = {}
        for name, failed in self.sensor_failed.items():
            if not failed:
                cfg   = self.kf.sensors[name]
                noise = np.array([np.random.normal(0, s) for s in eff_std[name]])
                z     = cfg["H"] @ truth + noise
                if name == "baro":
                    z[0] += baro_bias
                self.kf.update(name, z)
                readings[name] = z.tolist()
            else:
                readings[name] = None

        est = self.kf.x.tolist()

        # ── Raw position estimate — direct sensor readings, no filtering ───────
        # Priority: GPS > MAG (XY), GPS > BARO (Z). Holds last known if nothing available.
        raw = list(self.raw_pos)
        if readings['gps']:
            raw = [readings['gps'][0], readings['gps'][1], readings['gps'][2]]
        else:
            if readings['mag']:
                raw[0] = readings['mag'][0]
                raw[1] = readings['mag'][1]
            if readings['baro']:
                raw[2] = readings['baro'][0]
        self.raw_pos = raw

        # ── Rolling trail buffer (last 150 steps) ─────────────────────────────
        self.true_trail.append(truth[:3].tolist())
        self.est_trail.append(est[:3])
        self.raw_trail.append(list(raw))
        if len(self.true_trail) > 150:
            self.true_trail.pop(0)
            self.est_trail.pop(0)
            self.raw_trail.pop(0)

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
            "error":       float(np.linalg.norm(truth[:3] - np.array(est[:3]))),
            "uncertainty": self.kf.position_uncertainty(),
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
        for k in self.sensor_failed:
            self.sensor_failed[k] = False
        self.wind_disp = [0.0, 0.0]
        h0 = self._true_helix(0.0)
        self.raw_pos = [h0[0], h0[1], h0[2]]
        self._init_kf()
