"""
Export the trained GRU KalmanNET to web/weights.json and generate PyTorch
reference vectors for the browser port's parity tests (web/test/parity.mjs).

Run inside the venv (needs torch + numpy and an existing kalmannet_model.pt):
    .venv/Scripts/python.exe export_web_weights.py

See BROWSER_PORT_PLAN.md. Regenerate after any retraining you want the web
build to ship.
"""
import json
from pathlib import Path

import numpy as np
import torch

from kalman_filter import KalmanFilter
from kalmannet import (
    KalmanNET, _GRUKalmanNET, MODEL_PATH,
    SENSOR_ORDER, SENSOR_DIMS, HIDDEN_DIM,
)

ROOT     = Path(__file__).parent
WEB_DIR  = ROOT / "web"
TEST_DIR = WEB_DIR / "test"

RADIUS, OMEGA, VZ = 20.0, 0.3, 1.0   # must match DroneSim


def _helix(t: float) -> np.ndarray:
    return np.array([
        RADIUS * np.cos(OMEGA * t),          RADIUS * np.sin(OMEGA * t),          VZ * t,
        -RADIUS * OMEGA * np.sin(OMEGA * t), RADIUS * OMEGA * np.cos(OMEGA * t),  VZ,
    ])


def _active_sensors(i: int) -> list:
    """Deterministic periodic dropouts so references exercise partial updates."""
    active = list(SENSOR_ORDER)
    if i % 7 == 3:
        active.remove("gps")
    if i % 11 == 5:
        active.remove("baro")
    if i % 13 == 8:
        active.remove("imu")
    return active


def export_weights(state: dict) -> None:
    def tolist(key):
        return state[key].numpy().astype(float).tolist()

    weights = {
        "format": "gru-kalmannet-v1",
        "input":  9,
        "hidden": HIDDEN_DIM,
        "gru": {
            "w_ih": tolist("gru.weight_ih_l0"),   # (192, 9), gate rows r|z|n
            "w_hh": tolist("gru.weight_hh_l0"),   # (192, 64)
            "b_ih": tolist("gru.bias_ih_l0"),
            "b_hh": tolist("gru.bias_hh_l0"),
        },
        "heads": {
            s: {"w": tolist(f"heads.{s}.weight"), "b": tolist(f"heads.{s}.bias")}
            for s in SENSOR_ORDER
        },
    }
    out = WEB_DIR / "weights.json"
    with open(out, "w") as f:
        json.dump(weights, f)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


def export_gru_reference(state: dict) -> None:
    """Seeded innovation sequence through the raw net (pre-clip K outputs)."""
    net = _GRUKalmanNET()
    net.load_state_dict(state)
    net.eval()
    rng = np.random.default_rng(42)
    h = None
    steps = []
    for _ in range(100):
        iv9 = rng.normal(0.0, 2.0, 9).astype(np.float32)
        with torch.no_grad():
            K_out, h = net(torch.from_numpy(iv9).view(1, 1, 9), h)
        steps.append({
            "iv9": iv9.astype(float).tolist(),
            "K":   {s: K_out[s][0].numpy().astype(float).tolist() for s in SENSOR_ORDER},
        })
    out = TEST_DIR / "reference_gru.json"
    with open(out, "w") as f:
        json.dump({"steps": steps,
                   "h_final": h.squeeze().numpy().astype(float).tolist()}, f)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


def export_kf_reference() -> None:
    """Classical KF driven by a seeded measurement sequence with dropouts."""
    kf = KalmanFilter()
    kf.x = _helix(0.0)
    rng = np.random.default_rng(7)
    steps = []
    for i in range(200):
        truth  = _helix((i + 1) * 0.05)
        active = _active_sensors(i)
        z = {}
        for name in active:
            cfg   = kf.sensors[name]
            noise = rng.normal(0.0, cfg["noise_std"])
            z[name] = (cfg["H"] @ truth + noise).tolist()
        kf.predict()
        for name in active:
            kf.update(name, np.array(z[name]))
        steps.append({"z": z, "active": active, "x": kf.x.tolist()})
    out = TEST_DIR / "reference_kf.json"
    with open(out, "w") as f:
        json.dump({"x0": _helix(0.0).tolist(), "steps": steps,
                   "P_final": kf.P.tolist()}, f)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


def export_knstep_reference() -> None:
    """Trained KalmanNET end-to-end: helix + baro bias + dropouts, 300 steps."""
    kn = KalmanNET()
    assert kn.trained, "kalmannet_model.pt missing or failed to load — train first"
    kn.reset(_helix(0.0).tolist())
    ref_kf = KalmanFilter()   # only for sensor configs
    rng = np.random.default_rng(99)
    baro_bias = -25.0 * 0.25   # ΔT = +25 °C — the bias regime KN was trained for
    steps = []
    for i in range(300):
        truth  = _helix((i + 1) * 0.05)
        active = _active_sensors(i)
        z = {}
        for name in active:
            cfg   = ref_kf.sensors[name]
            noise = rng.normal(0.0, cfg["noise_std"])
            zv    = cfg["H"] @ truth + noise
            if name == "baro":
                zv[0] += baro_bias
            z[name] = zv.tolist()
        kn.step({name: np.array(v) for name, v in z.items()})
        steps.append({
            "z": z, "active": active,
            "x": kn._kf.x.tolist(),
            "k_ratio": {s: list(kn._k_ratio[s]) for s in SENSOR_ORDER},
        })
    out = TEST_DIR / "reference_knstep.json"
    with open(out, "w") as f:
        json.dump({"x0": _helix(0.0).tolist(), "baro_bias": baro_bias,
                   "steps": steps}, f)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit("kalmannet_model.pt not found — train in the app first.")
    WEB_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    export_weights(state)
    export_gru_reference(state)
    export_kf_reference()
    export_knstep_reference()
    print("done.")
