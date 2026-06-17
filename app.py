import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

from drone_sim import DroneSim

app = Flask(__name__)

sim      = DroneSim()
sim_lock = threading.Lock()

_state      = {}
_state_lock = threading.Lock()

_paused      = True
_paused_lock = threading.Lock()

_training_state = {"active": False, "phase": None, "progress": 0.0, "step": "idle"}
_training_lock  = threading.Lock()

_training_fast      = False   # when True the sim loop runs at 10× speed
_training_fast_lock = threading.Lock()

# ── Training sequence ─────────────────────────────────────────────────────────
# (phase_label, wind_m_s, temp_C, duration_s)
# Durations are wall-clock seconds at 10× sim speed (each second = ~200 sim steps).
_TRAIN_PHASES = [
    ("CALM BASELINE",     0,   20,   4),
    ("WIND  10 m/s",     10,   20,   3),
    ("WIND  20 m/s",     20,   20,   3),
    ("WIND   5 m/s",      5,   20,   2),
    ("CALM",              0,   20,   2),
    ("TEMP  40°C",        0,   40,   3),
    ("TEMP  50°C",        0,   50,   3),
    ("TEMP -10°C",        0,  -10,   3),
    ("COMBINED STRESS",  12,   40,   4),
    ("RETURN TO CALM",    0,   20,   2),
]


def _sim_loop() -> None:
    while True:
        with _paused_lock:
            paused = _paused
        if not paused:
            with sim_lock:
                state = sim.step()
            with _state_lock:
                _state.clear()
                _state.update(state)
        with _training_fast_lock:
            fast = _training_fast
        time.sleep(0.005 if fast else 0.05)  # 200 Hz during training, 20 Hz normally


threading.Thread(target=_sim_loop, daemon=True).start()


def _run_training() -> None:
    """Background thread: scripted env sequence → data collection → MLP training."""
    global _paused

    # Unblock sim and enable fast mode for data collection
    with _paused_lock:
        was_paused = _paused
        _paused = False
    with _training_fast_lock:
        global _training_fast
        _training_fast = True

    # Reset sim to a clean state
    with sim_lock:
        sim.reset()
        sim.training_buffer.clear()
        sim.collecting_data = True

    n = len(_TRAIN_PHASES)
    for i, (phase, wind, temp, dur) in enumerate(_TRAIN_PHASES):
        with _training_lock:
            _training_state.update({
                "active":    True,
                "phase":     phase,
                "phase_idx": i,
                "progress":  round(i / n, 3),
                "step":      "collecting",
                "wind":      wind,
                "temp":      temp,
            })
        with sim_lock:
            sim.set_environment(wind_speed=float(wind),
                                wind_heading=0.0,
                                temperature=float(temp))
        time.sleep(dur)

    # Stop collecting and return sim to normal speed
    with _training_fast_lock:
        _training_fast = False
    with sim_lock:
        sim.collecting_data = False
        buf = list(sim.training_buffer)

    with _training_lock:
        _training_state.update({
            "phase":    "OPTIMISING NETWORK",
            "progress": 0.95,
            "step":     "training",
            "wind":     0,
            "temp":     20,
        })

    # Save buffer for the training script
    buf_path = Path(__file__).parent / "training_buffer.json"
    with open(buf_path, "w") as f:
        json.dump(buf, f)

    # Run training script as a subprocess (keeps app.py responsive; ~15–30s on CPU)
    script = Path(__file__).parent / "train_kalmannet_gru.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )

    if result.returncode == 0:
        with sim_lock:
            sim.kn._load()
    else:
        print("[training] GRU training failed:\n", result.stderr[-1000:])

    # Pause immediately so the sim loop stops while we reset
    with _paused_lock:
        _paused = True

    # Reset drone to initial position (trained KN is preserved by _init_filters)
    with sim_lock:
        sim.set_environment(wind_speed=0.0, wind_heading=0.0, temperature=20.0)
        sim.reset()
        state = sim.step()  # one step so SSE has a valid state to send

    with _state_lock:
        _state.clear()
        _state.update(state)

    with _training_lock:
        _training_state.update({
            "active":   False,
            "phase":    None,
            "progress": 1.0,
            "step":     "done",
            "wind":     0,
            "temp":     20,
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes simulation state at ~20 Hz."""
    def generate():
        while True:
            with _state_lock:
                base = dict(_state) if _state else None
            if base:
                with _training_lock:
                    base["training"] = dict(_training_state)
                with _paused_lock:
                    base["paused"] = _paused
                yield f"data: {json.dumps(base)}\n\n"
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/toggle", methods=["POST"])
def toggle():
    name = (request.json or {}).get("sensor", "")
    with sim_lock:
        sim.toggle_sensor(name)
        failed = dict(sim.sensor_failed)
    return jsonify({"failed": failed})


@app.route("/environment", methods=["POST"])
def environment():
    data = request.json or {}
    with sim_lock:
        sim.set_environment(
            wind_speed   = float(data.get("wind_speed",   0)),
            wind_heading = float(data.get("wind_heading", 0)),
            temperature  = float(data.get("temperature",  20)),
        )
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset_sim():
    with sim_lock:
        sim.reset()
    with _state_lock:
        _state.clear()
    return jsonify({"ok": True})


@app.route("/pause", methods=["POST"])
def pause():
    global _paused
    with _paused_lock:
        _paused = not _paused
        state = _paused
    return jsonify({"paused": state})


@app.route("/train", methods=["POST"])
def train():
    with _training_lock:
        if _training_state.get("active"):
            return jsonify({"ok": False, "reason": "already training"})
    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Starting KalmanNET drone demo at http://localhost:5000")
    app.run(debug=False, threaded=True, port=5000)
