import json
import threading
import time

from flask import Flask, Response, jsonify, request

from drone_sim import DroneSim

app = Flask(__name__)

sim      = DroneSim()
sim_lock = threading.Lock()

_state      = {}
_state_lock = threading.Lock()


def _sim_loop() -> None:
    while True:
        with sim_lock:
            state = sim.step()
        with _state_lock:
            _state.clear()
            _state.update(state)
        time.sleep(0.05)  # 20 Hz


threading.Thread(target=_sim_loop, daemon=True).start()


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
                payload = json.dumps(dict(_state)) if _state else None
            if payload:
                yield f"data: {payload}\n\n"
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
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


if __name__ == "__main__":
    print("Starting KalmanNET drone demo at http://localhost:5000")
    app.run(debug=False, threaded=True, port=5000)
