'use strict';

// Local backend for the browser build — replicates app.py's loop and route
// semantics so the unmodified static/app.js frontend runs against it (see
// web/js/shim.js for the interception layer and BROWSER_PORT_PLAN.md for why).
//
// Loop contract (mirrors _sim_loop + the SSE generator): one 50 ms tick; the
// sim steps only while unpaused (10 steps/tick during the calibration flight —
// Python's 200 Hz fast mode); every tick emits the state dict merged with
// {training, paused} as a JSON string, exactly like the SSE stream.

const Engine = (() => {

  const sim = new DroneSim(0.05);
  let paused   = true;
  let fast     = false;
  let state    = null;
  let training = { active: false, phase: null, progress: 0.0, step: 'idle' };

  const listeners = new Set();
  const TRAINED_KEY = 'kalmannetTrained';

  // (phase_label, wind_m_s, temp_C, duration_s) — must match app.py's
  // _TRAIN_PHASES and the KN_PHASES list in static/app.js.
  const TRAIN_PHASES = [
    ['CALM BASELINE',    0,  20, 4],
    ['WIND  10 m/s',    10,  20, 3],
    ['WIND  20 m/s',    20,  20, 3],
    ['WIND   5 m/s',     5,  20, 2],
    ['CALM',             0,  20, 2],
    ['TEMP  40°C',       0,  40, 3],
    ['TEMP  50°C',       0,  50, 3],
    ['TEMP -10°C',       0, -10, 3],
    ['COMBINED STRESS', 12,  40, 4],
    ['RETURN TO CALM',   0,  20, 2],
  ];

  const sleep = ms => new Promise(res => setTimeout(res, ms));

  function hasTrainedFlag() {
    try { return localStorage.getItem(TRAINED_KEY) === '1'; } catch (_) { return false; }
  }
  function setTrainedFlag(on) {
    try {
      if (on) localStorage.setItem(TRAINED_KEY, '1');
      else    localStorage.removeItem(TRAINED_KEY);
    } catch (_) {}
  }

  async function loadTrained() {
    const res = await fetch('weights.json');
    if (!res.ok) throw new Error('weights.json fetch failed: ' + res.status);
    sim.kn.loadWeights(await res.json());
  }

  function tick() {
    if (!paused) {
      const n = fast ? 10 : 1;
      for (let i = 0; i < n; i++) state = sim.step();
    }
    if (!state) return;
    const frame = JSON.stringify(
      Object.assign({}, state, { training: Object.assign({}, training), paused })
    );
    listeners.forEach(fn => fn(frame));
  }

  // ── Training: real calibration flight, then pretrained weights ─────────────
  // The buffer is genuinely collected (Engine.downloadBuffer() saves it for
  // offline training); the optimiser itself is the repo's PyTorch trainer.
  async function runTraining() {
    paused = false;
    fast   = true;
    try {
      sim.reset();
      sim.trainingBuffer.length = 0;
      sim.collectingData = true;

      const n = TRAIN_PHASES.length;
      for (let i = 0; i < n; i++) {
        const [phase, wind, temp, dur] = TRAIN_PHASES[i];
        training = {
          active: true, phase, phase_idx: i,
          progress: Math.round(i / n * 1000) / 1000,
          step: 'collecting', wind, temp,
        };
        sim.setEnvironment(wind, 0.0, temp);
        await sleep(dur * 1000);
      }

      fast = false;
      sim.collectingData = false;
      console.info(`[engine] calibration flight done — ${sim.trainingBuffer.length} steps collected. ` +
                   'Engine.downloadBuffer() saves it for offline training (train_kalmannet_gru.py).');

      training = {
        active: true, phase: 'LOADING PRETRAINED WEIGHTS', phase_idx: n,
        progress: 0.9, step: 'loading', wind: 0, temp: 20,
      };
      await loadTrained();
      setTrainedFlag(true);
      await sleep(700);   // let the phase render before the reveal

      paused = true;
      sim.setEnvironment(0.0, 0.0, 20.0);
      sim.reset();
      state = sim.step();   // fresh frame so the UI updates while paused

      training = { active: false, phase: null, progress: 1.0, step: 'done', wind: 0, temp: 20 };
    } catch (err) {
      console.error('[engine] training flow failed:', err);
    } finally {
      // Safety net, mirrors _run_training's finally: always unlock the UI.
      fast = false;
      sim.collectingData = false;
      if (training.active) training = { active: false, phase: null, step: 'done', progress: training.progress };
    }
  }

  // ── Route handlers (same JSON shapes as the Flask routes) ───────────────────
  function handle(url, body) {
    switch (url) {
      case '/toggle':
        sim.toggleSensor(body.sensor || '');
        return { failed: Object.assign({}, sim.sensorFailed) };

      case '/environment':
        sim.setEnvironment(
          +(body.wind_speed   ?? 0),
          +(body.wind_heading ?? 0),
          +(body.temperature  ?? 20),
        );
        return { ok: true };

      case '/reset':
        sim.reset();
        state = sim.step();   // fresh frame even while paused
        return { ok: true };

      case '/pause':
        paused = !paused;
        return { paused };

      case '/clear-model':
        sim.kn = new KalmanNET(sim.dt);
        sim.kn.reset(sim.kf.x.slice());
        setTrainedFlag(false);
        state = sim.step();   // carries kn_trained=false
        return { ok: true };

      case '/train':
        if (training.active) return { ok: false, reason: 'already training' };
        training = { active: true, phase: 'STARTING', phase_idx: 0, progress: 0.0, step: 'starting' };
        runTraining();
        return { ok: true };

      default:
        return { ok: false, reason: 'unknown route' };
    }
  }

  function subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  function downloadBuffer() {
    const blob = new Blob([JSON.stringify(sim.trainingBuffer)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'training_buffer.json';
    a.click();
    URL.revokeObjectURL(a.href);
    return sim.trainingBuffer.length + ' steps';
  }

  async function init() {
    if (hasTrainedFlag()) {
      try { await loadTrained(); }
      catch (err) { console.warn('[engine] stored trained flag but weight load failed:', err); }
    }
    state = sim.step();      // prime, mirrors app.py's module-level priming step
    setInterval(tick, 50);
  }
  init();

  return { handle, subscribe, downloadBuffer, get sim() { return sim; } };
})();
