'use strict';

// Interception layer: static/app.js talks to the Flask backend through
// EventSource('/stream') and fetch() on six routes. This file reroutes those
// to the local Engine so app.js runs byte-for-byte unmodified. Anything else
// (e.g. weights.json) passes through to the real network.
//
// Also overrides the handful of tooltip/tour strings whose server-version
// wording ("saves to kalmannet_model.pt", "persists on disk", "optimised for
// 200 epochs") would be dishonest in a build that loads pretrained weights.

(function () {

  // ── fetch ────────────────────────────────────────────────────────────────────
  const nativeFetch = window.fetch.bind(window);
  const ROUTES = new Set(['/toggle', '/environment', '/reset', '/pause', '/clear-model', '/train']);

  window.fetch = function (url, opts) {
    if (typeof url === 'string' && ROUTES.has(url)) {
      let body = {};
      try { body = opts && opts.body ? JSON.parse(opts.body) : {}; } catch (_) {}
      const data = Engine.handle(url, body);
      return Promise.resolve({ ok: true, status: 200, json: async () => data });
    }
    return nativeFetch(url, opts);
  };

  // ── EventSource ──────────────────────────────────────────────────────────────
  // The engine pushes frames as JSON strings at 20 Hz; onerror never fires
  // (there is no connection to lose).
  window.EventSource = class {
    constructor(_url) {
      this.onmessage = null;
      this.onerror   = null;
      this._unsub = Engine.subscribe(frame => {
        if (this.onmessage) this.onmessage({ data: frame });
      });
    }
    close() {
      if (this._unsub) { this._unsub(); this._unsub = null; }
    }
  };

  // ── Honest labeling: tooltips (before tooltips.js applies them on
  //    DOMContentLoaded — this script evaluates first, so mutation is safe) ────
  if (typeof TOOLTIPS !== 'undefined') {
    TOOLTIPS['btn-train'] =
      `Flies the same scripted calibration sequence as the server version at 10× speed (~30 s) — calm, wind up to 20 m/s, temperatures from −10°C to 50°C, combined stress — collecting ~5,800 steps of sensor data with known ground truth.<br><br>
      This browser build ships <b>inference only</b>: after the flight it loads GRU weights pretrained offline (PyTorch, in the repo) on this exact flight, rather than running the optimiser in the page. The buffer you just collected is real — <code>Engine.downloadBuffer()</code> in the console saves it for offline training with train_kalmannet_gru.py.<br><br>
      Trained state persists in your browser (localStorage). Controls are locked while the flight runs.`;

    TOOLTIPS['btn-clear'] =
      `Reverts to the standard Kalman filter (white drone) and clears the trained state from your browser (localStorage). Re-enabling takes ~30 s — the calibration flight re-runs, then the pretrained weights reload.`;

    TOOLTIPS['panel-kn-title'] =
      `Kalman-NET status panel. UNTRAINED: the cyan drone is hidden and Kalman-NET mathematically mirrors the standard filter. Click TRAIN to run a scripted calibration flight — afterwards the cyan drone flies the learned-gain estimate and the white filter drone becomes a translucent silhouette for comparison. Trained state persists in your browser (localStorage) across visits.`;

    TOOLTIPS['badge-local'] =
      `This page has no backend. The drone simulation, the classical Kalman filter, and the Kalman-NET GRU (~18k parameters, 75 KB) all run in your browser tab in plain JavaScript, at the same 20 Hz as the server version — each visitor gets an independent simulation.<br><br>
      The GRU weights were trained offline in PyTorch and the JS filter math is held to the PyTorch/numpy implementation by parity tests (web/test/parity.mjs).`;
  }

  // ── Honest labeling: tour step 5 (app.js runs after this script, so mutate
  //    once the DOM is ready — TOUR_STEPS exists by then) ───────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    if (typeof TOUR_STEPS !== 'undefined' && TOUR_STEPS[4]) {
      TOUR_STEPS[4].body =
        'Click <b>TRAIN KALMAN-NET</b>. It flies a ~30 s scripted calibration flight at 10× speed, then loads a GRU pretrained on exactly this flight. In this browser build the network runs fully client-side; the optimiser itself lives in the repo\'s PyTorch trainer.';
    }
  });

})();
