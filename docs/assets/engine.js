/* A PINN that trains in the browser. Port of pinn/scratch.py, no libraries.
   tests/engine.test.mjs compares its gradients with the NumPy ones. */
(function (root) {
  "use strict";

  /* ---- physics: same closed form as pinn/physics.py ---- */
  function oscillator(m, c, k, x0, v0) {
    x0 = x0 === undefined ? 1 : x0; v0 = v0 || 0;
    var d = c / (2 * m), w = Math.sqrt(k / m - d * d), b = (v0 + d * x0) / w;
    return {
      m: m, c: c, k: k, delta: d, omegaD: w, omega0: Math.sqrt(k / m),
      zeta: c / (2 * Math.sqrt(m * k)),
      x: function (t) { return Math.exp(-d * t) * (x0 * Math.cos(w * t) + b * Math.sin(w * t)); },
      v: function (t) {
        return Math.exp(-d * t) * ((b * w - d * x0) * Math.cos(w * t) - (x0 * w + d * b) * Math.sin(w * t));
      },
      a: function (t) { return -(c * this.v(t) + k * this.x(t)) / m; }
    };
  }

  /* ---- small seeded random generator (mulberry32) ---- */
  function rng(seed) {
    var s = seed >>> 0;
    function u() {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return {
      uniform: function (a, b) { return a + (b - a) * u(); },
      normal: function () {
        return Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
      }
    };
  }

  function linspace(a, b, n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(n === 1 ? a : a + (b - a) * i / (n - 1));
    return out;
  }

  function measure(osc, opts) {
    opts = opts || {};
    var r = rng(opts.seed || 0), noise = opts.noise === undefined ? 0.01 : opts.noise;
    var t = linspace(0, opts.tMax || 0.3, opts.n || 10);
    return { t: t, x: t.map(function (q) { return osc.x(q) + noise * r.normal(); }) };
  }

  /* ---- the network ---- */
  function Net(opts) {
    opts = opts || {};
    var width = opts.width || 24, depth = opts.depth || 3, r = rng(opts.seed || 1);
    this.tMax = opts.tMax || 1;
    this.sizes = [1];
    for (var d = 0; d < depth; d++) this.sizes.push(width);
    this.sizes.push(1);
    this.W = []; this.b = []; this.gW = []; this.gb = []; this.buf = [];
    for (var l = 0; l + 1 < this.sizes.length; l++) {
      var nIn = this.sizes[l], nOut = this.sizes[l + 1], bound = Math.sqrt(6 / (nIn + nOut));
      var W = new Float64Array(nIn * nOut);
      for (var i = 0; i < W.length; i++) W[i] = r.uniform(-bound, bound);
      this.W.push(W); this.b.push(new Float64Array(nOut));
      this.gW.push(new Float64Array(nIn * nOut)); this.gb.push(new Float64Array(nOut));
      this.buf.push({
        a: new Float64Array(nIn), a1: new Float64Array(nIn), a2: new Float64Array(nIn),
        z: new Float64Array(nOut), z1: new Float64Array(nOut), z2: new Float64Array(nOut),
        g: new Float64Array(nOut), g1: new Float64Array(nOut), g2: new Float64Array(nOut)
      });
    }
  }

  /* One time in; x, x' and x'' out. Intermediate values stay in this.buf. */
  Net.prototype.forward = function (t) {
    var L = this.W.length, B = this.buf, i, j, l;
    B[0].a[0] = 2 * t / this.tMax - 1; B[0].a1[0] = 2 / this.tMax; B[0].a2[0] = 0;
    for (l = 0; l < L; l++) {
      var W = this.W[l], bias = this.b[l], p = B[l], nIn = this.sizes[l], nOut = this.sizes[l + 1];
      var next = l + 1 < L ? B[l + 1] : null;
      for (j = 0; j < nOut; j++) {
        var z = bias[j], z1 = 0, z2 = 0, o = j * nIn;
        for (i = 0; i < nIn; i++) {
          var w = W[o + i];
          z += w * p.a[i]; z1 += w * p.a1[i]; z2 += w * p.a2[i];
        }
        p.z[j] = z; p.z1[j] = z1; p.z2[j] = z2;
        if (next) {
          var h = Math.tanh(z), h1 = 1 - h * h, h2 = -2 * h * h1;
          next.a[j] = h;
          next.a1[j] = h1 * z1;
          next.a2[j] = h2 * z1 * z1 + h1 * z2;
        }
      }
    }
    var q = B[L - 1];
    return [q.z[0], q.z1[0], q.z2[0]];
  };

  /* Add this point's gradients. g = dL/dx, g1 = dL/dx', g2 = dL/dx''. */
  Net.prototype.backward = function (g, g1, g2) {
    var L = this.W.length, B = this.buf, i, j, l;
    B[L - 1].g[0] = g; B[L - 1].g1[0] = g1; B[L - 1].g2[0] = g2;
    for (l = L - 1; l >= 0; l--) {
      var W = this.W[l], gW = this.gW[l], gb = this.gb[l], p = B[l];
      var nIn = this.sizes[l], nOut = this.sizes[l + 1], prev = l > 0 ? B[l - 1] : null;
      if (prev) { prev.g.fill(0); prev.g1.fill(0); prev.g2.fill(0); }
      for (j = 0; j < nOut; j++) {
        var gz = p.g[j], gz1 = p.g1[j], gz2 = p.g2[j], o = j * nIn;
        if (l < L - 1) {
          var z1 = p.z1[j], z2 = p.z2[j], h = Math.tanh(p.z[j]);
          var h1 = 1 - h * h, h2 = -2 * h * h1, h3 = -2 * h1 * (1 - 3 * h * h);
          var u = gz * h1 + gz1 * h2 * z1 + gz2 * (h3 * z1 * z1 + h2 * z2);
          var u1 = gz1 * h1 + gz2 * 2 * h2 * z1;
          gz2 = gz2 * h1; gz1 = u1; gz = u;
        }
        gb[j] += gz;
        for (i = 0; i < nIn; i++) {
          gW[o + i] += gz * p.a[i] + gz1 * p.a1[i] + gz2 * p.a2[i];
          if (prev) {
            var w = W[o + i];
            prev.g[i] += gz * w; prev.g1[i] += gz1 * w; prev.g2[i] += gz2 * w;
          }
        }
      }
    }
  };

  Net.prototype.zeroGrad = function () {
    this.gW.forEach(function (a) { a.fill(0); });
    this.gb.forEach(function (a) { a.fill(0); });
  };

  Net.prototype.predict = function (t) { return this.forward(t)[0]; };

  Net.prototype.nParameters = function () {
    var n = 0;
    this.W.concat(this.b).forEach(function (a) { n += a.length; });
    return n;
  };

  /* mean((x - dots)^2) + weight * mean(r^2), and its gradient. */
  Net.prototype.lossAndGrads = function (data, tCol, phys, weight) {
    this.zeroGrad();
    var dataLoss = 0, physLoss = 0, n = data.t.length, i, out;
    for (i = 0; i < n; i++) {
      out = this.forward(data.t[i]);
      var e = out[0] - data.x[i];
      dataLoss += e * e / n;
      this.backward(2 * e / n, 0, 0);
    }
    if (weight > 0) {
      var N = tCol.length;
      for (i = 0; i < N; i++) {
        out = this.forward(tCol[i]);
        var r = (phys.m * out[2] + phys.c * out[1] + phys.k * out[0]) / phys.k;
        physLoss += r * r / N;
        var g = weight * 2 * r / N;
        this.backward(g, g * phys.c / phys.k, g * phys.m / phys.k);
      }
    }
    return { data: dataLoss, physics: physLoss, total: dataLoss + weight * physLoss };
  };

  /* ---- Adam ---- */
  function Adam(net, lr) {
    this.lr = lr || 3e-3; this.t = 0;
    this.params = net.W.concat(net.b); this.grads = net.gW.concat(net.gb);
    this.mean = this.params.map(function (p) { return new Float64Array(p.length); });
    this.vari = this.params.map(function (p) { return new Float64Array(p.length); });
  }
  Adam.prototype.step = function () {
    this.t++;
    var c1 = 1 - Math.pow(0.9, this.t), c2 = 1 - Math.pow(0.999, this.t);
    for (var k = 0; k < this.params.length; k++) {
      var p = this.params[k], g = this.grads[k], m = this.mean[k], v = this.vari[k];
      for (var i = 0; i < p.length; i++) {
        m[i] += 0.1 * (g[i] - m[i]);
        v[i] += 0.001 * (g[i] * g[i] - v[i]);
        p[i] -= this.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
      }
    }
  };

  /* A training run that advances a few steps at a time, so a page can draw. */
  function Trainer(opts) {
    this.osc = opts.osc; this.data = opts.data;
    this.weight = opts.weight === undefined ? 10 : opts.weight;
    this.phys = opts.phys || opts.osc;
    this.tCol = opts.tCol || linspace(0, 1, opts.nCol || 60);
    this.steps = opts.steps || 4000; this.lr = opts.lr || 3e-3;
    this.net = new Net({ width: opts.width, depth: opts.depth, seed: opts.seed });
    this.adam = new Adam(this.net, this.lr);
    this.step = 0; this.history = []; this.parts = null;
  }
  Trainer.prototype.advance = function (n) {
    while (n-- > 0 && this.step < this.steps) {
      var lo = this.lr / 30;
      this.adam.lr = lo + (this.lr - lo) * 0.5 * (1 + Math.cos(Math.PI * this.step / this.steps));
      this.parts = this.net.lossAndGrads(this.data, this.tCol, this.phys, this.weight);
      if (this.step % 20 === 0) this.history.push({ step: this.step, data: this.parts.data, physics: this.parts.physics });
      this.adam.step();
      this.step++;
    }
    return this.step < this.steps;
  };
  Trainer.prototype.futureMse = function (from) {
    var s = 0, n = 200, a = from === undefined ? 0.3 : from;
    for (var i = 1; i <= n; i++) {
      var t = a + (1 - a) * i / n, e = this.net.predict(t) - this.osc.x(t);
      s += e * e / n;
    }
    return s;
  };

  var api = { oscillator: oscillator, rng: rng, linspace: linspace, measure: measure, Net: Net, Adam: Adam, Trainer: Trainer };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof self !== "undefined" ? self : this);
