/* Figures that compute everything live, using engine.js. */
(function () {
  "use strict";
  var F = Fig.figures, el = Fig.el, sv = Fig.sv, E = window.Engine;
  var TS = E.linspace(0, 1, 201), OSC = E.oscillator(1, 3, 225);
  function map(f) { return TS.map(f); }
  function mse(f, a, b) {
    var s = 0, n = 300;
    for (var i = 1; i <= n; i++) { var t = a + (b - a) * i / n, e = f(t) - OSC.x(t); s += e * e / n; }
    return s;
  }
  function dotLoss(f, data) { return data.t.reduce(function (s, t, i) { return s + Math.pow(f(t) - data.x[i], 2); }, 0) / data.t.length; }

  /* ---- FIG_000: both nets, training in the page ---- */
  F.live = function (body) {
    var seed = 1, physics = true, running = false, visible = false, a, b, data;
    var plot = Fig.timePlot({ label: "Two networks training on ten dots", h: 360 });
    plot.band(0, 0.3, "TEN DOTS LIVE HERE");
    var truth = plot.path("s-truth"), red = plot.path("s-bad"), blue = plot.path("s-curve"), dots = plot.dots([], [], 5);
    truth.set(TS, map(OSC.x));
    var stats = Fig.stats(["step", "plain, future error", "PINN, future error", "PINN, equation error"]);
    var go = Fig.btn("Train", function () { if (a.step >= a.steps) reset(); running = !running; label(); if (running) tick(); });
    var tog = Fig.btn("Physics on", function () { physics = !physics; tog.setAttribute("aria-pressed", physics); tog.textContent = physics ? "Physics on" : "Physics off"; reset(); running = true; label(); tick(); }, true);
    var re = Fig.btn("New dots", function () { seed++; reset(); running = true; label(); tick(); });
    var read = Fig.readout("what is happening");

    function reset() {
      running = false;
      data = E.measure(OSC, { seed: seed });
      var o = { osc: OSC, data: data, steps: 5000, seed: seed, width: 20, nCol: 40 };
      a = new E.Trainer(Object.assign({ weight: 0 }, o));
      b = new E.Trainer(Object.assign({ weight: physics ? 10 : 0 }, o));
      dots.set(data.t, data.x); label(); draw();
    }
    function label() { go.textContent = running ? "Pause" : a.step >= a.steps ? "Train again" : a.step ? "Continue" : "Train"; }
    function draw() {
      red.set(TS, map(function (t) { return a.net.predict(t); }));
      blue.set(TS, map(function (t) { return b.net.predict(t); }));
      stats.set([a.step + " / " + a.steps, Fig.sci(a.futureMse()), Fig.sci(b.futureMse()), physics && b.parts ? Fig.sci(b.parts.physics) : "not used"]);
      read.say(a.step === 0 ? "Both networks start from the same random weights, so the two curves are identical. Press Train."
        : !physics ? "Without the equation the blue network is the red network. They get the same loss and the same weights, so they make the same wrong prediction."
        : a.step < 1200 ? "Both curves are learning the dots. The blue one is also checked against the equation at 40 times that have no dots."
        : a.step < a.steps ? "The red curve has finished with its dots and nothing pulls on the rest of it. The blue curve is being bent into a spring."
        : "Same network, same ten dots. Blue carries on as a spring. Red carries on as whatever tanh does when no one is looking.");
    }
    function tick() {
      if (!running || !visible) return;
      var t0 = performance.now(), more = true;
      while (more && performance.now() - t0 < 22) { a.advance(4); more = b.advance(4); }
      draw();
      if (more) requestAnimationFrame(tick); else { running = false; label(); }
    }
    body.appendChild(el("div", { class: "controls" }, [go, tog, re]));
    body.appendChild(Fig.legend([["truth", "true motion"], ["red", "plain net"], ["blue", "PINN"], ["dot", "10 noisy measurements"]]));
    body.appendChild(plot.wrap); body.appendChild(stats); body.appendChild(read);
    plot.hover(function (t) { return [["true", OSC.x(t)], ["plain", a.net.predict(t)], ["PINN", b.net.predict(t)]]; });
    reset();
    var first = true;
    Fig.whenVisible(body, function (v) {
      visible = v;
      if (v && first && !Fig.reduced) { first = false; running = true; label(); }
      if (v && running) tick();
    });
  };

  /* ---- Chapter 1: many curves fit the dots ---- */
  F.manycurves = function (body) {
    var data = E.measure(OSC, { seed: 1 });
    function poly(deg) {
      // Least squares in Chebyshev polynomials on [0, 0.3], normal equations by Gaussian elimination.
      var n = deg + 1, A = [], i, j, k;
      function basis(t) { var u = t / 0.15 - 1, v = [1, u]; for (var q = 2; q < n; q++) v.push(2 * u * v[q - 1] - v[q - 2]); return v.slice(0, n); }
      for (i = 0; i < n; i++) A.push(new Array(n + 1).fill(0));
      data.t.forEach(function (t, d) { var v = basis(t); for (i = 0; i < n; i++) { for (j = 0; j < n; j++) A[i][j] += v[i] * v[j]; A[i][n] += v[i] * data.x[d]; } });
      for (i = 0; i < n; i++) {
        var p = i; for (j = i + 1; j < n; j++) if (Math.abs(A[j][i]) > Math.abs(A[p][i])) p = j;
        var tmp = A[i]; A[i] = A[p]; A[p] = tmp;
        for (j = i + 1; j < n; j++) { var f = A[j][i] / A[i][i]; for (k = i; k <= n; k++) A[j][k] -= f * A[i][k]; }
      }
      var c = new Array(n);
      for (i = n - 1; i >= 0; i--) { var s = A[i][n]; for (j = i + 1; j < n; j++) s -= A[i][j] * c[j]; c[i] = s / A[i][i]; }
      return function (t) { return basis(t).reduce(function (q, v, m) { return q + v * c[m]; }, 0); };
    }
    var p5 = poly(5), p9 = poly(9), last = OSC.x(0.3);
    function fade(t) { var u = Math.min(1, (t - 0.3) / 0.12); return u <= 0 ? OSC.x(t) : OSC.x(t) * (1 - u) * (1 - u) * (1 + 2 * u); }
    var options = [
      ["True spring", OSC.x, "The curve the dots came from. It misses most dots slightly, because the dots carry noise."],
      ["Polynomial, degree 5", p5, "A smooth, respectable fit to the dots. It leaves the chart almost at once, because a polynomial has nothing else to do at large t."],
      ["Polynomial, degree 9", p9, "Ten coefficients, ten dots: it hits every dot exactly and is the worst prediction here. A training error of zero has bought nothing."],
      ["Stops moving", fade, "It follows the dots and then settles at zero. It beats both polynomials, and it is still wrong: a real spring has no reason to stop."],
      ["Flat line", function (t) { return t < 0.3 ? OSC.x(t) : last; }, "It follows the dots and then holds the last value. A tanh network also ends up flat, once it is past its last bend."]
    ];
    var plot = Fig.timePlot({ label: "Different curves through the same dots", y: [-2, 2], yt: [-2, -1, 0, 1, 2] });
    plot.band(0, 0.3, "MEASURED"); plot.back.appendChild(sv("text", { class: "s-text-sm", x: plot.X(.3) + 10, y: plot.T + 14, text: "NOT MEASURED" }));
    var line = plot.path("s-curve"), cur; plot.dots(data.t, data.x, 5);
    var stats = Fig.stats(["error on the dots", "error after the dots"]), read = Fig.readout("this curve");
    function show(i) {
      cur = options[i][1];
      line.set(TS, map(cur)); line.setAttribute("class", i ? "s-bad" : "s-curve");
      stats.set([Fig.sci(dotLoss(cur, data)), Fig.sci(mse(cur, 0.3, 1))]); read.say(options[i][2]);
    }
    body.appendChild(Fig.choice(options.map(function (o) { return o[0]; }), 1, show));
    body.appendChild(plot.wrap); body.appendChild(stats); body.appendChild(read);
    plot.hover(function (t) { return [["curve", Math.abs(cur(t)) > 99 ? "off chart" : cur(t)], ["true", OSC.x(t)]]; });
    show(1);
  };

  /* ---- Chapter 2: turn the knobs of the spring ---- */
  F.knobs = function (body) {
    var plot = Fig.timePlot({ label: "Damped oscillation for chosen m, c, k" });
    var up = plot.path("s-faint"), dn = plot.path("s-faint"), line = plot.path("s-curve"), osc;
    var stats = Fig.stats(["natural freq ω₀", "decay δ", "damping ratio ζ", "period"]), read = Fig.readout();
    var m, c, k;
    function f0(v) { return v.toFixed(0); }
    m = Fig.slider("mass m (kg)", 0.5, 3, 0.1, 1, function (v) { return v.toFixed(1); }, draw);
    c = Fig.slider("friction c (kg/s)", 0, 16, 0.5, 3, function (v) { return v.toFixed(1); }, draw);
    k = Fig.slider("stiffness k (N/m)", 50, 900, 5, 225, f0, draw);
    function draw() {
      osc = E.oscillator(m.value(), c.value(), k.value());
      line.set(TS, map(osc.x));
      var env = function (s) { return map(function (t) { return s * Math.exp(-osc.delta * t) / Math.sqrt(1 - osc.zeta * osc.zeta); }); };
      up.set(TS, env(1)); dn.set(TS, env(-1));
      var period = 2 * Math.PI / osc.omegaD;
      stats.set([osc.omega0.toFixed(1) + " /s", osc.delta.toFixed(2) + " /s", osc.zeta.toFixed(3), period.toFixed(3) + " s"]);
      read.say("&omega;<sub>0</sub> = &radic;(k/m) = &radic;(" + k.value() + "/" + m.value().toFixed(1) + ") and &delta; = c/2m = " + c.value().toFixed(1) + "/" + (2 * m.value()).toFixed(1) +
        ". The ten dots cover 0.3 s, which is <b>" + (0.3 / period).toFixed(2) + " of a period</b>. " +
        (osc.c === 0 ? "With no friction the swing never shrinks." : "Each full swing keeps " + (100 * Math.exp(-osc.delta * period)).toFixed(0) + "% of the one before it."));
    }
    plot.hover(function (t) { return [["x", osc.x(t)]]; });
    [plot.wrap, m, c, k, stats, read].forEach(function (n) { body.appendChild(n); });
    draw();
  };

  /* ---- Chapter 2: three forces that always cancel ---- */
  F.forces = function (body) {
    var plot = Fig.timePlot({ label: "Position with a movable time marker", h: 220, yt: [-1, 0, 1] });
    plot.path("s-curve").set(TS, map(OSC.x));
    var mark = sv("line", { class: "s-dash", y1: plot.T, y2: plot.B }), dot = sv("circle", { class: "s-node", r: 6 });
    plot.over.appendChild(mark); plot.over.appendChild(dot);
    var svg = sv("svg", { viewBox: "0 0 960 190", role: "img", "aria-label": "The three terms of the equation and their sum" }), g = sv("g");
    svg.appendChild(g);
    var read = Fig.readout(), mid = 560, scale = 300 / 240;
    var rows = [["m x″", "mass × acceleration"], ["c x′", "friction"], ["k x", "spring"], ["sum", "always zero"]];
    function draw(t) {
      var x = OSC.x(t), v = OSC.v(t), a = OSC.a(t), vals = [a, 3 * v, 225 * x, a + 3 * v + 225 * x];
      mark.setAttribute("x1", plot.X(t)); mark.setAttribute("x2", plot.X(t)); dot.setAttribute("cx", plot.X(t)); dot.setAttribute("cy", plot.Y(x));
      Fig.clear(g);
      g.appendChild(sv("line", { class: "s-axis", x1: mid, x2: mid, y1: 0, y2: 180 }));
      vals.forEach(function (val, i) {
        var y = 10 + i * 44, w = Math.abs(val) * scale;
        g.appendChild(sv("text", { class: "s-math", x: 20, y: y + 20, text: rows[i][0] }));
        g.appendChild(sv("text", { class: "s-text-sm", x: 92, y: y + 19, text: rows[i][1].toUpperCase() }));
        g.appendChild(sv("rect", { class: i === 3 ? "s-fill" : "s-box-soft", x: val < 0 ? mid - w : mid, y: y, width: Math.max(w, 1), height: 28, rx: 3 }));
        g.appendChild(sv("text", { class: "s-text", x: 948, y: y + 19, "text-anchor": "end", text: Fig.fixed(val, 1) + " N" }));
      });
      read.say(Math.abs(v) < 1 ? "At a turning point the mass is nearly still, so friction does almost nothing. The spring and the acceleration cancel each other."
        : Math.abs(x) < 0.06 ? "Passing through the middle, the spring is slack. What is left is friction slowing the mass down."
        : "The spring pulls back, friction resists the motion, and the acceleration is exactly what makes the total zero.");
    }
    var s = Fig.slider("time t (s)", 0, 1, 0.005, 0.1, function (v) { return v.toFixed(3); }, draw);
    [plot.wrap, s, svg, read].forEach(function (n) { body.appendChild(n); });
    draw(0.1);
  };

  /* ---- Chapter 3: one tanh neuron ---- */
  F.neuron = function (body) {
    var plot = Fig.plot({ x: [-1, 1], y: [-1.6, 1.6], xt: [-1, -.5, 0, .5, 1], yt: [-1, 0, 1], xlabel: "input s", label: "One, two or three tanh neurons", xfmt: function (v) { return String(v).replace("-", "−"); } });
    var parts = [plot.path("s-faint"), plot.path("s-faint"), plot.path("s-faint")], sum = plot.path("s-curve");
    var S = E.linspace(-1, 1, 201), n = 1, read = Fig.readout();
    var w = Fig.slider("weight w", 0.5, 12, 0.5, 4, function (v) { return v.toFixed(1); }, draw);
    var b = Fig.slider("bias b", -6, 6, 0.1, 0, function (v) { return v.toFixed(1); }, draw);
    function unit(i, s) { return [1, -0.9, 0.7][i] * Math.tanh(w.value() * s + b.value() + [0, 3, -3][i] * w.value() / 5); }
    function draw() {
      parts.forEach(function (p, i) { if (i < n && n > 1) p.set(S, S.map(function (s) { return unit(i, s); })); else p.setAttribute("d", ""); });
      sum.set(S, S.map(function (s) { var t = 0; for (var i = 0; i < n; i++) t += unit(i, s); return t; }));
      read.say(n === 1 ? "One neuron is one bend. <b>w</b> sets how sharp it is and <b>b</b> slides it sideways. Away from the bend it is flat, and a flat curve has slope zero."
        : n + " neurons make " + (n === 2 ? "a bump" : "a wiggle") + " where their bends overlap. Beyond the last bend the sum is flat again. This is why a tanh network cannot carry a rhythm outside the region that trained its bends.");
    }
    body.appendChild(Fig.choice(["1 neuron", "2 neurons", "3 neurons"], 0, function (i) { n = i + 1; draw(); }));
    [plot.wrap, w, b, read].forEach(function (x) { body.appendChild(x); });
    draw();
  };

  /* ---- Chapter 4: carry three numbers through a tiny network ---- */
  F.carry = function (body) {
    var w1 = 1.5, b1 = 0.2, v = -0.8, c = 0.1, t = 0.4, R = {};
    R.a = 2 * t - 1; R.z = w1 * R.a + b1; R.z1 = w1 * 2; R.z2 = 0;
    var h = Math.tanh(R.z), h1 = 1 - h * h, h2 = -2 * h * h1;
    R.h = h; R.hp = h1 * R.z1; R.hpp = h2 * R.z1 * R.z1 + h1 * R.z2;
    R.x = v * R.h + c; R.x1 = v * R.hp; R.x2 = v * R.hpp;
    function f(q) { return Fig.fixed(q, 4); }
    var boxes = [["input", "s"], ["linear", "z = 1.5 s + 0.2"], ["tanh", "h = tanh z"], ["linear", "x = −0.8 h + 0.1"]];
    var vals = [[R.a, 2, 0], [R.z, R.z1, R.z2], [R.h, R.hp, R.hpp], [R.x, R.x1, R.x2]];
    var svg = sv("svg", { viewBox: "0 0 960 250", role: "img", "aria-label": "Value, first and second derivative moving through four stages" });
    var code = el("pre", { class: "codeline" }), read = Fig.readout();
    var steps = [
      [0, "a = 2*t/t_max - 1;  a1 = 2/t_max;  a2 = 0", "Start with t = 0.4. The network rescales time to s = 2t − 1, so the slope of s is 2 and its second derivative is 0. Three numbers enter the first layer."],
      [1, "z, z1, z2 = a @ W.T + b, a1 @ W.T, a2 @ W.T", "A linear layer multiplies all three by the same weight. The bias is added to the value only, because the derivative of a constant is zero."],
      [2, "a1 = h1 * z1", "tanh uses the chain rule: the slope of h is tanh′(z) times the slope of z. Here tanh′ = 1 − h² = " + h1.toFixed(4) + "."],
      [2, "a2 = h2 * z1**2 + h1 * z2", "The second derivative has two parts. The bend of tanh itself, h2 · z1², and the bend that z already had, h1 · z2. The second part is zero in the first layer and matters from the next layer on."],
      [3, "a, a1, a2 = z, z1, z2", "The last linear layer gives x, x′ and x″. No step size and no approximation were involved. Autograd reaches the same three numbers by walking the graph backwards twice."]
    ];
    function show(i) {
      Fig.clear(svg);
      boxes.forEach(function (bx, j) {
        var x = 12 + j * 240, on = j === steps[i][0], done = j <= steps[i][0];
        if (j) { svg.appendChild(sv("path", { class: done ? "s-line" : "s-faint", d: "M" + (x - 34) + " 105 h26 m-7 -5 l7 5 l-7 5" })); }
        svg.appendChild(sv("text", { class: "s-text-sm", x: x, y: 18, text: "0" + (j + 1) + "  " + bx[0].toUpperCase() }));
        svg.appendChild(sv("rect", { class: on ? "s-box-soft" : done ? "s-box" : "s-box-mute", x: x, y: 28, width: 200, height: 158, rx: 4 }));
        svg.appendChild(sv("text", { class: "s-text", x: x + 14, y: 54, text: bx[1] }));
        ["value", "slope", "bend"].forEach(function (nm, q) {
          svg.appendChild(sv("text", { class: "s-text-sm", x: x + 14, y: 92 + q * 34, text: nm.toUpperCase() }));
          var ready = j < steps[i][0] || (on && (j !== 2 || i === 3 || q < 2));
          svg.appendChild(sv("text", { class: on && ready ? "s-text-blue" : "s-text", x: x + 186, y: 92 + q * 34, "text-anchor": "end", text: ready ? f(vals[j][q]) : "·" }));
        });
      });
      svg.appendChild(sv("text", { class: "s-text-sm", x: 12, y: 222, text: "VALUE: THE OUTPUT SO FAR · SLOPE: ITS d/dt · BEND: ITS d²/dt²" }));
      code.textContent = steps[i][1]; read.say(steps[i][2]);
    }
    var st = Fig.stepper(steps.length, show);
    [st, el("div", { class: "plotwrap" }, [svg]), code, read].forEach(function (n) { body.appendChild(n); });
    st.go(0);
  };

  /* ---- Chapter 4: the activation decides what x'' can be ---- */
  F.activations = function (body) {
    var acts = {
      tanh: [Math.tanh, function (z) { var h = Math.tanh(z); return 1 - h * h; }, function (z) { var h = Math.tanh(z); return -2 * h * (1 - h * h); }],
      sin: [Math.sin, Math.cos, function (z) { return -Math.sin(z); }],
      softplus: [function (z) { return Math.log1p(Math.exp(z)); }, function (z) { return 1 / (1 + Math.exp(-z)); }, function (z) { var p = 1 / (1 + Math.exp(-z)); return p * (1 - p); }],
      relu: [function (z) { return Math.max(0, z); }, function (z) { return z > 0 ? 1 : 0; }, function () { return 0; }]
    };
    var say = {
      tanh: "Smooth, bounded, and it bends both ways. Its second derivative changes sign, which is what an oscillation needs.",
      sin: "Smooth and periodic. It is its own second derivative with a minus sign, which makes it a natural fit for springs.",
      softplus: "A smooth ReLU. The second derivative exists but never changes sign, so layers have to combine many of them to bend both ways.",
      relu: "Straight lines with a kink. The second derivative is <span class='bad'>zero everywhere autograd looks</span>, so m x″ disappears from the residual and the network is trained on c x′ + k x = 0 instead. That is a different physical system."
    };
    var P = [0, 1, 2].map(function (i) { return Fig.plot({ w: 320, h: 210, left: 40, x: [-3, 3], y: [[-1.3, 2.2], [-1.3, 1.5], [-1.2, 1.2]][i], xt: [-3, 0, 3], yt: [-1, 0, 1], xfmt: function (v) { return String(v).replace("-", "−"); }, xlabel: ["value", "first derivative", "second derivative"][i], label: "activation " + i }); });
    var lines = P.map(function (p) { return p.path("s-curve"); }), Z = E.linspace(-3, 3, 241), names = Object.keys(acts), read = Fig.readout();
    var grid = el("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px" }, P.map(function (p) { return p.svg; }));
    function show(i) {
      lines.forEach(function (l, j) { l.set(Z, Z.map(acts[names[i]][j])); l.setAttribute("class", names[i] === "relu" && j === 2 ? "s-bad" : "s-curve"); });
      read.say(say[names[i]]);
    }
    body.appendChild(Fig.choice(names, 0, show)); body.appendChild(grid); body.appendChild(read); show(0);
  };

  /* ---- Chapter 5: fit by hand, watch both losses ---- */
  F.residual = function (body) {
    var data = E.measure(OSC, { seed: 1 }), plot = Fig.timePlot({ label: "A curve with four knobs" });
    plot.band(0, 0.3, "DOTS"); var line = plot.path("s-curve"); plot.dots(data.t, data.x, 5);
    var rp = Fig.plot({ h: 170, x: [0, 1], y: [-2, 2], xt: [0, .2, .4, .6, .8, 1], yt: [-2, 0, 2], label: "Residual of the equation over time", ylabel: "residual", xlabel: "time (s)" });
    var rline = rp.path("s-bad");
    var A, w, d, ph, read = Fig.readout(), stats = Fig.stats(["data loss", "physics loss", "data + 10 × physics"]);
    function f2(v) { return v.toFixed(2); }
    A = Fig.slider("height", -1.4, 1.4, 0.01, 1, f2, draw); w = Fig.slider("frequency", 0, 30, 0.05, 9, f2, draw);
    d = Fig.slider("decay", -1, 6, 0.05, 0.5, f2, draw); ph = Fig.slider("phase", -1.6, 1.6, 0.01, 0, f2, draw);
    function x(t) { return A.value() * Math.exp(-d.value() * t) * Math.cos(w.value() * t - ph.value()); }
    function r(t) {
      var e = A.value() * Math.exp(-d.value() * t), W = w.value(), D = d.value(), th = W * t - ph.value(), C = Math.cos(th), S = Math.sin(th);
      var x0 = e * C, x1 = e * (-D * C - W * S), x2 = e * ((D * D - W * W) * C + 2 * D * W * S);
      return (x2 + 3 * x1 + 225 * x0) / 225;
    }
    function draw() {
      line.set(TS, map(x)); rline.set(TS, map(r));
      var dl = dotLoss(x, data), tc = E.linspace(0, 1, 60), pl = tc.reduce(function (s, t) { var q = r(t); return s + q * q / 60; }, 0);
      stats.set([Fig.sci(dl), Fig.sci(pl), Fig.sci(dl + 10 * pl)]);
      read.say(dl < 3e-4 && pl < 1e-3 ? "<span class='good'>Both losses are small.</span> Only one family of curves can do this, and it is the spring."
        : pl < 1e-3 && dl > 3e-4 ? "The equation is satisfied and the dots are not. <b>Any</b> height and phase would satisfy it, including zero. The equation says what kind of motion it is, the dots say which one."
        : dl < 3e-3 ? "The dots are nearly satisfied and the equation is not. The residual below shows where the curve breaks the law."
        : "Neither loss is small yet. Try the presets, then change one knob at a time.");
    }
    var sol = Math.atan(1.5 / OSC.omegaD);
    function preset(a, b, c, e) { A.set(a); w.set(b); d.set(c); ph.set(e); draw(); }
    body.appendChild(el("div", { class: "controls" }, [
      ["the spring", function () { preset(1.005, OSC.omegaD, 1.5, sol); }], ["right law, wrong height", function () { preset(0.4, OSC.omegaD, 1.5, sol); }],
      ["zero", function () { preset(0, OSC.omegaD, 1.5, 0); }], ["no friction", function () { preset(1, 15, 0, 0); }], ["start again", function () { preset(1, 9, .5, 0); }]
    ].map(function (p) { return el("button", { type: "button", class: "chip", text: p[0], onclick: p[1] }); })));
    [plot.wrap, rp.wrap, A, w, d, ph, stats, read].forEach(function (n) { body.appendChild(n); });
    draw();
  };

  /* ---- By-hand chapter: check a gradient against finite differences ---- */
  F.gradcheck = function (body) {
    var data = E.measure(OSC, { seed: 1 }), net = new E.Net({ width: 8, depth: 2, seed: 5 }), tc = E.linspace(0, 1, 20);
    var total = 0, flat = [];
    net.W.concat(net.b).forEach(function (p, k) { for (var i = 0; i < p.length; i++) flat.push([p, net.gW.concat(net.gb)[k], i]); });
    total = flat.length;
    var pick = 17, plot = Fig.plot({ h: 300, x: [-12, -1], y: [1e-13, 1e1], logy: true, xt: [-12, -10, -8, -6, -4, -2], yt: [1e-12, 1e-8, 1e-4, 1], yfmt: function (v) { return "1e" + Math.round(Math.log10(v)); }, xfmt: function (v) { return "1e" + v; }, xlabel: "finite-difference step h", ylabel: "relative error", label: "Finite difference error against step size" });
    var line = plot.path("s-curve"), dot = sv("circle", { class: "s-node", r: 6 }); plot.over.appendChild(dot);
    var read = Fig.readout(), stats = Fig.stats(["by hand", "finite difference", "relative error"]);
    function loss() { return net.lossAndGrads(data, tc, OSC, 10).total; }
    function fd(lh) {
      var e = flat[pick], o = e[0][e[2]], h = Math.pow(10, lh);
      e[0][e[2]] = o + h; var up = loss(); e[0][e[2]] = o - h; var dn = loss(); e[0][e[2]] = o;
      return (up - dn) / (2 * h);
    }
    function exact() { loss(); return flat[pick][1][flat[pick][2]]; }
    function err(lh) { var g = exact(); return Math.max(Math.abs(fd(lh) - g) / (Math.abs(g) + 1e-12), 1e-13); }
    function draw() {
      var H = E.linspace(-12, -1, 56); line.set(H, H.map(err));
      var lh = s.value(), g = exact(), e = err(lh);
      dot.setAttribute("cx", plot.X(lh)); dot.setAttribute("cy", plot.Y(e));
      stats.set([g.toExponential(6), fd(lh).toExponential(6), Fig.sci(e)]);
      read.say(lh > -3.5 ? "The step is too big. The loss curves between the two samples, and the error shrinks as h² while you move left."
        : lh < -7 ? "The step is too small. The two losses agree in almost every digit, so subtracting them leaves rounding noise."
        : "The sweet spot. The two methods agree to about " + Math.max(1, Math.round(-Math.log10(e))) + " digits, which is as much as a finite difference can give.");
    }
    var s = Fig.slider("step h", -12, -1, 0.2, -5, function (v) { return "1e" + v.toFixed(1); }, draw);
    var p = Fig.slider("which weight", 0, total - 1, 1, pick, function (v) { return (v + 1) + " / " + total; }, function (v) { pick = v; draw(); });
    [plot.wrap, s, p, stats, read].forEach(function (n) { body.appendChild(n); });
    draw();
  };
})();
