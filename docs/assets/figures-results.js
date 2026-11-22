/* Figures that draw recorded runs from results-data.js (built from results/*.json). */
(function () {
  "use strict";
  var F = Fig.figures, el = Fig.el, sv = Fig.sv, R = window.RESULTS || {}, OSC = Engine.oscillator(1, 3, 225);
  function median(v) { v = v.slice().sort(function (a, b) { return a - b; }); var m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; }
  function rowsOf(name, setting) { return R[name].filter(function (r) { return r.setting === setting; }); }
  function settings(name) { var s = []; R[name].forEach(function (r) { if (s.indexOf(r.setting) < 0) s.push(r.setting); }); return s; }
  function med(name, setting, key) { return median(rowsOf(name, setting).map(function (r) { return r[key || "mse"]; })); }
  function seed0(name, setting) { return rowsOf(name, setting).filter(function (r) { return r.curve; })[0]; }
  function head(method, seed) { return R.headline.rows.filter(function (r) { return r.method === method && r.seed === seed; })[0]; }
  function e1(v) { return "1e" + Math.round(Math.log10(v)); }
  function dotsFor(seed) { return head("pinn", seed).dots; }

  /* ---- Chapter 3: what the plain net does after the last dot ---- */
  F.plainseeds = function (body) {
    var T = R.headline.t, plot = Fig.timePlot({ label: "Plain networks from different seeds", y: [-2.2, 2.2], yt: [-2, -1, 0, 1, 2] }), cur;
    plot.band(0, .3, "DOTS"); plot.path("s-truth").set(T, R.headline.truth);
    var line = plot.path("s-bad"), dots = plot.dots([], [], 5), stats = Fig.stats(["seed", "error on the dots", "error after the dots"]), read = Fig.readout();
    var seeds = [0, 1, 2];
    function show(i) {
      var r = head("plain", seeds[i]); cur = r.curve; line.set(T, cur); dots.set(r.dots.t, r.dots.x);
      stats.set([String(seeds[i]), Fig.sci(r.data_loss), Fig.sci(r.mse)]);
      var end = cur[cur.length - 1];
      read.say("At t = 1 this net says <b>" + Fig.fixed(end, 2) + "</b> and the spring is at " + Fig.fixed(OSC.x(1), 2) + ". Nothing in the loss prefers one continuation over another, so the random starting weights decide.");
    }
    body.appendChild(Fig.choice(["seed 0", "seed 1", "seed 2"], 0, show));
    body.appendChild(Fig.legend([["truth", "true motion"], ["red", "plain net"], ["dot", "measurements"]]));
    [plot.wrap, stats, read].forEach(function (n) { body.appendChild(n); });
    plot.hover(function (t) { return [["plain", Fig.at(T, cur, t)], ["true", OSC.x(t)]]; }); show(0);
  };

  /* ---- Chapter 5: where the equation is enforced ---- */
  F.coverage = function (body) {
    var T = R.headline.t, names = settings("ablation_coverage"), plot = Fig.timePlot({ label: "PINN with the equation enforced up to a chosen time" });
    var shade = sv("rect", { class: "s-fill-soft", y: plot.T, height: plot.B - plot.T }), tag = sv("text", { class: "s-text-blue", y: plot.T + 14 }), ticks = sv("g");
    [shade, tag, ticks].forEach(function (n) { plot.back.appendChild(n); });
    plot.path("s-truth").set(T, R.headline.truth);
    var line = plot.path("s-curve"), d = dotsFor(0), cur; plot.dots(d.t, d.x, 5);
    var stats = Fig.stats(["enforced on", "equation error where enforced", "error after the dots (median of 3)"]), read = Fig.readout();
    function show(i) {
      var t = names[i], r = seed0("ablation_coverage", t); cur = r.curve;
      shade.setAttribute("x", plot.X(0)); shade.setAttribute("width", plot.X(t) - plot.X(0));
      tag.setAttribute("x", plot.X(0) + 8); tag.textContent = "EQUATION ENFORCED HERE";
      Fig.clear(ticks);
      for (var k = 0; k < 60; k++) ticks.appendChild(sv("line", { class: "s-line", x1: plot.X(t * k / 59), x2: plot.X(t * k / 59), y1: plot.B - 7, y2: plot.B }));
      line.set(T, cur); line.setAttribute("class", t < 1 ? "s-bad" : "s-curve");
      stats.set(["0 to " + t + " s", Fig.sci(med("ablation_coverage", t, "physics_loss")), Fig.sci(med("ablation_coverage", t))]);
      read.say(t === 0.3 ? "The equation is checked only where there are dots. Inside the window the curve is excellent. Outside, this is an ordinary network again."
        : t < 1 ? "The curve is right as far as the equation reaches and then lets go. The 60 ticks are the only places where the network is told anything about springs."
        : "The equation covers the whole second, so the curve is held all the way across. The dots did not move and no new data was added.");
    }
    body.appendChild(Fig.choice(names.map(function (n) { return "up to " + n + " s"; }), 0, show));
    [plot.wrap, stats, read].forEach(function (n) { body.appendChild(n); });
    plot.hover(function (t) { return [["PINN", Fig.at(T, cur, t)], ["true", OSC.x(t)]]; }); show(0);
  };

  /* ---- Chapter 5: the weight ---- */
  F.weight = function (body) {
    var T = R.headline.t, names = settings("ablation_weight"), i0 = Math.max(0, names.indexOf(10));
    var plot = Fig.timePlot({ label: "PINN at a chosen physics weight" }), line, d = dotsFor(0), cur;
    plot.path("s-truth").set(T, R.headline.truth); line = plot.path("s-curve"); plot.dots(d.t, d.x, 5);
    var stats = Fig.stats(["weight λ", "data loss", "physics loss", "error after the dots"]), read = Fig.readout("median of three seeds");
    var best = names.reduce(function (b, n) { return med("ablation_weight", n) < med("ablation_weight", b) ? n : b; }, names[0]);
    function show(i) {
      var w = names[i]; cur = seed0("ablation_weight", w).curve; line.set(T, cur);
      var m = med("ablation_weight", w), dl = med("ablation_weight", w, "data_loss"), pl = med("ablation_weight", w, "physics_loss");
      stats.set([String(w), Fig.sci(dl), Fig.sci(pl), Fig.sci(m)]);
      var noise = 1e-4;
      read.say((w === best ? "<span class='good'>Best of these.</span> " : "") +
        (dl < noise / 3 ? "The data loss is well below the noise variance of 1e-4, which means the net is bending to follow noise. " : dl > 3 * noise ? "The data loss is far above the noise variance of 1e-4. The equation is so loud that the dots are being ignored. " : "The data loss is about the noise variance of 1e-4, where an honest fit should stop. ") +
        "The equation error is " + Fig.sci(pl) + ".");
    }
    var c = Fig.choice(names.map(function (n) { return "λ = " + n; }), i0, show, "chip");
    [c, plot.wrap, stats, read].forEach(function (n) { body.appendChild(n); });
    plot.hover(function (t) { return [["PINN", Fig.at(T, cur, t)], ["true", OSC.x(t)]]; }); show(i0);
  };

  /* ---- Chapter 6: loss curves with a scrubber ---- */
  F.history = function (body) {
    var h = head("pinn", 0).history, max = h[h.length - 1].step, steps = h.map(function (r) { return r.step; });
    var xt = []; for (var q = 0; q <= max; q += 1000) xt.push(q);
    var plot = Fig.plot({ h: 320, x: [0, max], y: [1e-8, 10], logy: true, xt: xt, yt: [1e-8, 1e-6, 1e-4, 1e-2, 1], yfmt: e1, xlabel: "loss evaluations", ylabel: "loss", label: "Data and physics loss during training" });
    var sw = h.filter(function (r) { return r.stage === "lbfgs"; })[0].step;
    plot.back.appendChild(sv("line", { class: "s-dash", x1: plot.X(sw), x2: plot.X(sw), y1: plot.T, y2: plot.B }));
    plot.back.appendChild(sv("text", { class: "s-text-sm", x: plot.X(sw) - 8, y: plot.T + 14, "text-anchor": "end", text: "ADAM" }));
    plot.back.appendChild(sv("text", { class: "s-text-sm", x: plot.X(sw) + 8, y: plot.T + 14, text: "L-BFGS" }));
    plot.back.appendChild(sv("line", { class: "s-faint", x1: plot.L, x2: plot.R, y1: plot.Y(1e-4), y2: plot.Y(1e-4) }));
    plot.back.appendChild(sv("text", { class: "s-text-sm", x: plot.R - 4, y: plot.Y(1e-4) - 6, "text-anchor": "end", text: "NOISE VARIANCE" }));
    plot.path("s-truth").set(steps, h.map(function (r) { return r.data; })); plot.path("s-curve").set(steps, h.map(function (r) { return r.physics; }));
    var mark = sv("line", { class: "s-line", y1: plot.T, y2: plot.B }); plot.over.appendChild(mark);
    var stats = Fig.stats(["evaluation", "stage", "data loss", "physics loss"]), read = Fig.readout();
    var start = h[0].physics;
    function draw(i) {
      var r = h[i]; mark.setAttribute("x1", plot.X(r.step)); mark.setAttribute("x2", plot.X(r.step));
      stats.set([String(r.step), r.stage === "adam" ? "Adam" : "L-BFGS", Fig.sci(r.data), Fig.sci(r.physics)]);
      read.say(i === 0 ? "Random weights. The curve is nearly flat, so it almost satisfies the equation already (physics loss " + Fig.sci(start) + "), and it misses every dot."
        : r.stage === "adam" && r.data > 5e-3 ? "The network has found the lazy answer: stay near zero, which the equation likes, and pay for the dots."
        : r.stage === "adam" ? "The dots are being fitted and the bend is being pushed out of the window, a little further each time, into the part with no data."
        : "L-BFGS uses the recent gradients to estimate curvature, so it takes useful steps where Adam zigzags. The physics loss falls about a hundredfold within a few hundred evaluations and then stays put, because the noisy dots pull against it.");
    }
    var s = Fig.slider("scrub", 0, h.length - 1, 1, 0, function (v) { return String(h[v].step); }, draw);
    body.appendChild(Fig.legend([["truth", "data loss"], ["blue", "physics loss"]]));
    [plot.wrap, s, stats, read].forEach(function (n) { body.appendChild(n); }); draw(0);
  };

  /* ---- Chapter 7: every seed, every method ---- */
  F.strip = function (body) {
    var M = [["plain", "plain net", "Adam only. The best version of the plain net: turning L-BFGS on makes it worse."],
      ["plain_lbfgs", "plain net + L-BFGS", "The same net trained harder. It fits the noise in the dots almost perfectly and throws the future away."],
      ["zero", "always say 0", "No training at all: it answers that the spring has stopped. Any method that loses to this has learned nothing about the future."],
      ["pinn", "PINN", "The same network as the plain net, plus the equation."],
      ["classical", "least squares", "The exact solution formula with two unknown constants fitted to the dots. It uses the same information as the PINN, and it is the target."]];
    var W = 960, L = 170, RR = 930, lo = -8, hi = 3, read = Fig.readout("method"), g = sv("g");
    function X(v) { return L + (Math.log10(v) - lo) / (hi - lo) * (RR - L); }
    var svg = sv("svg", { viewBox: "0 0 " + W + " 330", role: "img", "aria-label": "Future error for each method and seed, log scale" });
    for (var p = lo; p <= hi; p += 2) {
      svg.appendChild(sv("line", { class: "s-grid", x1: X(Math.pow(10, p)), x2: X(Math.pow(10, p)), y1: 20, y2: 280 }));
      svg.appendChild(sv("text", { class: "s-text-sm", x: X(Math.pow(10, p)), y: 298, "text-anchor": "middle", text: "1e" + p }));
    }
    svg.appendChild(sv("text", { class: "s-text-sm", x: (L + RR) / 2, y: 322, "text-anchor": "middle", text: "MEAN SQUARED ERROR ON 0.3 TO 1 s (LOG SCALE, LEFT IS BETTER)" }));
    var fl = X(R.headline.noise_floor);
    svg.appendChild(sv("line", { class: "s-dash", x1: fl, x2: fl, y1: 14, y2: 280 }));
    svg.appendChild(sv("text", { class: "s-text-blue", x: fl, y: 10, "text-anchor": "middle", text: "noise floor " + Fig.sci(R.headline.noise_floor) }));
    svg.appendChild(g);
    function draw(on) {
      Fig.clear(g);
      M.forEach(function (m, i) {
        var y = 48 + i * 52, rows = R.headline.rows.filter(function (r) { return r.method === m[0]; }), md = R.headline.median_mse[m[0]];
        var row = sv("g", { "data-hit": "1", tabindex: "0", role: "button", "aria-label": m[1] + ", median error " + Fig.sci(md), onclick: function () { draw(i); }, onkeydown: function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); draw(i); } } });
        row.appendChild(sv("rect", { x: 0, y: y - 24, width: W, height: 46, fill: i === on ? "#eaedfe" : "transparent", "fill-opacity": .7 }));
        row.appendChild(sv("text", { class: i === on ? "s-text-blue" : "s-text", x: 10, y: y + 4, text: m[1] }));
        rows.forEach(function (r, j) {
          var bad = m[0].indexOf("plain") === 0;
          row.appendChild(sv("circle", { cx: X(Math.min(Math.max(r.mse, 1e-8), 1e3)), cy: y + (m[0] === "zero" ? 0 : (j % 3 - 1) * 7), r: 4.5, fill: bad ? "var(--red)" : m[0] === "pinn" ? "var(--blue)" : "var(--muted)", "fill-opacity": .75, stroke: "#fff", "stroke-width": 1.2 }));
        });
        row.appendChild(sv("line", { x1: X(md), x2: X(md), y1: y - 18, y2: y + 16, stroke: "var(--ink)", "stroke-width": 1.6 }));
        g.appendChild(row);
      });
      var v = R.headline.rows.filter(function (r) { return r.method === M[on][0]; }).map(function (r) { return r.mse; });
      read.say("<b>" + M[on][1] + "</b>. " + M[on][2] + " Median " + Fig.sci(R.headline.median_mse[M[on][0]]) + ", best seed " + Fig.sci(Math.min.apply(0, v)) + ", worst seed " + Fig.sci(Math.max.apply(0, v)) + ".");
    }
    body.appendChild(el("div", { class: "plotwrap" }, [svg])); body.appendChild(read); draw(3);
  };

  F.curves = function (body) {
    var T = R.headline.t, plot = Fig.timePlot({ label: "Plain net and PINN after training", h: 350 }), a, b;
    plot.band(0, .3, "DOTS");
    var red = plot.path("s-bad"), blue = plot.path("s-curve"), truth = plot.path("s-truth"), dots = plot.dots([], [], 5);
    truth.set(T, R.headline.truth);
    var stats = Fig.stats(["plain net", "PINN", "least squares", "plain / PINN"]);
    function show(i) {
      a = head("plain", i); b = head("pinn", i); red.set(T, a.curve); blue.set(T, b.curve); dots.set(b.dots.t, b.dots.x);
      stats.set([Fig.sci(a.mse), Fig.sci(b.mse), Fig.sci(head("classical", i).mse), Math.round(a.mse / b.mse).toLocaleString("en") + "×"]);
    }
    body.appendChild(Fig.choice(["seed 0", "seed 1", "seed 2"], 0, show));
    body.appendChild(Fig.legend([["truth", "true motion"], ["red", "plain net"], ["blue", "PINN"], ["dot", "measurements"]]));
    body.appendChild(plot.wrap); body.appendChild(stats);
    plot.hover(function (t) { return [["true", OSC.x(t)], ["PINN", Fig.at(T, b.curve, t)], ["plain", Fig.at(T, a.curve, t)]]; }); show(0);
  };

  /* ---- Chapter 8: break it, one thing at a time ---- */
  F.breaks = function (body) {
    var T = R.headline.t, groups = [["ablation_activation", "activation"], ["ablation_collocation", "points checked"], ["ablation_optimiser", "optimiser"], ["ablation_noise", "noise in dots"], ["ablation_wrong_physics", "equation told"]];
    var plot = Fig.timePlot({ label: "PINN curve for the chosen setting" }), cur, line, gi = 0;
    line = plot.path("s-curve"); plot.path("s-truth").set(T, R.headline.truth);
    var tbody = el("tbody"), table = el("table", { class: "row-pick" }, [el("thead", {}, [el("tr", {}, ["setting", "future error", "least squares", "loss on points", "true equation error"].map(function (h, i) { return el("th", { class: i ? "num" : "", text: h }); }))]), tbody]);
    function pick(name, s) {
      cur = seed0(name, s).curve; line.set(T, cur); var m = med(name, s); line.setAttribute("class", m > 3e-3 ? "s-bad" : "s-curve");
      tbody.querySelectorAll("tr").forEach(function (tr) { tr.setAttribute("aria-selected", tr.dataset.s === String(s)); });
    }
    function group(i) {
      gi = i; Fig.clear(tbody);
      settings(groups[i][0]).forEach(function (s) {
        var n = groups[i][0], tr = el("tr", { tabindex: "0", "data-s": s, onclick: function () { pick(n, s); }, onkeydown: function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(n, s); } } },
          [el("td", { text: String(s) })].concat([med(n, s), med(n, s, "classical"), med(n, s, "physics_loss"), med(n, s, "dense_residual")].map(function (v) { return el("td", { class: "num", text: Fig.sci(v) }); })));
        tbody.appendChild(tr);
      });
      var all = settings(groups[i][0]); pick(groups[i][0], i === 1 ? all[0] : all[all.length - 1]);
    }
    body.appendChild(Fig.choice(groups.map(function (g) { return g[1]; }), 0, group));
    body.appendChild(plot.wrap); body.appendChild(el("div", { class: "tablewrap", style: "margin:16px 0 0" }, [table]));
    plot.hover(function (t) { return [["PINN", Fig.at(T, cur, t)], ["true", OSC.x(t)]]; }); group(0);
  };

  F.frequency = function (body) {
    var T = R.headline.t, rows = R.frequency.rows, names = [];
    rows.forEach(function (r) { if (names.indexOf(r.setting) < 0) names.push(r.setting); });
    var plot = Fig.timePlot({ label: "PINN on a faster spring" }), truth = plot.path("s-truth"), line = plot.path("s-curve"), ticks = sv("g"), cur; plot.back.appendChild(ticks);
    plot.band(0, .3, "DOTS");
    var stats = Fig.stats(["loss at the checked points", "same error, checked densely", "future error (median of 3)"]), read = Fig.readout();
    function m(s, key) { return median(rows.filter(function (r) { return r.setting === s; }).map(function (r) { return r[key]; })); }
    function show(i) {
      var s = names[i], r = rows.filter(function (q) { return q.setting === s && q.curve; })[0], omega = s.match(/omega (\d+)/)[1], n = +s.match(/(\d+) points/)[1];
      cur = r.curve; truth.set(T, R.frequency.truth[omega]); line.set(T, cur); line.setAttribute("class", m(s, "mse") > 3e-3 ? "s-bad" : "s-curve");
      Fig.clear(ticks);
      for (var k = 0; k < n; k++) ticks.appendChild(sv("line", { class: "s-line", x1: plot.X(k / (n - 1)), x2: plot.X(k / (n - 1)), y1: plot.B - 7, y2: plot.B }));
      var pl = m(s, "physics_loss"), dr = m(s, "dense_residual"), ms = m(s, "mse");
      stats.set([Fig.sci(pl), Fig.sci(dr), Fig.sci(ms)]);
      read.say((ms < 1e-3 ? "The prediction holds. " : "<span class='bad'>The prediction fails.</span> ") +
        (dr > 100 * pl ? "The training loss says " + Fig.sci(pl) + ". Measured at 2,000 times instead of " + n + ", the same quantity is " + Fig.sci(dr) + ". The network obeys the equation at the ticks and breaks it between them."
          : "The equation error is about the same on and between the ticks, so the training loss can be believed here."));
    }
    body.appendChild(Fig.choice(names, 0, show, "chip"));
    [plot.wrap, stats, read].forEach(function (x) { body.appendChild(x); });
    plot.hover(function (t) { return [["PINN", Fig.at(T, cur, t)]]; }); show(0);
  };

  /* ---- Chapter 9: discover c and k ---- */
  F.inverse = function (body) {
    var rows = R.inverse, noises = []; rows.forEach(function (r) { if (noises.indexOf(r.noise) < 0) noises.push(r.noise); });
    var max = Math.max.apply(0, rows.filter(function (r) { return r.trace; }).map(function (r) { return r.trace[r.trace.length - 1][0]; }));
    var xt = [0, 1000, 2000, 3000, 4000, 5000, 6000].filter(function (v) { return v <= max; });
    var pk = Fig.plot({ h: 220, x: [0, max], y: [90, 260], xt: xt, yt: [100, 150, 200, 250], ylabel: "k", label: "Estimate of k during training" });
    var pc = Fig.plot({ h: 220, x: [0, max], y: [0, 6], xt: xt, yt: [0, 2, 4, 6], xlabel: "loss evaluations", ylabel: "c", label: "Estimate of c during training" });
    [[pk, 225], [pc, 3]].forEach(function (p) { p[0].back.appendChild(sv("line", { class: "s-truth", x1: p[0].L, x2: p[0].R, y1: p[0].Y(p[1]), y2: p[0].Y(p[1]) })); p[0].back.appendChild(sv("text", { class: "s-text-sm", x: p[0].R - 4, y: p[0].Y(p[1]) - 7, "text-anchor": "end", text: "TRUE VALUE " + p[1] })); });
    var lk = pk.path("s-curve"), lc = pc.path("s-curve"), stats = Fig.stats(["k found", "c found", "range of k, 3 seeds", "range of c, 3 seeds"]), read = Fig.readout();
    function show(i) {
      var rs = rows.filter(function (r) { return r.noise === noises[i]; }), r = rs.filter(function (q) { return q.trace; })[0], tr = r.trace;
      var st = tr.map(function (q) { return q[0]; });
      lk.set(st, tr.map(function (q) { return q[2]; })); lc.set(st, tr.map(function (q) { return q[1]; }));
      function rng(key, n) { var v = rs.map(function (q) { return q[key]; }); return Math.min.apply(0, v).toFixed(n) + " to " + Math.max.apply(0, v).toFixed(n); }
      stats.set([r.k.toFixed(1), r.c.toFixed(2), rng("k", 1), rng("c", 2)]);
      read.say(noises[i] === 0 ? "With perfect dots both numbers are recovered closely. What error remains comes from the optimiser and the finite weight on the dots."
        : "With noise of " + noises[i] + " in each dot, k is pinned down much better than c. Stiffness sets the timing of the swing, which ten dots see clearly. Friction only shows as a slow loss of height, and 0.3 s is less than one swing.");
    }
    body.appendChild(Fig.choice(noises.map(function (n) { return "noise " + n; }), 1, show));
    [pk.wrap, pc.wrap, stats, read].forEach(function (x) { body.appendChild(x); }); show(1);
  };
})();
