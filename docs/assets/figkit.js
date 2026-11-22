/* Helpers shared by every figure, and the loader that builds them. */
(function () {
  "use strict";
  var Fig = window.Fig = { figures: {} };
  var NS = "http://www.w3.org/2000/svg";
  Fig.reduced = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function put(node, attrs, kids) {
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined) return;
      if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
      else if (k === "html") node.innerHTML = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    });
    (kids || []).forEach(function (c) { if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  var el = Fig.el = function (tag, attrs, kids) { return put(document.createElement(tag), attrs, kids); };
  var sv = Fig.sv = function (tag, attrs, kids) { return put(document.createElementNS(NS, tag), attrs, kids); };
  Fig.clear = function (node) { while (node.firstChild) node.removeChild(node.firstChild); return node; };

  Fig.btn = function (label, onclick, pressed) {
    var b = el("button", { type: "button", class: "btn", text: label, onclick: onclick });
    if (pressed !== undefined) b.setAttribute("aria-pressed", pressed);
    return b;
  };

  /* A row of buttons where exactly one is on. */
  Fig.choice = function (labels, start, onpick, cls) {
    var buttons = labels.map(function (label, i) {
      var b = Fig.btn(label, function () { set(i); }, i === start);
      if (cls) b.className = cls;
      return b;
    });
    function set(i) {
      buttons.forEach(function (b, j) { b.setAttribute("aria-pressed", j === i); });
      onpick(i);
    }
    var node = el("div", { class: "controls", role: "group" }, buttons);
    node.set = set;
    return node;
  };

  Fig.slider = function (label, min, max, step, value, fmt, oninput) {
    var out = el("output", { text: fmt(value) });
    var input = el("input", { type: "range", min: min, max: max, step: step, value: value, "aria-label": label,
      oninput: function () { out.textContent = fmt(+input.value); oninput(+input.value); } });
    var node = el("label", { class: "slider" }, [el("span", { text: label }), input, out]);
    node.set = function (v) { input.value = v; out.textContent = fmt(+input.value); };
    node.value = function () { return +input.value; };
    return node;
  };

  Fig.readout = function (tag) {
    var body = el("div");
    var node = el("div", { class: "readout", "aria-live": "polite" }, [tag ? el("span", { class: "tag", text: tag }) : null, body]);
    node.say = function (html) { body.innerHTML = html; };
    return node;
  };

  Fig.stats = function (labels) {
    var cells = labels.map(function (l) { return el("b", { text: "..." }); });
    var node = el("div", { class: "stats" }, labels.map(function (l, i) { return el("div", {}, [el("span", { text: l }), cells[i]]); }));
    node.set = function (values) { values.forEach(function (v, i) { if (cells[i].textContent !== v) cells[i].textContent = v; }); };
    return node;
  };

  /* PREV  3 / 10  NEXT */
  Fig.stepper = function (n, show) {
    var i = 0, count = el("span", { class: "count" });
    var prev = Fig.btn("← Prev", function () { go(i - 1); }), next = Fig.btn("Next →", function () { go(i + 1); });
    function go(j) {
      i = Math.max(0, Math.min(n - 1, j));
      prev.disabled = i === 0; next.disabled = i === n - 1;
      count.textContent = (i + 1) + " / " + n;
      show(i);
    }
    var node = el("div", { class: "controls" }, [prev, count, next]);
    node.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") go(i - 1); else if (e.key === "ArrowRight") go(i + 1);
    });
    node.go = go;
    return node;
  };

  Fig.tween = function (from, to, draw, ms) {
    if (Fig.reduced) { draw(to); return function () {}; }
    var t0 = performance.now(), stop = false;
    (function frame(now) {
      if (stop) return;
      var u = Math.min(1, (now - t0) / (ms || 450)), e = 1 - Math.pow(1 - u, 3);
      draw(from.map(function (a, i) { return a + (to[i] - a) * e; }));
      if (u < 1) requestAnimationFrame(frame);
    })(t0);
    return function () { stop = true; };
  };

  Fig.sci = function (v) {
    if (v === 0) return "0";
    if (!isFinite(v)) return "∞";
    var a = Math.abs(v);
    if (a >= 0.01 && a < 1000) return v.toPrecision(2).indexOf("e") < 0 ? String(+v.toPrecision(3)) : v.toFixed(0);
    var p = v.toExponential(1).split("e");
    return p[0] + "e" + (+p[1]);
  };
  Fig.fixed = function (v, n) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(n); };

  Fig.legend = function (items) {
    var styles = { truth: ["dash", "--muted"], blue: ["", "--blue"], red: ["", "--red"], dot: ["dot", ""] };
    return el("div", { class: "legend" }, items.map(function (it) {
      var s = styles[it[0]], i = el("i", { class: s[0] });
      if (s[1]) i.style.borderColor = "var(" + s[1] + ")";
      return el("span", {}, [i, it[1]]);
    }));
  };

  /* ---- plot: axes, scales, paths and a hover crosshair ---- */
  Fig.plot = function (o) {
    var W = o.w || 960, H = o.h || 330, L = o.left === undefined ? 58 : o.left, R = W - (o.right || 18), T = o.top || 14, B = H - 44;
    var xr = o.x, yr = o.y, lg = o.logy;
    function X(v) { return L + (v - xr[0]) / (xr[1] - xr[0]) * (R - L); }
    function Y(v) {
      if (lg) { v = Math.log10(Math.max(v, 1e-30)); return B - (v - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0])) * (B - T); }
      return B - (v - yr[0]) / (yr[1] - yr[0]) * (B - T);
    }
    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.label || "plot" });
    var clipId = "clip" + (Fig.plot.n = (Fig.plot.n || 0) + 1);
    svg.appendChild(sv("defs", {}, [sv("clipPath", { id: clipId }, [sv("rect", { x: L, y: T - 6, width: R - L, height: B - T + 12 })])]));
    var back = sv("g"), axes = sv("g"), layer = sv("g", { "clip-path": "url(#" + clipId + ")" }), over = sv("g");
    [back, axes, layer, over].forEach(function (g) { svg.appendChild(g); });

    function drawAxes() {
      Fig.clear(axes);
      (o.yt || []).forEach(function (v) {
        axes.appendChild(sv("line", { class: "s-grid", x1: L, x2: R, y1: Y(v), y2: Y(v) }));
        axes.appendChild(sv("text", { class: "s-text-sm", x: L - 9, y: Y(v) + 3.5, "text-anchor": "end", text: o.yfmt ? o.yfmt(v) : String(v).replace("-", "−") }));
      });
      (o.xt || []).forEach(function (v) {
        axes.appendChild(sv("line", { class: "s-axis", x1: X(v), x2: X(v), y1: B, y2: B + 5 }));
        axes.appendChild(sv("text", { class: "s-text-sm", x: X(v), y: B + 18, "text-anchor": "middle", text: o.xfmt ? o.xfmt(v) : String(v) }));
      });
      axes.appendChild(sv("line", { class: "s-axis", x1: L, x2: R, y1: B, y2: B }));
      if (o.xlabel) axes.appendChild(sv("text", { class: "s-text-sm", x: (L + R) / 2, y: H - 6, "text-anchor": "middle", text: o.xlabel }));
      if (o.ylabel) axes.appendChild(sv("text", { class: "s-text-sm", transform: "translate(12 " + (T + B) / 2 + ") rotate(-90)", "text-anchor": "middle", text: o.ylabel }));
    }
    drawAxes();
    var p = { svg: svg, X: X, Y: Y, back: back, layer: layer, over: over, L: L, R: R, T: T, B: B, W: W, H: H };
    p.setY = function (y, yt) { yr = y; if (yt) o.yt = yt; drawAxes(); };
    p.setX = function (x, xt) { xr = x; if (xt) o.xt = xt; drawAxes(); };
    p.band = function (a, b, text) {
      back.appendChild(sv("rect", { class: "s-fill-soft", x: X(a), y: T, width: X(b) - X(a), height: B - T }));
      if (text) back.appendChild(sv("text", { class: "s-text-sm", x: X(a) + 8, y: T + 14, text: text }));
    };
    p.d = function (ts, xs) {
      var s = "", pen = false;
      for (var i = 0; i < ts.length; i++) {
        var y = xs[i];
        if (!isFinite(y)) { pen = false; continue; }
        s += (pen ? "L" : "M") + X(ts[i]).toFixed(1) + " " + Math.max(-4000, Math.min(4000, Y(y))).toFixed(1);
        pen = true;
      }
      return s;
    };
    p.path = function (cls) { var n = sv("path", { class: cls }); layer.appendChild(n); n.set = function (ts, xs) { n.setAttribute("d", p.d(ts, xs)); }; return n; };
    p.dots = function (ts, xs, r) {
      var g = sv("g"); over.appendChild(g);
      g.set = function (a, b) { Fig.clear(g); a.forEach(function (t, i) { g.appendChild(sv("circle", { class: "s-dot", cx: X(t), cy: Y(b[i]), r: r || 5 })); }); };
      g.set(ts, xs); return g;
    };
    /* series: function(t) -> [[name, value], ...] */
    p.hover = function (series, fmtx) {
      var line = sv("line", { class: "s-axis", y1: T, y2: B, visibility: "hidden" });
      var box = sv("g", { visibility: "hidden" }), rect = sv("rect", { class: "s-card", rx: 3 }), txt = sv("g");
      box.appendChild(rect); box.appendChild(txt); over.appendChild(line); over.appendChild(box);
      function move(e) {
        var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
        if (px < L || px > R) return hide();
        var t = xr[0] + (px - L) / (R - L) * (xr[1] - xr[0]), rows = series(t);
        if (!rows) return hide();
        Fig.clear(txt);
        [[fmtx ? fmtx(t) : "t = " + t.toFixed(2) + " s", null]].concat(rows).forEach(function (row, i) {
          txt.appendChild(sv("text", { class: i ? "s-text" : "s-text-sm", x: 10, y: 17 + i * 17, text: row[0] }));
          if (row[1] !== null) txt.appendChild(sv("text", { class: "s-text", x: 168, y: 17 + i * 17, "text-anchor": "end", text: typeof row[1] === "number" ? Fig.fixed(row[1], 3) : row[1] }));
        });
        var h = 26 + rows.length * 17;
        rect.setAttribute("width", 178); rect.setAttribute("height", h);
        box.setAttribute("transform", "translate(" + (px + 190 > R ? px - 190 : px + 12) + " " + (T + 6) + ")");
        line.setAttribute("x1", px); line.setAttribute("x2", px);
        line.setAttribute("visibility", "visible"); box.setAttribute("visibility", "visible");
      }
      function hide() { line.setAttribute("visibility", "hidden"); box.setAttribute("visibility", "hidden"); }
      svg.addEventListener("pointermove", move); svg.addEventListener("pointerdown", move); svg.addEventListener("pointerleave", hide);
    };
    p.wrap = el("div", { class: "plotwrap" }, [svg]);
    return p;
  };
  Fig.timePlot = function (extra) {
    var o = { x: [0, 1], y: [-1.5, 1.5], xt: [0, .2, .4, .6, .8, 1], yt: [-1.5, -1, -.5, 0, .5, 1, 1.5], xlabel: "time (s)", ylabel: "position" };
    Object.keys(extra || {}).forEach(function (k) { o[k] = extra[k]; });
    return Fig.plot(o);
  };
  Fig.at = function (ts, xs, t) {
    var u = (t - ts[0]) / (ts[ts.length - 1] - ts[0]) * (ts.length - 1), i = Math.max(0, Math.min(ts.length - 2, Math.floor(u)));
    return xs[i] + (xs[i + 1] - xs[i]) * (u - i);
  };
  Fig.whenVisible = function (node, fn) {
    if (!("IntersectionObserver" in window)) return fn(true);
    new IntersectionObserver(function (es) { es.forEach(function (e) { fn(e.isIntersecting); }); }, { threshold: .15 }).observe(node);
  };

  Fig.load = function () {
    document.querySelectorAll("figure.fig[data-fig]").forEach(function (fig) {
      var body = fig.querySelector(".fig-body"), make = Fig.figures[fig.dataset.fig];
      try {
        if (!make) throw new Error("no figure called " + fig.dataset.fig);
        make(body);
      } catch (err) {
        body.textContent = "This figure failed to load."; body.classList.add("failed");
        console.error(err);
      }
    });
  };
  document.addEventListener("DOMContentLoaded", Fig.load);
})();
