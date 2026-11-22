/* Annotated source reader. Links look like source.html#path@text or #path:12 */
(function () {
  "use strict";
  var files = window.SOURCE, nav = document.getElementById("src-nav"), view = document.getElementById("src-view");
  var filter = document.getElementById("src-filter"), buttons = {}, current = null;
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }

  /* ---- quiet highlighting: strings, comments, keywords, numbers, definitions ---- */
  var KEYWORDS = "and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield None True False function var new this typeof const let of".split(" ");
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function highlight(lines, js) {
    var inDoc = false;
    return lines.map(function (line) {
      if (js) {
        if (inDoc) { if (line.indexOf("*/") >= 0) inDoc = false; return '<span class="cm">' + esc(line) + "</span>"; }
        if (/^\s*\/\*/.test(line)) { if (line.indexOf("*/") < 0) inDoc = true; return '<span class="cm">' + esc(line) + "</span>"; }
      } else {
        var quotes = (line.match(/"""/g) || []).length;
        if (inDoc) { if (quotes) inDoc = false; return '<span class="st">' + esc(line) + "</span>"; }
        if (/^\s*[rf]?"""/.test(line)) { if (quotes === 1) inDoc = true; return '<span class="st">' + esc(line) + "</span>"; }
      }
      var re = js ? /(\/\/.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d[\d.e+-]*\b|\b[A-Za-z_]\w*\b)/g
                  : /(#.*$|[rf]?"(?:[^"\\]|\\.)*"|[rf]?'(?:[^'\\]|\\.)*'|@\w+|\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_]\w*\b)/g;
      var out = "", at = 0, m, prev = "";
      while ((m = re.exec(line))) {
        var t = m[0];
        out += esc(line.slice(at, m.index)); at = m.index + t.length;
        var cls = /^(#|\/\/)/.test(t) && (js ? t[0] === "/" : t[0] === "#") ? "cm" : /^[rf]?["']/.test(t) ? "st" : t[0] === "@" ? "pp" : /^\d/.test(t) ? "nu"
          : KEYWORDS.indexOf(t) >= 0 ? "k" : prev === "def" || prev === "class" || (js && prev === "function") ? "fn" : /^[A-Z][a-zA-Z]+$/.test(t) ? "ty" : "";
        out += cls ? '<span class="' + cls + '">' + esc(t) + "</span>" : esc(t);
        prev = t;
      }
      return out + esc(line.slice(at));
    });
  }

  function show(path, keepHash) {
    var f = files.filter(function (q) { return q.path === path; })[0] || files[0];
    if (current === f.path) return f;
    current = f.path;
    Object.keys(buttons).forEach(function (p) { buttons[p].setAttribute("aria-current", p === f.path); });
    view.innerHTML = "";
    var head = el("div", "src-head"); head.appendChild(el("h1", "", f.path)); head.appendChild(el("p", "", f.title)); view.appendChild(head);
    var js = /\.m?js$/.test(f.path), all = [];
    f.blocks.forEach(function (b) { all = all.concat(b.code); });
    var html = highlight(all, js), k = 0;
    f.blocks.forEach(function (b) {
      var block = el("section", "block"), code = el("div", "code"), note = el("div", "note"), inner = el("div");
      code.tabIndex = 0; code.setAttribute("aria-label", f.path + " from line " + b.line);
      block.dataset.line = b.line; block.dataset.end = b.line + b.code.length - 1;
      b.code.forEach(function (_, i) {
        var row = el("div", "ln"); row.appendChild(el("i", "", String(b.line + i)));
        var s = el("span"); s.innerHTML = html[k++] || "​"; row.appendChild(s); code.appendChild(row);
      });
      inner.innerHTML = b.note; note.appendChild(inner); block.appendChild(code); block.appendChild(note); view.appendChild(block);
    });
    if (!keepHash) history.replaceState(null, "", "#" + f.path);
    document.title = f.path + " · Ten Dots";
    return f;
  }

  function go() {
    var h = decodeURIComponent(location.hash.slice(1));
    if (!h) return show(files[0].path, true);
    var at = h.indexOf("@"), colon = h.lastIndexOf(":"), path = h, line = 0;
    if (at > 0) path = h.slice(0, at); else if (colon > 0 && /^\d+$/.test(h.slice(colon + 1))) { path = h.slice(0, colon); line = +h.slice(colon + 1); }
    var f = show(path, true);
    if (at > 0) {
      var text = h.slice(at + 1);
      f.blocks.some(function (b) { return b.code.some(function (c, i) { if (c.indexOf(text) >= 0) { line = b.line + i; return true; } }); });
    }
    if (!line) return window.scrollTo(0, 0);
    view.querySelectorAll(".block").forEach(function (b) {
      if (line >= +b.dataset.line && line <= +b.dataset.end) {
        var row = b.querySelectorAll(".ln")[line - +b.dataset.line];
        (b.offsetHeight > innerHeight * .8 ? row : b).scrollIntoView({ block: "center" });
        b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
        row.style.background = "var(--blue-wash)";
      }
    });
  }

  var group = "";
  files.forEach(function (f) {
    if (f.group !== group) { group = f.group; var h = el("h3", "", group); nav.appendChild(h); }
    var b = el("button", "", f.path); b.type = "button"; b.addEventListener("click", function () { show(f.path); window.scrollTo(0, 0); });
    buttons[f.path] = b; nav.appendChild(b);
  });
  filter.addEventListener("input", function () {
    var q = filter.value.toLowerCase();
    files.forEach(function (f) {
      var hit = (f.path + " " + f.title).toLowerCase().indexOf(q) >= 0 || f.blocks.some(function (b) { return b.code.join("\n").toLowerCase().indexOf(q) >= 0; });
      buttons[f.path].style.display = hit ? "" : "none";
    });
  });
  window.addEventListener("hashchange", go);
  go();
})();
