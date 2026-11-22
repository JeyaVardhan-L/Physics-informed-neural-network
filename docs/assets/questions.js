/* The questions wheel: the active question sits in the middle, the rest blur away. */
(function () {
  "use strict";
  var all = window.QUESTIONS, wheel = document.getElementById("wheel"), list = document.getElementById("wheel-list"), card = document.getElementById("card");
  var select = document.getElementById("qselect"), foot = document.getElementById("qfoot"), chips = document.getElementById("qfilters");
  var cats = ["All"], cat = "All", shown = all, i = 0, buttons = [];
  all.forEach(function (q) { if (cats.indexOf(q.cat) < 0) cats.push(q.cat); });
  function el(tag, attrs, text) { var n = document.createElement(tag); Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); }); if (text !== undefined) n.textContent = text; return n; }

  cats.forEach(function (c) {
    var b = el("button", { type: "button", class: "chip", "aria-pressed": c === cat }, c);
    b.addEventListener("click", function () {
      cat = c; chips.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b); });
      shown = c === "All" ? all : all.filter(function (q) { return q.cat === c; }); build();
    });
    chips.appendChild(b);
  });

  function build() {
    list.innerHTML = ""; select.innerHTML = ""; i = 0;
    buttons = shown.map(function (q, j) {
      var li = el("li"), b = el("button", { type: "button" }, q.q);
      b.addEventListener("click", function () { go(j); }); li.appendChild(b); list.appendChild(li);
      select.appendChild(el("option", { value: j }, q.q));
      return b;
    });
    go(0);
  }
  function go(j) {
    i = Math.max(0, Math.min(shown.length - 1, j));
    var q = shown[i];
    buttons.forEach(function (b, k) {
      var d = Math.abs(k - i);
      b.style.filter = d ? "blur(" + Math.min(4.5, .9 + d * .9) + "px)" : "none";
      b.style.opacity = d ? Math.max(.12, .78 - d * .14) : 1;
      b.setAttribute("aria-current", d === 0); b.tabIndex = d < 4 ? 0 : -1;
    });
    if (wheel.clientHeight) {
      var b = buttons[i];
      list.style.transform = "translateY(" + (wheel.clientHeight / 2 - b.offsetTop - b.offsetHeight / 2) + "px)";
    }
    card.innerHTML = '<div class="row"><span class="lab">IN:</span><span class="q"></span></div><div class="row"><span class="lab">OUT:</span><div class="a">' + q.a + "</div></div>";
    card.querySelector(".q").textContent = q.q;
    select.value = i;
    foot.innerHTML = "<span>" + (i + 1) + " / " + shown.length + " · " + q.cat + "</span>" + (q.src ? '<a href="' + q.src + '">read more ↗</a>' : "");
  }
  document.addEventListener("keydown", function (e) {
    if (/^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); go(i + 1); }
    if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); go(i - 1); }
  });
  var last = 0;
  wheel.addEventListener("wheel", function (e) {
    e.preventDefault();
    if (Date.now() - last < 130 || Math.abs(e.deltaY) < 4) return;
    last = Date.now(); go(i + (e.deltaY > 0 ? 1 : -1));
  }, { passive: false });
  select.addEventListener("change", function () { go(+select.value); });
  document.getElementById("qprev").addEventListener("click", function () { go(i - 1); });
  document.getElementById("qnext").addEventListener("click", function () { go(i + 1); });
  window.addEventListener("resize", function () { go(i); });
  build();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { go(i); });
})();
