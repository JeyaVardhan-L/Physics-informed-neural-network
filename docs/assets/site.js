/* Top bar behaviour, and the checklist memory on the route page. */
(function () {
  "use strict";
  var bar = document.querySelector(".topbar");
  function onScroll() { if (bar) bar.classList.toggle("scrolled", window.scrollY > 4); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  var here = document.querySelector('.topbar [aria-current="page"]');
  if (here && here.parentNode.scrollWidth > here.parentNode.clientWidth) here.parentNode.scrollLeft = here.offsetLeft - here.parentNode.offsetLeft - 24;

  document.querySelectorAll(".route input[type=checkbox]").forEach(function (box) {
    var key = "tendots." + box.id;
    try { box.checked = localStorage.getItem(key) === "1"; } catch (e) { /* storage is off */ }
    box.addEventListener("change", function () {
      try { localStorage.setItem(key, box.checked ? "1" : "0"); } catch (e) { /* storage is off */ }
    });
  });
})();
