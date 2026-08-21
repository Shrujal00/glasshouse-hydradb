/* Glasshouse showcase — reveal on scroll, count-up, mobile sheet. */
(function () {
  'use strict';

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── reveal ──────────────────────────────────────────── */
  var anim = document.querySelectorAll('.anim');
  if (reduced) {
    anim.forEach(function (el) { el.classList.add('visible'); });
  } else {
    var reveal = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var d = getComputedStyle(e.target).getPropertyValue('--d');
        if (d) e.target.style.transitionDelay = d;
        e.target.classList.add('visible');
        reveal.unobserve(e.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    anim.forEach(function (el) { reveal.observe(el); });
  }

  /* ── count-up ────────────────────────────────────────── */
  var nums = document.querySelectorAll('[data-target]');

  function run(el) {
    var target = parseFloat(el.dataset.target);
    if (isNaN(target)) return;
    if (reduced) { el.textContent = target.toLocaleString('en-US'); return; }

    var dur = 1400, t0 = null;
    function tick(now) {
      if (t0 === null) t0 = now;
      var p = Math.min((now - t0) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(eased * target).toLocaleString('en-US');
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  var counter = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      run(e.target);
      counter.unobserve(e.target);
    });
  }, { threshold: 0.4 });
  nums.forEach(function (el) { counter.observe(el); });

  /* ── mobile sheet ────────────────────────────────────── */
  var burger = document.querySelector('.burger');
  var sheet = document.querySelector('.sheet');

  if (burger && sheet) {
    var close = function () {
      burger.setAttribute('aria-expanded', 'false');
      sheet.setAttribute('hidden', '');
      document.body.style.overflow = '';
    };

    burger.addEventListener('click', function () {
      if (burger.getAttribute('aria-expanded') === 'true') { close(); return; }
      burger.setAttribute('aria-expanded', 'true');
      sheet.removeAttribute('hidden');
      document.body.style.overflow = 'hidden';
    });

    sheet.querySelectorAll('a').forEach(function (a) { a.addEventListener('click', close); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
    window.addEventListener('resize', function () { if (window.innerWidth > 760) close(); });
  }
})();
