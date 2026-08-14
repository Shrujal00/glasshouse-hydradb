/* --------------------------------------------------------------------------
   GLASSHOUSE — HYPER-FRAME CAMERA ZOOM-IN ENGINE & TERMINAL
   -------------------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
  initLenisAndGSAP();
  initMouseEffects();
  initHyperFrameCanvasEngine();
  initTerminal();
});

/* --------------------------------------------------------------------------
   1. LENIS + GSAP SCROLLTRIGGER SYNCHRONIZATION
   -------------------------------------------------------------------------- */
let lenis;
function initLenisAndGSAP() {
  if (typeof Lenis === 'undefined') return;

  lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
    smoothTouch: false
  });

  if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);

    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    const heroTl = gsap.timeline({ defaults: { ease: 'power3.out', duration: 1 } });
    heroTl.from('.hero-anim', {
      y: 40,
      opacity: 0,
      stagger: 0.15,
      clearProps: 'all'
    });

    document.querySelectorAll('.scroll-reveal').forEach(sec => {
      const cards = sec.querySelectorAll('.gsap-card, .gsap-pill');
      const titles = sec.querySelectorAll('.head-title, .head-desc');

      if (titles.length) {
        gsap.from(titles, {
          scrollTrigger: {
            trigger: sec,
            start: 'top 85%',
            toggleActions: 'play none none reverse'
          },
          y: 35,
          opacity: 0,
          stagger: 0.1,
          duration: 0.8,
          ease: 'power3.out',
          clearProps: 'transform,opacity'
        });
      }

      if (cards.length) {
        gsap.from(cards, {
          scrollTrigger: {
            trigger: sec,
            start: 'top 75%',
            toggleActions: 'play none none reverse'
          },
          y: 45,
          opacity: 0,
          stagger: 0.1,
          duration: 0.8,
          ease: 'power3.out',
          clearProps: 'transform,opacity'
        });
      }
    });

    const card = document.querySelector('.engine-canvas-container');
    if (card) card.classList.add('active-glow');
  }

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = anchor.getAttribute('href');
      if (targetId && targetId !== '#') {
        const target = document.querySelector(targetId);
        if (target) {
          lenis.scrollTo(target, { offset: -70 });
        }
      }
    });
  });
}

/* --------------------------------------------------------------------------
   2. MOUSE SPOTLIGHT & MAGNETIC HOVER EFFECTS
   -------------------------------------------------------------------------- */
function initMouseEffects() {
  const spotlight = document.getElementById('mouse-spotlight');

  window.addEventListener('mousemove', (e) => {
    if (spotlight) {
      spotlight.style.left = e.clientX + 'px';
      spotlight.style.top = e.clientY + 'px';
    }
  });

  const magnetTargets = document.querySelectorAll('.magnet-target');
  magnetTargets.forEach(btn => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      btn.style.transform = `translate(${x * 0.25}px, ${y * 0.25}px)`;
    });

    btn.addEventListener('mouseleave', () => {
      btn.style.transform = 'translate(0px, 0px)';
    });
  });
}

/* --------------------------------------------------------------------------
   3. HYPER-FRAME CAMERA CONTROLLER & NODE ZOOM-IN CANVAS ENGINE
   -------------------------------------------------------------------------- */
function initLegacyHyperFrameCanvasEngine() {
  const canvas = document.getElementById('engine-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const answer = document.getElementById('engine-answer');
  const evidence = document.getElementById('evidence-card');
  const counter = document.getElementById('engine-counter');
  const caption = document.getElementById('engine-caption');
  const railSteps = document.querySelectorAll('.footer-phase');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let dpr = window.devicePixelRatio || 1;
  let width, height;

  function resize() {
    const rect = stage.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resize();
  window.addEventListener('resize', resize);

  const funnelX = width * 0.38;
  const funnelY = height * 0.5;

  // Geometric Nodes
  const nodes = [
    { id: 'funnel', x: funnelX, y: funnelY, r: 4.5, type: 'white' },
    { id: 'n1', x: funnelX + 55, y: funnelY - 55, r: 4, type: 'white' },
    { id: 'n2', x: funnelX + 120, y: funnelY - 90, r: 5, type: 'purple' },
    { id: 'n3', x: funnelX + 185, y: funnelY - 45, r: 4, type: 'white' },
    { id: 'n4', x: funnelX + 235, y: funnelY - 15, r: 3.5, type: 'white' },
    { id: 'target', x: funnelX + 255, y: funnelY + 25, r: 8.5, type: 'target' },
    { id: 'n6', x: funnelX + 215, y: funnelY + 75, r: 3.5, type: 'white' },
    { id: 'n7', x: funnelX + 140, y: funnelY + 115, r: 5, type: 'purple' },
    { id: 'n8', x: funnelX + 75, y: funnelY + 85, r: 3.5, type: 'white' },
    { id: 'n9', x: funnelX + 105, y: funnelY - 15, r: 4.5, type: 'white' },
    { id: 'n10', x: funnelX + 150, y: funnelY + 28, r: 5.5, type: 'gold' },
    { id: 'n11', x: funnelX + 195, y: funnelY, r: 3.5, type: 'white' }
  ];

  const edges = [
    ['funnel', 'n1'], ['funnel', 'n8'], ['funnel', 'n9'],
    ['n1', 'n2'], ['n2', 'n3'], ['n3', 'n4'], ['n4', 'target'],
    ['target', 'n6'], ['n6', 'n7'], ['n7', 'n8'],
    ['n9', 'n2'], ['n9', 'n10'], ['n10', 'target'], ['n10', 'n7'],
    ['n11', 'n3'], ['n11', 'target'], ['n11', 'n9']
  ];

  // Floating Document Particles
  const docParticles = [];
  for (let i = 0; i < 65; i++) {
    docParticles.push({
      x: Math.random() * (funnelX - 50),
      y: Math.random() * height,
      w: 8 + Math.random() * 6,
      h: 11 + Math.random() * 8,
      speedX: 0.6 + Math.random() * 0.8,
      opacity: 0.15 + Math.random() * 0.35,
      rot: (Math.random() - 0.5) * 0.4
    });
  }

  // Hyper-Frame Sub-Pixel Camera Controller Object
  const camera = {
    x: 0,
    y: 0,
    scale: 1.0,
    targetX: 0,
    targetY: 0,
    targetScale: 1.0
  };

  let frame = 0;
  const maxFrames = reducedMotion ? 1 : 320;
  let isFinished = false;
  let isPlaying = false;
  let lastPhase = -1;

  function updatePhase(phase) {
    if (phase === lastPhase) return;
    lastPhase = phase;
    const phaseData = [
      ['01 / COLLECTING', 'Collecting evidence', 'Documents arrive from nine connected systems'],
      ['02 / TRAVERSING', 'Following identity signals', 'Aliases converge across Slack, Jira, and Confluence'],
      ['03 / RESOLVED', 'Entity resolved with provenance', 'One answer, three hops, zero unsupported claims']
    ][phase];
    if (counter) counter.textContent = phaseData[0];
    if (caption) caption.textContent = phaseData[2];
    railSteps.forEach((step, index) => step.classList.toggle('active', index === phase));
    if (answer) answer.classList.toggle('show', phase === 2);
    if (evidence) evidence.classList.toggle('show', phase === 2);
  }

  window.playSingleCanvasSequence = function() {
    frame = 0;
    isFinished = false;
    isPlaying = true;
    lastPhase = -1;
    camera.targetScale = 1.0;
    camera.targetX = 0;
    camera.targetY = 0;
    updatePhase(0);
    if (answer) answer.classList.remove('show');
    if (evidence) evidence.classList.remove('show');
  };

  window.setCanvasProgress = function(progress) {
    isPlaying = false;
    frame = Math.round(Math.max(0, Math.min(1, progress)) * maxFrames);
    isFinished = frame >= maxFrames;
  };

  const replayBtn = document.getElementById('replay-canvas-btn');
  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      window.playSingleCanvasSequence();
    });
  }

  function render() {
    ctx.clearRect(0, 0, width, height);

    const currentFrame = isFinished ? maxFrames : frame;
    if (isPlaying && !isFinished) {
      frame += 1;
      if (frame >= maxFrames) isFinished = true;
    }
    const isTraversing = currentFrame > 70 && currentFrame < 225;
    const isResolved = currentFrame >= 225;
    updatePhase(isResolved ? 2 : isTraversing ? 1 : 0);

    // Hyper-Frame Camera Target Updates based on animation phase
    if (isResolved) {
      // Zoom camera smoothly into resolved target node cluster (1.45x zoom)
      const targetNode = nodes.find(n => n.id === 'target');
      camera.targetScale = 1.45;
      camera.targetX = (width / 2 - targetNode.x + 80) * 0.35;
      camera.targetY = (height / 2 - targetNode.y) * 0.35;
    } else if (isTraversing) {
      camera.targetScale = 1.15;
      camera.targetX = -20;
      camera.targetY = 0;
    } else {
      camera.targetScale = 1.0;
      camera.targetX = 0;
      camera.targetY = 0;
    }

    // Sub-pixel Hyper-Frame Lerp Interpolation (60 FPS smooth camera dynamics)
    camera.scale += (camera.targetScale - camera.scale) * 0.05;
    camera.x += (camera.targetX - camera.x) * 0.05;
    camera.y += (camera.targetY - camera.y) * 0.05;

    // Apply Camera Transform Matrix
    ctx.save();
    ctx.translate(width / 2, height / 2);
    ctx.scale(camera.scale, camera.scale);
    ctx.translate(-width / 2 + camera.x, -height / 2 + camera.y);

    // 1. Draw Stream Lines
    ctx.lineWidth = 0.7;
    for (let i = 0; i < 18; i++) {
      const startY = (height / 18) * i + 12;
      ctx.beginPath();
      ctx.moveTo(0, startY);
      ctx.quadraticCurveTo(funnelX * 0.65, startY, funnelX, funnelY);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.stroke();
    }

    // 2. Draw Floating Document Particles
    docParticles.forEach(p => {
      if (!isFinished) {
        p.x += p.speedX;
        if (p.x > funnelX * 0.35) {
          p.y += (funnelY - p.y) * 0.035;
        }
        if (p.x >= funnelX - 8) {
          p.x = 10;
          p.y = Math.random() * height;
        }
      }

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);

      ctx.fillStyle = `rgba(255, 255, 255, ${p.opacity * 0.4})`;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);

      ctx.strokeStyle = `rgba(255, 255, 255, ${p.opacity * 0.7})`;
      ctx.lineWidth = 0.8;
      ctx.strokeRect(-p.w / 2, -p.h / 2, p.w, p.h);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
      ctx.fillRect(-p.w / 3, -p.h / 3, p.w * 0.6, 1.2);
      ctx.fillRect(-p.w / 3, -p.h / 8, p.w * 0.4, 1.2);
      ctx.restore();
    });

    // 3. Draw Graph Edges & Active Path Traversal Highlight
    edges.forEach(([sId, tId]) => {
      const s = nodes.find(n => n.id === sId);
      const t = nodes.find(n => n.id === tId);

      if (s && t) {
        let isPathEdge = false;
        let progress = 0;

        if (isTraversing) {
          if (sId === 'funnel' && tId === 'n9' && currentFrame > 60) {
            isPathEdge = true;
            progress = Math.min(1, (currentFrame - 60) / 40);
          } else if (sId === 'n9' && tId === 'n10' && currentFrame > 100) {
            isPathEdge = true;
            progress = Math.min(1, (currentFrame - 100) / 40);
          } else if (sId === 'n10' && tId === 'target' && currentFrame > 140) {
            isPathEdge = true;
            progress = Math.min(1, (currentFrame - 140) / 40);
          }
        } else if (isResolved) {
          if ((sId === 'funnel' && tId === 'n9') || (sId === 'n9' && tId === 'n10') || (sId === 'n10' && tId === 'target')) {
            isPathEdge = true;
            progress = 1;
          }
        }

        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);

        if (isPathEdge) {
          ctx.strokeStyle = '#FF5C39';
          ctx.lineWidth = 2.4 / camera.scale;
          ctx.shadowColor = '#FF5C39';
          ctx.shadowBlur = 12;
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
          ctx.lineWidth = 0.9 / camera.scale;
          ctx.shadowBlur = 0;
        }

        ctx.stroke();
        ctx.shadowBlur = 0;

        // Active Traversal Spark
        if (isPathEdge && progress < 1) {
          const px = s.x + (t.x - s.x) * progress;
          const py = s.y + (t.y - s.y) * progress;

          ctx.beginPath();
          ctx.arc(px, py, 3.5 / camera.scale, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.shadowColor = '#FF5C39';
          ctx.shadowBlur = 12;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }
    });

    // 4. Draw Nodes with Expanding Halos on Zoom
    nodes.forEach(n => {
      let isActive = false;
      if (n.id === 'funnel' && currentFrame >= 60) isActive = true;
      if (n.id === 'n9' && currentFrame >= 100) isActive = true;
      if (n.id === 'n10' && currentFrame >= 140) isActive = true;
      if (n.id === 'target' && isResolved) isActive = true;

      // Glow Halos
      if (n.id === 'target' && isResolved) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * 2.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 92, 57, 0.25)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * 1.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 92, 57, 0.45)';
        ctx.fill();
      } else if (n.type === 'purple') {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * 2.2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(168, 85, 247, 0.2)';
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);

      if (n.id === 'target') {
        ctx.fillStyle = isResolved ? '#FF5C39' : '#EDEDF0';
        ctx.shadowColor = '#FF5C39';
        ctx.shadowBlur = isResolved ? 18 : 0;
      } else if (n.type === 'purple') {
        ctx.fillStyle = '#A855F7';
        ctx.shadowColor = '#A855F7';
        ctx.shadowBlur = 10;
      } else if (n.type === 'gold') {
        ctx.fillStyle = isActive ? '#F0C674' : '#EDEDF0';
        ctx.shadowColor = '#F0C674';
        ctx.shadowBlur = isActive ? 10 : 0;
      } else {
        ctx.fillStyle = isActive ? '#5FD3C4' : '#EDEDF0';
        ctx.shadowBlur = 0;
      }

      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // 5. Draw Target Resolved Glass Tooltip (Crisp scale-adjusted tooltip)
    if (isResolved) {
      const targetNode = nodes.find(n => n.id === 'target');
      const boxX = targetNode.x - 220;
      const boxY = targetNode.y + 18;

      ctx.fillStyle = 'rgba(12, 13, 18, 0.92)';
      ctx.strokeStyle = 'rgba(255, 92, 57, 0.6)';
      ctx.lineWidth = 1 / camera.scale;

      ctx.beginPath();
      ctx.roundRect(boxX, boxY, 210, 68, 8);
      ctx.fill();
      ctx.stroke();

      ctx.font = '700 10px "Fira Code", monospace';
      ctx.fillStyle = '#FF5C39';
      ctx.fillText('✓ ENTITY RESOLVED', boxX + 12, boxY + 18);

      ctx.font = '600 12px "Inter", sans-serif';
      ctx.fillStyle = '#EDEDF0';
      ctx.fillText('Priya Nair (Lead Engineer)', boxX + 12, boxY + 36);

      ctx.font = '500 10px "Fira Code", monospace';
      ctx.fillStyle = '#9CA3AF';
      ctx.fillText('Source: Confluence ADR #3491 (99.8%)', boxX + 12, boxY + 52);
    }

    ctx.restore(); // Restore camera matrix

    requestAnimationFrame(render);
  }

  render();
}

/* --------------------------------------------------------------------------
   3. HYPERFRAME STORY: NOISE -> GRAPH -> GLASSHOUSE
   -------------------------------------------------------------------------- */
function initHyperFrameCanvasEngine() {
  const canvas = document.getElementById('engine-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const answer = document.getElementById('engine-answer');
  const evidence = document.getElementById('evidence-card');
  const counter = document.getElementById('engine-counter');
  const caption = document.getElementById('engine-caption');
  const phases = document.querySelectorAll('.footer-phase');
  let width = 0, height = 0, progress = 0, playing = false, startedAt = 0;

  function resize() {
    const rect = stage.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  const docs = Array.from({ length: 28 }, (_, i) => ({
    x: 0.12 + Math.random() * 0.3,
    y: 0.2 + Math.random() * 0.6,
    rotation: (Math.random() - .5) * .45,
    tone: i % 4 === 0 ? '#FF5C39' : '#9CA3AF'
  }));

  function ease(t) { return t * t * (3 - 2 * t); }
  function label(text, x, y, color = '#6B7280', size = 10) {
    ctx.fillStyle = color;
    ctx.font = `${size}px "Fira Code", monospace`;
    ctx.fillText(text, x, y);
  }
  function updateUI(scene) {
    const data = [
      ['01 / UNSTRUCTURED', 'A single agent searches the mess — and invents an answer.', '01 Collect'],
      ['02 / HYDRADB', 'HydraDB turns every fragment into connected, queryable structure.', '02 Traverse'],
      ['03 / GLASSHOUSE', 'Glasshouse follows the evidence and returns only what is true.', '03 Resolve']
    ][scene];
    if (counter) counter.textContent = data[0];
    if (caption) caption.textContent = data[1];
    phases.forEach((phase, index) => phase.classList.toggle('active', index === scene));
    answer?.classList.toggle('show', scene === 2);
    evidence?.classList.toggle('show', scene === 2);
  }
  function drawDocument(x, y, rotation, color, scale = 1) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rotation);
    ctx.globalAlpha = .35;
    ctx.strokeStyle = color;
    ctx.fillStyle = 'rgba(255,255,255,.025)';
    ctx.lineWidth = 1;
    ctx.fillRect(-18 * scale, -24 * scale, 36 * scale, 48 * scale);
    ctx.strokeRect(-18 * scale, -24 * scale, 36 * scale, 48 * scale);
    ctx.globalAlpha = .5;
    ctx.fillRect(-11 * scale, -11 * scale, 19 * scale, 2 * scale);
    ctx.fillRect(-11 * scale, -3 * scale, 24 * scale, 2 * scale);
    ctx.fillRect(-11 * scale, 5 * scale, 14 * scale, 2 * scale);
    ctx.restore();
  }
  function draw(progressValue) {
    ctx.clearRect(0, 0, width, height);
    const scene = progressValue < .34 ? 0 : progressValue < .68 ? 1 : 2;
    updateUI(scene);
    const local = scene === 0 ? progressValue / .34 : scene === 1 ? (progressValue - .34) / .34 : (progressValue - .68) / .32;
    const t = ease(Math.max(0, Math.min(1, local)));
    const cx = width * .54, cy = height * .53;

    if (scene === 0) {
      docs.forEach((doc, i) => {
        const drift = t * (i % 2 ? 28 : -20);
        drawDocument(width * doc.x + drift, height * doc.y, doc.rotation, doc.tone);
      });
      ctx.strokeStyle = 'rgba(255,92,57,.5)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(width * .56, height * .3); ctx.lineTo(width * .72, height * .3); ctx.stroke();
      ctx.fillStyle = 'rgba(255,92,57,.08)'; ctx.fillRect(width * .67, height * .23, width * .25, height * .22);
      label('AGENT OUTPUT', width * .69, height * .28, '#FF5C39');
      label('Priya owns it. Due Friday.', width * .69, height * .35, '#F3F4F6', 13);
      label('X unsupported claim', width * .69, height * .42, '#FF5C39');
      label('scattered docs / conflicting names / missing context', width * .12, height * .88);
    } else if (scene === 1) {
      docs.forEach((doc, i) => {
        const startX = width * (.14 + (i % 3) * .12);
        const startY = height * (.25 + (i % 7) * .07);
        const x = startX + (cx - startX) * t;
        const y = startY + (cy - startY) * t;
        drawDocument(x, y, doc.rotation * (1 - t), doc.tone, 1 - t * .45);
      });
      ctx.strokeStyle = 'rgba(95,211,196,.5)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 7; i++) {
        const angle = i * Math.PI * 2 / 7;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(angle) * 100, cy + Math.sin(angle) * 100); ctx.stroke();
      }
      ctx.strokeStyle = '#5FD3C4'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, 38 + t * 12, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#5FD3C4'; ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill();
      label('HYDRADB', cx - 32, cy + 65, '#5FD3C4');
      label('entities', width * .15, height * .88, '#5FD3C4');
      label('relationships', width * .35, height * .88, '#5FD3C4');
      label('provenance', width * .61, height * .88, '#5FD3C4');
    } else {
      ctx.strokeStyle = 'rgba(95,211,196,.18)'; ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.arc(cx, cy, 60 + i * 38, 0, Math.PI * 2); ctx.stroke(); }
      ctx.fillStyle = 'rgba(255,92,57,.12)'; ctx.beginPath(); ctx.arc(cx, cy, 64 + t * 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#FF5C39'; ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.fill();
      label('GLASSHOUSE', cx - 40, cy - 94, '#FF5C39');
      label('verified answer', cx - 42, cy + 105, '#F3F4F6', 12);
      label('every claim carries its source', cx - 92, cy + 123, '#6B7280');
    }
  }
  function animate() {
    if (playing) {
      progress = Math.min(1, (performance.now() - startedAt) / 7200);
      if (progress >= 1) playing = false;
    }
    draw(progress);
    requestAnimationFrame(animate);
  }
  window.setCanvasProgress = value => { playing = false; progress = Math.max(0, Math.min(1, value)); };
  window.playSingleCanvasSequence = () => { progress = 0; startedAt = performance.now(); playing = true; };
  document.getElementById('replay-canvas-btn')?.addEventListener('click', window.playSingleCanvasSequence);
  window.playSingleCanvasSequence();
  animate();
}

/* --------------------------------------------------------------------------
   4. TERMINAL QUERY SANDBOX ENGINE
   -------------------------------------------------------------------------- */
const scenarios = [
  {
    q: "Who owns the billing API refactor and when is it due?",
    steps: [
      "Recalled 14 candidate document chunks across Slack, Confluence, and Jira.",
      "Collapsed '@sam_r', 'sam.ratn@redwood.com', and 'S. Ratnaparkhi' to Entity #4902.",
      "Arbitrated conflict: Confluence ADR #3491 (March 4) overrides Slack thread (March 1)."
    ],
    answer: "Sam Ratnaparkhi (@sam_r) owns the billing API refactor. Due date: March 24, 2026.",
    citation: "Confluence ADR #3491 (2026-03-04): 'Target release date confirmed for March 24 following security audit.'"
  },
  {
    q: "What is our SOC2 compliance scope according to Slack and Confluence?",
    steps: [
      "Recalled 8 chunks matching 'SOC2 compliance scope'.",
      "Executed Cypher path traversal connecting #security-compliance to ADRs.",
      "Linked 3 security policies directly to auditor sign-off artifacts."
    ],
    answer: "SOC2 Type II scope covers production AWS infrastructure, database encryption, and GitHub access logging.",
    citation: "Confluence Policy #104: 'Scope locked for 2026 audit: AWS US-East-1, HydraDB graph cluster, employee SSO access logs.'"
  },
  {
    q: "What is the projected revenue for Q4 2028?",
    steps: [
      "Searched 500,000 documents for 'Q4 2028 projected revenue'.",
      "Verified no financial model in corpus contains projections beyond Q4 2027.",
      "Honest abstention triggered."
    ],
    answer: "No financial projections exist for Q4 2028. Latest available projections extend through Q4 2027 ($45M ARR target).",
    citation: "Zero-Hallucination Policy: 'Glasshouse explicitly surfaces unanswerability when evidence is absent from the corpus.'"
  }
];

function initTerminal() {
  const chips = document.querySelectorAll('.chip');
  const input = document.getElementById('term-input');
  const btn = document.getElementById('term-run');
  const streamEl = document.getElementById('term-stream');
  const answerEl = document.getElementById('term-answer');

  function runScenario(idx) {
    const data = scenarios[idx] || scenarios[0];
    input.value = data.q;
    streamEl.innerHTML = '';
    answerEl.innerHTML = '';

    data.steps.forEach((step, i) => {
      setTimeout(() => {
        const div = document.createElement('div');
        div.className = 'term-step';
        div.innerHTML = `<span>[0${i+1}]</span> ${step}`;
        streamEl.appendChild(div);
      }, i * 150);
    });

    setTimeout(() => {
      answerEl.innerHTML = `
        <p><strong>Answer:</strong> ${data.answer}</p>
        <div class="cite-box">
          <strong>PROVENANCE:</strong> ${data.citation}
        </div>
      `;
    }, data.steps.length * 150 + 100);
  }

  chips.forEach((chip, idx) => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      runScenario(idx);
    });
  });

  if (btn) {
    btn.addEventListener('click', () => runScenario(0));
  }

  runScenario(0);
}
