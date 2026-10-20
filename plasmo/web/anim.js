// Animaciones de Plasmo: arranque, impresión por capas, "pensando" e imagen que se revela.
const NS = "http://www.w3.org/2000/svg";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let logoText = null;
// Cada copia del logo lleva su prefijo: los degradados no chocan por id repetido
export async function logoSVG(prefix = "p") {
  if (!logoText) logoText = await (await fetch("/logo.svg")).text();
  return logoText.replaceAll('id="p-', `id="${prefix}-`).replaceAll("url(#p-", `url(#${prefix}-`);
}

// Perfil de la gota (de arriba hacia abajo), el mismo del logo
const PROFILE = [2.8, 6.6, 10.2, 13.4, 15.9, 17.4, 17.6, 15.9, 11.4];
// Degradado de calor de abajo (violeta) hacia arriba (amarillo)
const STOPS = [[0, [156, 72, 220]], [0.3, [240, 80, 122]], [0.65, [255, 138, 76]], [1, [255, 214, 102]]];

function heat(t) {
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * k)).join(",")})`;
    }
  }
  return "rgb(255,214,102)";
}

function radiusAt(h) {
  const u = (1 - h) * (PROFILE.length - 1);
  const i = Math.floor(u);
  const f = u - i;
  const a = PROFILE[i];
  const b = PROFILE[Math.min(i + 1, PROFILE.length - 1)];
  return (a + (b - a) * f) / 17.6;
}

const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};

// ------------------------------------------------------------ arranque
export async function splashReady() {
  const splash = document.getElementById("splash");
  if (!splash) return () => {};
  const started = performance.now();
  try {
    splash.querySelector(".splash-logo").innerHTML = await logoSVG("splash");
  } catch {}
  splash.classList.add("play");
  return async () => {
    await sleep(Math.max(0, (reducedMotion() ? 200 : 2100) - (performance.now() - started)));
    splash.classList.add("out");
    setTimeout(() => splash.remove(), 700);
  };
}

// ---------------------------------------------------- impresión por capas
export class PrinterArt {
  constructor(svg, layers = 24) {
    this.svg = svg;
    this.n = layers;
    this.done = -1;
    svg.innerHTML = "";
    svg.setAttribute("viewBox", "0 0 260 172");
    const defs = el("defs", {}, svg);
    defs.innerHTML = `
      <radialGradient id="pa-glow"><stop offset="0" stop-color="#fff6d6"/><stop offset=".35" stop-color="#ffb347" stop-opacity=".9"/><stop offset="1" stop-color="#ff7a45" stop-opacity="0"/></radialGradient>
      <radialGradient id="pa-bed"><stop offset="0" stop-color="#ff7a45" stop-opacity=".55"/><stop offset="1" stop-color="#ff7a45" stop-opacity="0"/></radialGradient>
      <linearGradient id="pa-metal" x1="0" x2="1"><stop offset="0" stop-color="#5b6577"/><stop offset=".5" stop-color="#c9d0dc"/><stop offset="1" stop-color="#4a5262"/></linearGradient>
      <linearGradient id="pa-hot" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff9a5a"/><stop offset="1" stop-color="#e5487a"/></linearGradient>
      <linearGradient id="pa-sheen" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <filter id="pa-blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2"/></filter>`;
    this.bedY = 152;
    this.pitch = 118 / layers;
    this.bedGlow = el("ellipse", { cx: 130, cy: this.bedY + 2, rx: 92, ry: 7, fill: "url(#pa-bed)", opacity: 0.15 }, svg);
    el("rect", { x: 28, y: this.bedY, width: 204, height: 5, rx: 2.5, fill: "#2a313d" }, svg);
    const ticks = el("g", { stroke: "#3a4352", "stroke-width": 1 }, svg);
    for (let x = 40; x <= 220; x += 15) el("line", { x1: x, y1: this.bedY + 1.5, x2: x, y2: this.bedY + 3.5 }, ticks);

    this.ghost = el("g", { class: "pa-ghost" }, svg);
    this.layersG = el("g", { class: "pa-layers" }, svg);
    this.layers = [];
    for (let k = 0; k < layers; k++) {
      const h = k / (layers - 1);
      const w = Math.max(7, 2 * 64 * radiusAt(h));
      const y = this.bedY - (k + 1) * this.pitch;
      const lh = this.pitch * 0.8;
      const attrs = { x: 130 - w / 2, y, width: w, height: lh, rx: lh / 2 };
      el("rect", { ...attrs, fill: "none" }, this.ghost);
      const r = el("rect", { ...attrs, fill: heat(h), class: "pa-layer" }, this.layersG);
      this.layers.push({ r, w, y });
    }
    this.sheen = el("rect", { x: 60, y: this.bedY - 130, width: 140, height: 26, fill: "url(#pa-sheen)", class: "pa-sheen", opacity: 0 }, svg);

    // capa en curso: crece al paso de la boquilla
    this.current = el("rect", { class: "pa-current", rx: 1.5 }, svg);
    // boquilla (punta en 0,0)
    this.head = el("g", { class: "pa-head" }, svg);
    el("rect", { x: -96, y: -49, width: 192, height: 3.5, rx: 1.75, fill: "#2f3644" }, this.head);
    this.sweep = el("g", { class: "pa-sweep" }, this.head);
    this.sweep.innerHTML = `
      <rect x="-12" y="-44" width="24" height="17" rx="3.5" fill="url(#pa-metal)"/>
      <path d="M-9 -40h18M-9 -36h18M-9 -32h18" stroke="#3f4757" stroke-width="1.1"/>
      <rect x="-9.5" y="-27" width="19" height="12" rx="2.5" fill="url(#pa-hot)"/>
      <path d="M-5.5 -15 L5.5 -15 L2 -3 L-2 -3 Z" fill="#dfe4ec"/>
      <circle cx="0" cy="-1" r="7.5" fill="url(#pa-glow)" filter="url(#pa-blur)" class="pa-tipglow"/>
      <circle cx="0" cy="-1.6" r="1.6" fill="#fffbe8"/>
      <g class="pa-sparks"><circle r="1" cx="0" cy="0"/><circle r="0.8" cx="0" cy="0"/><circle r="0.9" cx="0" cy="0"/></g>`;
    this.set(0, "corriendo");
  }

  set(pct, state = "corriendo") {
    const finished = state === "listo";
    const target = finished ? this.n - 1 : Math.min(this.n - 2, Math.floor((pct / 100) * this.n) - 1);
    while (this.done < target) {
      this.done++;
      const L = this.layers[this.done];
      L.r.classList.add("on");
      L.r.classList.remove("just");
      void L.r.getBBox();
      L.r.classList.add("just");
    }
    this.bedGlow.setAttribute("opacity", String(0.15 + 0.6 * Math.min(1, pct / 100)));
    const next = this.layers[Math.min(this.done + 1, this.n - 1)];
    if (!finished && next) {
      const lh = this.pitch * 0.8;
      Object.entries({ x: 130 - next.w / 2, y: next.y, width: next.w, height: lh, fill: heat((this.done + 1) / (this.n - 1)) }).forEach(([k, v]) =>
        this.current.setAttribute(k, v));
      this.svg.style.setProperty("--sweep", `${next.w / 2 - 2}px`);
      this.head.style.transform = `translate(130px, ${next.y - 0.5}px)`;
    }
    this.svg.classList.toggle("finished", finished);
    this.svg.classList.toggle("failed", state === "error");
  }
}

// ------------------------------------------------- imagen que se revela
export class NoiseArt {
  constructor(svg, size = 14) {
    this.svg = svg;
    svg.innerHTML = "";
    svg.setAttribute("viewBox", "0 0 260 172");
    const g = el("g", { transform: "translate(70 12)" }, svg);
    el("rect", { x: -6, y: -6, width: 132, height: 132, rx: 10, fill: "#151920", stroke: "#2a313d" }, g);
    const cell = 120 / size;
    this.cells = [];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x - size / 2 + 0.5) / (size / 2);
        const dy = (y - size / 2 + 0.5) / (size / 2);
        const d = Math.sqrt(dx * dx * 1.1 + (dy + 0.1) ** 2);
        const inside = d < 0.78;
        const target = inside ? heat(Math.max(0, Math.min(1, 0.95 - (dy + 1) / 2.1))) : "#f2efe9";
        const g0 = Math.round(60 + Math.random() * 120);
        const r = el("rect", { x: x * cell, y: y * cell, width: cell + 0.3, height: cell + 0.3, fill: `rgb(${g0},${g0},${g0 + 10})`, class: "na-cell" }, g);
        r.style.animationDelay = `${-Math.random() * 1.2}s`;
        this.cells.push({ r, target });
      }
    }
    this.order = this.cells.map((_, i) => i).sort(() => Math.random() - 0.5);
    this.settled = 0;
  }

  set(pct, state) {
    const want = state === "listo" ? this.cells.length : Math.floor((pct / 100) * this.cells.length * 0.96);
    while (this.settled < want) {
      const c = this.cells[this.order[this.settled++]];
      c.r.classList.add("settled");
      c.r.setAttribute("fill", c.target);
    }
    this.svg.classList.toggle("finished", state === "listo");
  }
}

// ---------------------------------------------------------- pensando
const THINK_MSGS = [
  "Mirando la foto…", "Buscando el sujeto principal…", "Midiendo proporciones…", "Pensando en la mejor forma…",
  "Calculando el tamaño…", "Revisando voladizos…", "Escribiendo consejos…",
];

export function thinking(container, messages = THINK_MSGS) {
  container.innerHTML = `<div class="thinking"><div class="orb"><b></b><i></i><i></i><i></i></div><span class="thinking-text">${messages[0]}</span></div>`;
  const text = container.querySelector(".thinking-text");
  let i = 0;
  const timer = setInterval(() => {
    i = (i + 1) % messages.length;
    text.classList.add("swap");
    setTimeout(() => { text.textContent = messages[i]; text.classList.remove("swap"); }, 220);
  }, 1900);
  return () => clearInterval(timer);
}

export function typingDots() {
  const s = document.createElement("span");
  s.className = "typing";
  s.innerHTML = "<i></i><i></i><i></i>";
  return s;
}

// Lluvia de capas de colores desde un punto (fin del tutorial)
export function celebrate(x = window.innerWidth / 2, y = window.innerHeight / 2, n = 46) {
  if (reducedMotion()) return;
  for (let k = 0; k < n; k++) {
    const p = document.createElement("i");
    p.className = "confetti";
    p.style.background = heat(Math.random());
    p.style.left = `${x}px`;
    p.style.top = `${y}px`;
    document.body.appendChild(p);
    const ang = Math.random() * Math.PI * 2;
    const dist = 120 + Math.random() * 220;
    const dx = Math.cos(ang) * dist;
    const dy = Math.sin(ang) * dist - 80;
    p.animate(
      [
        { transform: "translate(0,0) rotate(0deg)", opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) rotate(${Math.random() * 540}deg)`, opacity: 1, offset: 0.7 },
        { transform: `translate(${dx * 1.1}px, ${dy + 160}px) rotate(${Math.random() * 720}deg)`, opacity: 0 },
      ],
      { duration: 1300 + Math.random() * 700, easing: "cubic-bezier(.2,.7,.3,1)" }
    ).onfinish = () => p.remove();
  }
}

export function fmtClock(s) {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
