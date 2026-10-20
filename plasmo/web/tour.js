// Tutorial guiado: un foco animado recorre la app y algunos pasos esperan una acción real.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/**
 * Paso: { target, title, text, placement, before, action: { label, run }, extra, wide }
 *  - target: selector o función que devuelve el elemento (null = tarjeta centrada)
 *  - before: se ejecuta antes de mostrar el paso (abrir un panel, elegir una forma…)
 *  - action: botón principal que hace algo de verdad; al terminar pasa al siguiente paso
 *  - text / extra: HTML (o funciones que lo devuelven) para el cuerpo de la tarjeta
 */
export class Tour {
  constructor(steps, { onEnd } = {}) {
    this.steps = steps;
    this.onEnd = onEnd;
    this.i = -1;
    this.busy = false;
    this.root = document.createElement("div");
    this.root.className = "tour";
    this.root.innerHTML = `
      <div class="tour-block"></div>
      <div class="tour-spot"></div>
      <div class="tour-card" role="dialog" aria-live="polite">
        <div class="tour-bar"><i></i></div>
        <div class="tour-count"></div>
        <h3 class="tour-title"></h3>
        <div class="tour-text"></div>
        <div class="tour-extra"></div>
        <div class="tour-actions">
          <button class="ghost sm tour-skip">Saltar tutorial</button>
          <span class="tour-gap"></span>
          <button class="sm tour-prev">Atrás</button>
          <button class="primary sm tour-next">Siguiente</button>
        </div>
      </div>`;
    this.spot = this.root.querySelector(".tour-spot");
    this.card = this.root.querySelector(".tour-card");
    this.root.querySelector(".tour-skip").onclick = () => this.end(false);
    this.root.querySelector(".tour-prev").onclick = () => this.go(this.i - 1);
    this.root.querySelector(".tour-next").onclick = () => this.primary();
    this.onKey = (e) => {
      if (this.busy) return;
      if (e.key === "Escape") this.end(false);
      else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); this.primary(); }
      else if (e.key === "ArrowLeft" && this.i > 0) this.go(this.i - 1);
    };
    this.onMove = () => this.place();
  }

  start(from = 0) {
    document.body.appendChild(this.root);
    requestAnimationFrame(() => this.root.classList.add("on"));
    window.addEventListener("keydown", this.onKey, true);
    window.addEventListener("resize", this.onMove);
    document.addEventListener("scroll", this.onMove, true);
    return this.go(from);
  }

  element(step) {
    if (!step.target) return null;
    const el = typeof step.target === "function" ? step.target() : document.querySelector(step.target);
    return el && el.getClientRects().length ? el : null;
  }

  async go(i) {
    if (i < 0 || i >= this.steps.length || this.busy) return;
    const step = this.steps[i];
    this.i = i;
    this.skipFailed = false;
    this.card.classList.add("swap");
    try {
      await step.before?.();
    } catch {}
    const el = this.element(step);
    if (el) {
      el.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
      await sleep(320);
    }
    const n = this.steps.length;
    this.root.querySelector(".tour-bar i").style.width = `${((i + 1) / n) * 100}%`;
    this.root.querySelector(".tour-count").textContent = i === 0 ? "Tutorial" : `Paso ${i} de ${n - 1}`;
    this.root.querySelector(".tour-title").textContent = step.title;
    this.root.querySelector(".tour-text").innerHTML = typeof step.text === "function" ? step.text() : step.text || "";
    this.root.querySelector(".tour-extra").innerHTML = typeof step.extra === "function" ? step.extra() : step.extra || "";
    const next = this.root.querySelector(".tour-next");
    next.textContent = step.action?.label || (i === n - 1 ? "Terminar" : i === 0 ? "Empezar" : "Siguiente");
    next.classList.toggle("tour-do", !!step.action);
    this.root.querySelector(".tour-prev").style.visibility = i > 0 && i < n - 1 ? "visible" : "hidden";
    this.root.querySelector(".tour-skip").style.visibility = i < n - 1 ? "visible" : "hidden";
    this.card.classList.toggle("wide", !!step.wide);
    this.root.classList.toggle("centered", !el);
    this.target = el;
    this.place();
    this.card.classList.remove("swap");
    await step.after?.(this.card);
  }

  async primary() {
    const step = this.steps[this.i];
    if (this.busy) return;
    if (this.skipFailed) {
      this.skipFailed = false;
      return this.go(this.i + 1);
    }
    if (step?.action) {
      this.busy = true;
      this.root.classList.add("working");
      const next = this.root.querySelector(".tour-next");
      next.disabled = true;
      next.innerHTML = `<span class="spinner"></span> ${esc(step.action.busy || "Un momento…")}`;
      try {
        await step.action.run();
        this.busy = false;
        this.root.classList.remove("working");
        next.disabled = false;
        return this.go(this.i + 1);
      } catch (e) {
        // si la acción falla (p. ej. falta un motor), se explica y se deja seguir
        this.busy = false;
        this.root.classList.remove("working");
        next.disabled = false;
        next.textContent = "Saltar este paso";
        this.skipFailed = true;
        this.root.querySelector(".tour-extra").innerHTML = `<p class="tour-error">${esc(e.message || e)}</p>`;
        this.place();
        return;
      }
    }
    if (this.i >= this.steps.length - 1) return this.end(true);
    return this.go(this.i + 1);
  }

  place() {
    const pad = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const cw = this.card.offsetWidth || 360;
    const ch = this.card.offsetHeight || 220;
    if (!this.target) {
      this.card.dataset.place = "centro";
      Object.assign(this.spot.style, { left: `${vw / 2}px`, top: `${vh / 2}px`, width: "0px", height: "0px" });
      Object.assign(this.card.style, { left: `${(vw - cw) / 2}px`, top: `${Math.max(16, (vh - ch) / 2)}px` });
      return;
    }
    const r = this.target.getBoundingClientRect();
    const box = { x: r.left - pad, y: r.top - pad, w: r.width + pad * 2, h: r.height + pad * 2 };
    Object.assign(this.spot.style, { left: `${box.x}px`, top: `${box.y}px`, width: `${box.w}px`, height: `${box.h}px` });
    const step = this.steps[this.i] || {};
    const gap = 16;
    const fits = {
      right: box.x + box.w + gap + cw < vw - 12,
      left: box.x - gap - cw > 12,
      bottom: box.y + box.h + gap + ch < vh - 12,
      top: box.y - gap - ch > 12,
    };
    let place = step.placement && fits[step.placement] ? step.placement : ["right", "bottom", "left", "top"].find((p) => fits[p]) || "over";
    let x;
    let y;
    if (place === "right") { x = box.x + box.w + gap; y = box.y + box.h / 2 - ch / 2; }
    else if (place === "left") { x = box.x - gap - cw; y = box.y + box.h / 2 - ch / 2; }
    else if (place === "bottom") { x = box.x + box.w / 2 - cw / 2; y = box.y + box.h + gap; }
    else if (place === "top") { x = box.x + box.w / 2 - cw / 2; y = box.y - gap - ch; }
    else { x = vw - cw - 20; y = vh - ch - 20; }
    x = Math.min(Math.max(12, x), vw - cw - 12);
    y = Math.min(Math.max(12, y), vh - ch - 12);
    this.card.dataset.place = place;
    Object.assign(this.card.style, { left: `${x}px`, top: `${y}px` });
  }

  async end(completed) {
    window.removeEventListener("keydown", this.onKey, true);
    window.removeEventListener("resize", this.onMove);
    document.removeEventListener("scroll", this.onMove, true);
    this.root.classList.remove("on");
    await sleep(320);
    this.root.remove();
    this.onEnd?.(completed);
  }
}
