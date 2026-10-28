import { Viewer } from "/viewer.js";
import { icon, hydrateIcons } from "/icons.js";
import { NoiseArt, PrinterArt, celebrate, fmtClock, logoSVG, reducedMotion, splashReady, thinking, typingDots } from "/anim.js";
import { Tour } from "/tour.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  config: null,
  forms: [],
  project: null,
  form: null,
  values: {},
  version: null,
  job: null,
  chat: [],
  imgView: "recorte",
  status: null,
};
let viewer = null;

// ------------------------------------------------------------------ API
async function api(path, opts = {}) {
  const headers = { "X-Plasmo": "1", ...(opts.headers || {}) };
  let body = opts.body;
  if (body && !(body instanceof FormData) && typeof body !== "string") {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  const res = await fetch(path, { ...opts, headers, body });
  if (!res.ok) {
    let msg = `Error ${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res;
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toast").appendChild(el);
  setTimeout(() => {
    el.classList.add("bye");
    setTimeout(() => el.remove(), 320);
  }, kind === "err" ? 7000 : 3500);
}

const fileUrl = (pid, name, download = false) =>
  `/api/proyectos/${encodeURIComponent(pid)}/archivo/${name.split("/").map(encodeURIComponent).join("/")}${download ? "?descargar=true" : `?t=${Date.now()}`}`;

// ------------------------------------------------------------ trabajos
function watchJob(job, { fixed = false } = {}) {
  return new Promise((resolve, reject) => {
    state.job = job;
    const box = $("#progress");
    box.classList.toggle("fixed", fixed);
    if (fixed) { box.style.position = "fixed"; box.style.zIndex = 40; document.body.appendChild(box); }
    else if (box.parentElement !== $(".viewer-card")) { box.style.position = ""; box.style.zIndex = ""; $(".viewer-card").appendChild(box); }
    box.classList.remove("hidden");
    // texto→imagen se "revela" como una difusión; lo demás se imprime capa por capa
    const art = job.tipo === "texto" ? new NoiseArt($("#pg-art")) : new PrinterArt($("#pg-art"), 24);
    const list = $("#pg-phases");
    list.innerHTML = job.fases.map((f) => `<li data-k="${esc(f.clave)}"><i></i><span>${esc(f.nombre)}</span><span class="pp"></span></li>`).join("");
    const started = Date.now();
    const clock = () => { $("#pg-elapsed").textContent = `Tiempo ${fmtClock((Date.now() - started) / 1000)}`; };
    clock();
    const tick = setInterval(clock, 500);
    const render = (j) => {
      const pct = j.porcentaje ?? 0;
      $("#pg-pct").textContent = `${Math.round(pct)} %`;
      $("#pg-bar").style.width = `${pct}%`;
      $("#pg-msg").textContent = j.mensaje || "";
      $("#pg-eta").textContent = j.eta_s ? `≈ ${fmtTime(j.eta_s)} restantes` : "";
      for (const f of j.fases) {
        const li = list.querySelector(`li[data-k="${CSS.escape(f.clave)}"]`);
        if (!li) continue;
        if (li.className !== f.estado) {
          li.className = f.estado;
          li.querySelector("i").textContent = f.estado === "lista" ? "✓" : "";
        }
        li.querySelector(".pp").textContent = f.estado === "activa" && f.progreso > 0 && f.progreso < 1 ? `${Math.round(f.progreso * 100)} %` : "";
      }
      art.set(pct, j.estado);
    };
    render(job);
    const close = (ok) => {
      clearInterval(tick);
      state.job = null;
      // deja ver el final de la animación antes de cerrar
      setTimeout(() => box.classList.add("hidden"), ok ? (reducedMotion() ? 150 : 1150) : 800);
    };
    const timer = setInterval(async () => {
      try {
        const j = await api(`/api/trabajos/${job.id}`);
        render(j);
        if (["listo", "error", "cancelado"].includes(j.estado)) {
          clearInterval(timer);
          close(j.estado === "listo");
          if (j.estado === "listo") setTimeout(() => resolve(j.resultado || {}), reducedMotion() ? 0 : 950);
          else reject(new Error(j.estado === "cancelado" ? "Cancelado" : j.error || "Falló"));
        }
      } catch (e) {
        clearInterval(timer);
        close(false);
        reject(e);
      }
    }, 400);
  });
}

function fmtTime(s) {
  s = Math.round(s);
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${s % 60 ? (s % 60) + " s" : ""}`;
}

// ------------------------------------------------------------- estado
async function refreshStatus() {
  if (state.starting) return; // las píldoras muestran el arranque en curso
  try {
    const st = await api("/api/estado");
    state.status = st;
    const o = st.ollama;
    const pillO = $("#pill-ollama");
    const okModel = o.ok && o.modelo_listo;
    pillO.querySelector(".dot").className = `dot ${okModel ? "ok" : o.ok ? "warn" : "err"}`;
    pillO.querySelector(".txt").textContent = !o.ok ? "Ollama apagado" : okModel ? `Ollama · ${o.modelo_activo}` : `Falta ${o.modelo_activo}`;
    pillO.title = !o.ok ? `No responde en ${o.url}` : okModel ? `Ollama ${o.version}` : "Descárgalo en Ajustes";
    const g = st.generador;
    const pillG = $("#pill-gen");
    pillG.querySelector(".dot").className = `dot ${g.ok ? "ok" : "err"}`;
    pillG.querySelector(".txt").textContent = g.ok ? `${g.nombre} · ${g.gpu ? g.gpu.replace(/^cuda:\d+\s*/, "").replace("NVIDIA GeForce ", "") : "listo"}` : `${g.nombre} sin conexión`;
    pillG.title = g.mensaje || "";
    renderPower();
  } catch {}
}

// ------------------------------------------------------ encender motores
const ENGINE_NAMES = { ollama: "Ollama", comfyui: "ComfyUI" };
const ENGINE_PILL = { ollama: "#pill-ollama", comfyui: "#pill-gen" };

function enginesOff(st = state.status) {
  if (!st) return [];
  const off = [];
  if (!st.ollama.ok) off.push("ollama");
  if (st.generador.tipo === "comfyui" && !st.generador.responde) off.push("comfyui");
  return off;
}

function renderPower() {
  if (state.starting) return;
  const off = enginesOff();
  const btn = $("#btn-power");
  btn.classList.toggle("hidden", !off.length);
  btn.querySelector(".power-label").textContent = off.length > 1 ? "Encender IA" : off.length ? `Encender ${ENGINE_NAMES[off[0]]}` : "";
  btn.title = off.length ? `Arranca ${off.map((n) => ENGINE_NAMES[n]).join(" y ")} sin abrir ventanas` : "";
}

async function startEngines(names = enginesOff()) {
  if (!names.length || state.starting) return false;
  const btn = $("#btn-power");
  const label = btn.querySelector(".power-label");
  state.starting = true;
  btn.classList.remove("hidden");
  btn.classList.add("busy");
  btn.disabled = true;
  label.textContent = "Encendiendo…";
  for (const n of names) {
    $(`${ENGINE_PILL[n]} .dot`).className = "dot warn pulse";
    $(`${ENGINE_PILL[n]} .txt`).textContent = `${ENGINE_NAMES[n]}: en cola…`;
  }
  let ok = false;
  try {
    const job = await api("/api/motores/encender", { method: "POST", body: { motores: names } });
    let j = job;
    while (!["listo", "error", "cancelado"].includes(j.estado)) {
      await new Promise((r) => setTimeout(r, 500));
      j = await api(`/api/trabajos/${job.id}`);
      label.textContent = `Encendiendo… ${Math.round(j.porcentaje)} %`;
      btn.title = j.mensaje || "";
      for (const f of j.fases) {
        const txt = $(`${ENGINE_PILL[f.clave]} .txt`);
        if (f.estado === "activa") txt.textContent = f.detalle || `${ENGINE_NAMES[f.clave]}: arrancando…`;
        else if (f.estado === "lista") txt.textContent = `${ENGINE_NAMES[f.clave]}: listo`;
      }
    }
    if (j.estado === "error") throw new Error(j.error);
    const res = j.resultado?.resultados || {};
    const bad = Object.entries(res).filter(([, r]) => r.estado === "error");
    const good = Object.entries(res).filter(([, r]) => r.estado !== "error").map(([n]) => ENGINE_NAMES[n]);
    if (good.length) toast(`${good.join(" y ")} ${good.length > 1 ? "encendidos" : "encendido"}`, "ok");
    bad.forEach(([, r]) => toast(r.mensaje.split("\n")[0], "err"));
    ok = !bad.length;
  } catch (e) {
    toast(`${e.message.split("\n")[0]} · Revisa Ajustes → Motores.`, "err");
  } finally {
    state.starting = false;
    btn.classList.remove("busy");
    btn.disabled = false;
    await refreshStatus();
  }
  return ok;
}

// ----------------------------------------------------------- proyectos
async function loadProjects() {
  const { proyectos } = await api("/api/proyectos");
  const list = $("#project-list");
  if (!proyectos.length) {
    list.innerHTML = `<div class="empty-list">Aún no hay proyectos.</div>`;
    return;
  }
  list.innerHTML = proyectos.map((p, i) => `
    <div class="project ${state.project?.id === p.id ? "active" : ""}" data-id="${esc(p.id)}" style="--k:${Math.min(i, 12)}">
      <img src="${fileUrl(p.id, p.miniatura)}" alt="" loading="lazy">
      <div style="min-width:0"><div class="p-name">${esc(p.nombre || p.id)}</div>
      <div class="p-meta">${p.forma ? esc(formName(p.forma)) : "Sin pieza"} · ${new Date((p.creado || 0) * 1000).toLocaleDateString("es-DO", { day: "numeric", month: "short" })}</div></div>
    </div>`).join("");
  $$(".project", list).forEach((el) => el.addEventListener("click", () => openProject(el.dataset.id)));
}

const formName = (key) => state.forms.find((f) => f.clave === key)?.nombre || key;

async function createProject(file) {
  if (!file || !file.type.startsWith("image/")) return toast("Eso no parece una imagen.", "err");
  const fd = new FormData();
  fd.append("imagen", file, file.name || "pegada.png");
  fd.append("recorte", state.config?.bg_model || "");
  showWorkspace();
  const local = URL.createObjectURL(file);
  $("#img-preview").src = local;
  setCutting(true, "Separando el sujeto del fondo…");
  renderAnalysis(null);
  try {
    const meta = await api("/api/proyectos", { method: "POST", body: fd });
    if (meta.aviso_recorte) toast(meta.aviso_recorte);
    await loadProjects();
    setCutting(false);
    await openProject(meta.id, { reveal: true });
    if (state.status?.ollama?.modelo_listo) analyze();
  } catch (e) {
    setCutting(false);
    toast(e.message, "err");
    showWelcome();
  } finally {
    setTimeout(() => URL.revokeObjectURL(local), 4000);
  }
}

let cutTimer = null;
function setCutting(on, label = "") {
  const box = $(".image-box");
  box.classList.toggle("cutting", on);
  box.querySelector(".box-label")?.remove();
  clearInterval(cutTimer);
  if (!on) return;
  box.insertAdjacentHTML("beforeend", `<div class="box-label"><span class="spinner"></span><span>${esc(label)}</span><span class="muted"></span></div>`);
  const started = Date.now();
  // el recorte de máxima calidad tarda: se muestran los segundos para que no parezca colgado
  cutTimer = setInterval(() => {
    const el = box.querySelector(".box-label .muted");
    if (el) el.textContent = `${Math.round((Date.now() - started) / 1000)} s`;
  }, 1000);
}

const CUT_NAMES = { auto: "máxima calidad", "isnet-general-use": "IS-Net", "birefnet-lite": "BiRefNet", u2netp: "U2-Net", simple: "por color", ninguno: "imagen completa" };

function renderCutQuality() {
  const box = $("#cut-quality");
  const r = state.project?.recorte;
  const a = state.project?.analisis;
  if (!r || r.nota == null) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const objetos = r.objetos > 1 ? ` · ${r.objetos} objetos` : "";
  const alert = a && a.recorte_completo === false
    ? `<div class="q-alert"><b>La IA dice que al recorte le falta algo:</b> ${esc(a.recorte_comentario || "revisa el recorte")}
        ${r.modelo !== "auto" ? `<div style="margin-top:6px"><button class="sm" id="q-max">Recortar con máxima calidad</button></div>` : ""}</div>`
    : "";
  box.innerHTML = `
    <div class="q-head q-${esc(r.calidad)}"><span>Recorte <b>${esc(r.calidad)}</b> · ${esc(CUT_NAMES[r.modelo] || r.modelo)}${objetos}</span>
      <span class="q-meter"><i style="width:${Math.max(4, r.nota)}%"></i></span><span class="muted">${r.nota}/100</span></div>
    ${(r.avisos || []).length ? `<ul>${r.avisos.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}${alert}`;
  $("#q-max")?.addEventListener("click", () => { $("#cut-model").value = "auto"; $("#btn-recut").click(); });
}

function showWorkspace() {
  $("#welcome").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  if (!viewer) {
    viewer = new Viewer($("#viewer"));
    const p = state.config?.printer;
    if (p) viewer.setBed(p.bed_x, p.bed_y);
  }
}

function showWelcome() {
  state.project = null;
  history.replaceState(null, "", location.pathname);
  $("#workspace").classList.add("hidden");
  $("#welcome").classList.remove("hidden");
  $$(".project").forEach((el) => el.classList.remove("active"));
}

async function openProject(pid, { reveal = false } = {}) {
  const meta = await api(`/api/proyectos/${encodeURIComponent(pid)}`);
  state.project = meta;
  history.replaceState(null, "", `#p=${encodeURIComponent(pid)}`);
  showWorkspace();
  $$(".project").forEach((el) => el.classList.toggle("active", el.dataset.id === pid));
  renderImage(reveal);
  if (meta.recorte?.modelo) $("#cut-model").value = meta.recorte.modelo;
  $("#cut-objects").value = meta.recorte?.objetos_modo === "principal" ? "principal" : "todos";
  renderAnalysis(meta.analisis);
  renderCutQuality();
  const versions = meta.versiones || [];
  const last = versions[versions.length - 1];
  selectForm(last?.forma || meta.analisis?.forma_recomendada || state.form || "relieve", last);
  renderVersions();
  if (last) showVersion(last);
  else {
    viewer.clear();
    $("#viewer-empty").classList.remove("hidden");
    $("#card-report").classList.add("hidden");
    $("#viewer-info").textContent = "";
  }
}

function renderImage(reveal = false) {
  const p = state.project;
  if (!p) return;
  const name = state.imgView === "recorte" && p.recorte ? "recorte.png" : "original.png";
  const img = $("#img-preview");
  img.classList.remove("reveal");
  img.onload = reveal ? () => { void img.offsetWidth; img.classList.add("reveal"); } : null;
  img.src = fileUrl(p.id, name);
  $$("#img-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.v === state.imgView));
}

// ------------------------------------------------------------- análisis
const PROBLEMS = {
  varios_objetos: "Varios objetos", fondo_recargado: "Fondo recargado", angulo_malo: "Ángulo difícil", desenfocada: "Desenfocada",
  poca_luz: "Poca luz", objeto_cortado: "Objeto cortado", transparente_o_reflectante: "Transparente o brillante",
};

function renderAnalysis(a) {
  const body = $("#analysis-body");
  if (!a) {
    body.className = "muted small";
    body.innerHTML = "Pídele a Ollama que mire la foto: te dirá qué forma conviene, el tamaño y consejos de impresión.";
    return;
  }
  body.className = "";
  body.innerHTML = `
    <p>${esc(a.descripcion)}</p>
    <div class="chips">
      <span class="chip accent">Sugerida: ${esc(formName(a.forma_recomendada))}</span>
      <span class="chip">${Math.round(a.tamano_mm)} mm</span>
      <span class="chip">Soportes: ${esc(a.soportes)}</span>
      <span class="chip">${esc(a.material)}</span>
      ${a.problemas.map((p) => `<span class="chip warn">${esc(PROBLEMS[p] || p)}</span>`).join("")}
    </div>
    ${(a.objetos || []).length ? `<div class="chips"><span class="chip">Ve:</span>${a.objetos.map((o) => `<span class="chip">${esc(o)}</span>`).join("")}</div>` : ""}
    ${a.por_que ? `<p class="muted small">${esc(a.por_que)}</p>` : ""}
    ${a.consejos.length ? `<ul class="tips">${a.consejos.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}`;
  renderForms();
}

async function analyze() {
  if (!state.project) return;
  const btn = $("#btn-analyze");
  btn.disabled = true;
  const body = $("#analysis-body");
  body.className = "";
  const stop = thinking(body);
  $(".image-box").classList.add("scanning");
  try {
    const job = await api(`/api/proyectos/${encodeURIComponent(state.project.id)}/analizar`, { method: "POST" });
    const res = await pollQuiet(job.id);
    state.project.analisis = res.analisis;
    renderAnalysis(res.analisis);
    renderCutQuality();
    if (!(state.project.versiones || []).length) {
      selectForm(res.analisis.forma_recomendada);
      applyRecommendedSize(res.analisis);
    }
  } catch (e) {
    stop();
    renderAnalysis(state.project.analisis);
    $("#analysis-body").insertAdjacentHTML("afterbegin", `<p class="small" style="color:var(--err)">${esc(e.message)}</p>`);
  } finally {
    stop();
    $(".image-box").classList.remove("scanning");
    btn.disabled = false;
  }
}

async function pollQuiet(jid) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 600));
    const j = await api(`/api/trabajos/${jid}`);
    if (j.estado === "listo") return j.resultado;
    if (j.estado === "error" || j.estado === "cancelado") throw new Error(j.error || j.estado);
  }
}

function applyRecommendedSize(a) {
  const v = state.values[state.form];
  if (!v || !a?.tamano_mm) return;
  if ("tamano_mm" in v.parametros) v.parametros.tamano_mm = Math.round(a.tamano_mm);
  if ("tamano_mm" in v.impresion) v.impresion.tamano_mm = Math.round(a.tamano_mm);
  renderFields();
}

// --------------------------------------------------------------- formas
function defaults(list) {
  return Object.fromEntries((list || []).map((f) => [f.clave, f.defecto]));
}

function selectForm(key, version) {
  const form = state.forms.find((f) => f.clave === key) || state.forms[1];
  state.form = form.clave;
  if (!state.values[form.clave]) state.values[form.clave] = { parametros: defaults(form.parametros), impresion: defaults(form.impresion) };
  if (version && version.forma === form.clave) {
    Object.assign(state.values[form.clave].parametros, version.parametros || {});
    Object.assign(state.values[form.clave].impresion, version.impresion || {});
  }
  renderForms();
  renderFields();
}

function renderForms() {
  const rec = state.project?.analisis?.forma_recomendada;
  $("#forms").innerHTML = state.forms.map((f) => `
    <button class="form-card ${f.clave === state.form ? "on" : ""}" data-k="${f.clave}">
      ${icon(f.clave)}<b>${esc(f.nombre)}</b>${f.clave === rec ? `<span class="badge">sugerida</span>` : ""}
    </button>`).join("");
  $$("#forms .form-card").forEach((b) => b.addEventListener("click", () => selectForm(b.dataset.k)));
  const form = state.forms.find((f) => f.clave === state.form);
  $("#form-desc").textContent = form?.descripcion || "";
  const needsGen = form?.requiere === "generador";
  const genOk = state.status?.generador?.ok;
  $("#btn-generate").title = needsGen && !genOk ? state.status?.generador?.mensaje || "" : "";
}

function fieldHtml(f, value, group) {
  const id = `f-${group}-${f.clave}`;
  if (f.tipo === "si_no") {
    return `<label class="switch field" title="${esc(f.ayuda || "")}"><input type="checkbox" id="${id}" data-g="${group}" data-k="${f.clave}" ${value ? "checked" : ""}>${esc(f.nombre)}</label>`;
  }
  if (f.tipo === "opcion") {
    return `<div class="field"><label for="${id}" title="${esc(f.ayuda || "")}">${esc(f.nombre)}</label>
      <select id="${id}" data-g="${group}" data-k="${f.clave}">${f.opciones.map(([v, l]) => `<option value="${esc(v)}" ${String(v) === String(value) ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></div>`;
  }
  return `<div class="field"><label for="${id}" title="${esc(f.ayuda || "")}">${esc(f.nombre)} ${f.unidad ? `<span class="unit">(${esc(f.unidad)})</span>` : ""}</label>
    <input type="number" id="${id}" data-g="${group}" data-k="${f.clave}" value="${value}" min="${f.min}" max="${f.max}" step="${f.paso}" title="${esc(f.ayuda || "")}"></div>`;
}

function renderFields() {
  const form = state.forms.find((f) => f.clave === state.form);
  const v = state.values[state.form];
  $("#fields").innerHTML = form.parametros.map((f) => fieldHtml(f, v.parametros[f.clave], "parametros")).join("");
  $("#print-fields").innerHTML = form.impresion.map((f) => fieldHtml(f, v.impresion[f.clave], "impresion")).join("");
  $$("#fields [data-k], #print-fields [data-k]").forEach((el) => el.addEventListener("change", () => {
    const def = [...form.parametros, ...form.impresion].find((f) => f.clave === el.dataset.k);
    let val = el.type === "checkbox" ? el.checked : el.value;
    if (def?.tipo === "numero") val = Math.min(def.max, Math.max(def.min, Number(val)));
    else if (def?.tipo === "opcion" && typeof def.defecto === "number") val = Number(val);
    state.values[state.form][el.dataset.g][el.dataset.k] = val;
    if (el.type === "number") el.value = val;
  }));
}

async function generate() {
  if (!state.project) return;
  const form = state.forms.find((f) => f.clave === state.form);
  if (form.requiere === "generador" && !state.status?.generador?.ok) {
    await refreshStatus();
    if (!state.status?.generador?.ok) return toast(state.status?.generador?.mensaje || "El generador 3D no está disponible.", "err");
  }
  const v = state.values[state.form];
  const btn = $("#btn-generate");
  btn.disabled = true;
  btn.classList.add("busy");
  try {
    const job = await api(`/api/proyectos/${encodeURIComponent(state.project.id)}/generar`, { method: "POST", body: { forma: state.form, ...v } });
    const res = await watchJob(job);
    state.project = await api(`/api/proyectos/${encodeURIComponent(state.project.id)}`);
    renderVersions();
    await showVersion(res);
    loadProjects();
    toast(`${res.forma_nombre} lista`, "ok");
    return true;
  } catch (e) {
    if (e.message !== "Cancelado") toast(e.message, "err");
    return false;
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
  }
}

// ------------------------------------------------------- versiones/reporte
function renderVersions() {
  const versions = state.project?.versiones || [];
  $("#versions").innerHTML = versions.map((v) => `<button class="${state.version?.version === v.version ? "on" : ""}" data-v="${esc(v.version)}" title="${esc(v.forma_nombre)}">${esc(v.version.split("-")[0])} · ${esc(v.forma_nombre)}</button>`).join("");
  $$("#versions button").forEach((b) => b.addEventListener("click", () => {
    const v = versions.find((x) => x.version === b.dataset.v);
    if (v) { showVersion(v); selectForm(v.forma, v); }
  }));
}

async function showVersion(v) {
  state.version = v;
  renderVersions();
  $("#viewer-empty").classList.add("hidden");
  try {
    const size = await viewer.load(fileUrl(state.project.id, `${v.version}/${v.archivos.glb}`));
    $("#viewer-info").textContent = `${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} mm · ${v.reporte.metricas.caras.toLocaleString("es-DO")} caras`;
  } catch (e) {
    toast(`No pude mostrar el modelo: ${e.message}`, "err");
  }
  renderReport(v);
}

function renderReport(v) {
  const r = v.reporte;
  const m = r.metricas;
  const card = $("#card-report");
  card.classList.remove("hidden");
  const label = { sana: "Lista para imprimir", aviso: "Imprimible con avisos", error: "Necesita arreglo" }[r.estado] || r.estado;
  const pid = state.project.id;
  const f = v.archivos;
  const dl = (key, text) => f[key] ? `<a href="${fileUrl(pid, `${v.version}/${f[key]}`, true)}" download><button class="sm" data-icon="download">${text}</button></a>` : "";
  card.innerHTML = `
    <div class="report-head">
      <h2 style="margin:0">Imprimibilidad</h2>
      <span class="status-badge ${r.estado}">${esc(label)}</span>
      <span class="muted small">${esc(v.forma_nombre)} · ${r.dimensiones_mm.join(" × ")} mm</span>
    </div>
    <div class="metrics">
      <div class="metric"><span>Cerrada</span><b class="${m.cerrada ? "ok" : "bad"}">${m.cerrada ? "Sí" : "No"}</b></div>
      <div class="metric"><span>Aristas no manifold</span><b class="${m.aristas_no_manifold ? "bad" : "ok"}">${m.aristas_no_manifold}</b></div>
      <div class="metric"><span>Agujeros</span><b class="${m.agujeros ? "bad" : "ok"}">${m.agujeros}</b></div>
      <div class="metric"><span>Volumen</span><b>${m.volumen_cm3} cm³</b></div>
      <div class="metric"><span>Peso (${esc(r.material)})</span><b>≈ ${r.peso_g} g</b></div>
      <div class="metric"><span>Voladizos</span><b>${Math.round(r.voladizos * 100)} %</b></div>
      <div class="metric"><span>Paredes &lt; ${r.pared_minima_mm} mm</span><b>${r.paredes_finas == null ? "—" : Math.round(r.paredes_finas * 100) + " %"}</b></div>
      <div class="metric"><span>Estable</span><b class="${r.estabilidad.estable ? "ok" : "bad"}">${r.estabilidad.estable ? "Sí" : "No"}</b></div>
      <div class="metric"><span>Cabe en la cama</span><b class="${r.cabe_en_cama ? "ok" : "bad"}">${r.cabe_en_cama ? "Sí" : "No"}</b></div>
      <div class="metric"><span>Piezas</span><b>${v.piezas.length}</b></div>
    </div>
    <ul class="issues">
      ${r.errores.map((t) => `<li class="error">${esc(t)}</li>`).join("")}
      ${r.avisos.map((t) => `<li class="aviso">${esc(t)}</li>`).join("")}
      ${(v.notas || []).map((t) => `<li class="nota">${esc(t)}</li>`).join("")}
      ${r.nota_paredes ? `<li class="nota">${esc(r.nota_paredes)}</li>` : ""}
    </ul>
    <div class="exports">
      ${dl("3mf", "3MF")}${dl("stl", "STL")}${dl("obj", "OBJ")}${dl("obj_color", "OBJ a color")}${dl("zip", "Piezas (ZIP)")}
      <button class="sm" id="open-slicer" data-icon="slicer">Abrir en el slicer</button>
      <button class="sm" id="open-folder" data-icon="folder">Abrir carpeta</button>
    </div>`;
  hydrateIcons(card);
  $$(".metric", card).forEach((m, i) => m.style.setProperty("--k", i));
  $$(".issues li", card).forEach((m, i) => m.style.setProperty("--k", i));
  $("#open-folder").onclick = () => api(`/api/proyectos/${encodeURIComponent(pid)}/abrir`, { method: "POST", body: { version: v.version } })
    .catch((e) => toast(e.message, "err"));
  $("#open-slicer").onclick = () => api(`/api/proyectos/${encodeURIComponent(pid)}/abrir`, { method: "POST", body: { archivo: `${v.version}/${f["3mf"]}` } })
    .then(() => toast("Abriendo el 3MF con tu slicer…")).catch((e) => toast(e.message, "err"));
}

// ----------------------------------------------------------------- chat
function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  $("#chat-log").appendChild(el);
  $("#chat-log").scrollTop = 1e9;
  return el;
}

async function sendChat(text) {
  text = text.trim();
  if (!text) return;
  $("#chat-input").value = "";
  addMsg("user", text);
  state.chat.push({ role: "user", content: text });
  const out = addMsg("assistant", "");
  out.appendChild(typingDots());
  try {
    const res = await api("/api/chat", { method: "POST", body: { mensajes: state.chat, proyecto: state.project?.id } });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let acc = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      if (acc) out.textContent = acc;
      $("#chat-log").scrollTop = 1e9;
    }
    state.chat.push({ role: "assistant", content: acc });
  } catch (e) {
    out.textContent = `⚠️ ${e.message}`;
  }
}

// -------------------------------------------------------------- ajustes
async function openSettings() {
  const [cfg, st, mot] = await Promise.all([api("/api/config"), api("/api/estado"), api("/api/motores")]);
  state.config = cfg;
  state.status = st;
  const o = st.ollama;
  const g = st.generador;
  const vision = o.vision || [];
  const modelOpts = [...new Set([...vision, cfg.vision_model])].map((m) => `<option ${m === cfg.vision_model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const installed = new Set((o.modelos || []).map((m) => m.name));
  const ckpts = g.checkpoints || [];
  const ckptOptions = (sel) => [...new Set([sel, ...ckpts].filter(Boolean))].map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
  const p = cfg.printer;
  $("#settings-body").innerHTML = `
    <section>
      <h3>Motores</h3>
      <div id="engine-rows">${Object.keys(ENGINE_NAMES).map((n) => engineRow(n, mot[n])).join("")}</div>
      <div class="grid2" style="margin-top:10px">
        <div class="field"><label>Ruta de Ollama (opcional)</label><input type="text" id="s-ollama-exe" data-engine-field placeholder="Se busca sola" value="${esc(cfg.ollama_exe)}"></div>
        <div class="field"><label>Carpeta de ComfyUI</label><input type="text" id="s-comfy-dir" data-engine-field placeholder="Ej.: D:\\IA\\ComfyUI_windows_portable" value="${esc(cfg.comfy_dir)}"></div>
      </div>
      <label class="switch"><input type="checkbox" id="s-autostart" ${cfg.encender_al_abrir ? "checked" : ""}>Encender los motores al abrir Plasmo</label>
      <details class="params">
        <summary>Avanzado: comandos personalizados</summary>
        <div class="grid2" style="margin-top:8px">
          <div class="field"><label>Comando de Ollama</label><input type="text" id="s-ollama-cmd" data-engine-field value="${esc(cfg.ollama_comando)}"></div>
          <div class="field"><label>Comando de ComfyUI</label><input type="text" id="s-comfy-cmd" data-engine-field value="${esc(cfg.comfy_comando)}"></div>
        </div>
        <div class="hint">Si los llenas, Plasmo los usa en vez de buscar las instalaciones. Ejemplo: <code>"D:\\IA\\ComfyUI_windows_portable\\python_embeded\\python.exe" -s ComfyUI\\main.py --lowvram --port 8188</code></div>
      </details>
    </section>
    <section>
      <h3>Ollama · visión y chat</h3>
      <div class="grid2">
        <div class="field"><label>Dirección</label><input type="url" id="s-ollama" value="${esc(cfg.ollama_url)}"></div>
        <div class="field"><label>Modelo de visión</label><select id="s-vision">${modelOpts}</select></div>
      </div>
      <div class="hint">${o.ok ? `Ollama ${esc(o.version)} conectado.` : `No responde en ${esc(o.url)}. Instálalo desde ollama.com y ábrelo.`}</div>
      ${o.recomendados.map((r) => `
        <div class="model-row"><div class="grow"><b>${esc(r.tag)}</b> <span class="muted small">· ${r.size_gb} GB · ${esc(r.label)}</span></div>
        ${installed.has(r.tag) ? `<span class="chip">Descargado</span>` : `<button class="sm" data-pull="${esc(r.tag)}" ${o.ok ? "" : "disabled"} data-icon="download">Descargar</button>`}</div>`).join("")}
    </section>
    <section>
      <h3>Generador 3D</h3>
      <div class="grid2">
        <div class="field"><label>Motor</label><select id="s-gen">
          <option value="comfyui" ${cfg.generator === "comfyui" ? "selected" : ""}>ComfyUI + Hunyuan3D (recomendado)</option>
          <option value="hunyuan_api" ${cfg.generator === "hunyuan_api" ? "selected" : ""}>Servidor Hunyuan3D-2 (api_server)</option></select></div>
        <div class="field"><label>Dirección de ComfyUI</label><input type="url" id="s-comfy" value="${esc(cfg.comfy_url)}"></div>
        <div class="field"><label>Checkpoint 3D</label><select id="s-ckpt">${ckptOptions(cfg.comfy_checkpoint)}</select></div>
        <div class="field"><label>Checkpoint texto→imagen (opcional)</label><select id="s-t2i"><option value="">— ninguno —</option>${ckptOptions(cfg.comfy_t2i_checkpoint).replace(/<option selected><\/option>/, "")}</select></div>
        <div class="field"><label>Dirección del servidor Hunyuan3D</label><input type="url" id="s-hy" value="${esc(cfg.hunyuan_api_url)}"></div>
      </div>
      <div class="hint">${esc(g.mensaje || "")}${g.gpu ? ` · ${esc(g.gpu)} · ${g.vram_gb} GB VRAM` : ""}</div>
      ${(g.catalogo || []).map((c) => `<div class="model-row"><div class="grow"><b>${esc(c.nombre)}</b><div class="muted small">${esc(c.archivo)} · ${c.gb} GB → <code>ComfyUI/models/checkpoints</code></div></div>
        ${ckpts.includes(c.archivo) ? `<span class="chip">Instalado</span>` : `<a href="${esc(c.url)}" target="_blank" rel="noopener"><button class="sm" data-icon="download">Enlace</button></a>`}</div>`).join("")}
    </section>
    <section>
      <h3>Recorte y profundidad (corren en CPU)</h3>
      <div class="grid2">
        <div class="field"><label>Quitar fondo con</label><select id="s-bg">
          ${[["auto", "Máxima calidad: IS-Net + BiRefNet (recomendado)"], ["isnet-general-use", "IS-Net"], ["birefnet-lite", "BiRefNet lite"], ["u2netp", "U2-Net P (rápido)"], ["simple", "Por color (sin modelo)"]]
            .map(([k, l]) => `<option value="${k}" ${cfg.bg_model === k ? "selected" : ""}>${l}</option>`).join("")}</select></div>
        <div class="field"><label>Profundidad para relieves</label><select id="s-depth">
          <option value="depth-anything-v2-small" ${cfg.depth_model === "depth-anything-v2-small" ? "selected" : ""}>Depth Anything V2 Small</option>
          <option value="luminancia" ${cfg.depth_model === "luminancia" ? "selected" : ""}>Luz y sombra (sin modelo)</option></select></div>
      </div>
      ${st.onnx.map((m) => `<div class="model-row"><div class="grow"><b>${esc(m.nombre)}</b> <span class="muted small">· ${m.mb} MB · ${esc(m.tipo)}</span></div>
        ${m.descargado ? `<span class="chip">Descargado</span>` : `<button class="sm" data-onnx="${esc(m.clave)}" data-icon="download">Descargar</button>`}</div>`).join("")}
      <div class="hint">Si falta un modelo, se descarga solo la primera vez que se usa.</div>
    </section>
    <section>
      <h3>Impresora</h3>
      <div class="grid4">
        <div class="field"><label>Cama X (mm)</label><input type="number" id="s-bx" value="${p.bed_x}"></div>
        <div class="field"><label>Cama Y (mm)</label><input type="number" id="s-by" value="${p.bed_y}"></div>
        <div class="field"><label>Alto Z (mm)</label><input type="number" id="s-bz" value="${p.bed_z}"></div>
        <div class="field"><label>Boquilla (mm)</label><input type="number" id="s-noz" value="${p.nozzle}" step="0.1"></div>
        <div class="field"><label>Tecnología</label><select id="s-kind"><option value="fdm" ${p.kind === "fdm" ? "selected" : ""}>FDM (filamento)</option><option value="sla" ${p.kind === "sla" ? "selected" : ""}>Resina</option></select></div>
        <div class="field"><label>Material</label><select id="s-mat">${["PLA", "PETG", "ABS", "ASA", "TPU", "Resina"].map((m) => `<option ${p.material === m ? "selected" : ""}>${m}</option>`).join("")}</select></div>
      </div>
    </section>
    <section>
      <h3>Carpeta de proyectos</h3>
      <input type="text" id="s-out" value="${esc(cfg.output_dir)}">
    </section>`;
  hydrateIcons($("#settings-body"));
  bindEngineRows();
  $$("[data-engine-field]").forEach((el) => el.addEventListener("change", async () => {
    await api("/api/config", { method: "PUT", body: engineFields() });
    refreshEngineRows();
  }));
  $$("[data-pull]").forEach((b) => b.addEventListener("click", () => downloadJob(b, "/api/ollama/descargar", { modelo: b.dataset.pull })));
  $$("[data-onnx]").forEach((b) => b.addEventListener("click", () => downloadJob(b, "/api/onnx/descargar", { clave: b.dataset.onnx })));
  if (!$("#settings").open) $("#settings").showModal();
}

function engineFields() {
  return {
    ollama_exe: $("#s-ollama-exe").value.trim(),
    comfy_dir: $("#s-comfy-dir").value.trim(),
    ollama_comando: $("#s-ollama-cmd").value.trim(),
    comfy_comando: $("#s-comfy-cmd").value.trim(),
    encender_al_abrir: $("#s-autostart").checked,
  };
}

function engineRow(n, m) {
  const missing = n === "ollama" ? "No encontré Ollama: instálalo desde ollama.com o indica su ruta abajo." : "No encontré ComfyUI: indica su carpeta abajo.";
  const action = m.encendido
    ? (m.lanzado_por_plasmo ? `<button class="sm" data-motor-off="${n}">Apagar</button>` : "")
    : `<button class="sm" data-motor-on="${n}" data-icon="power" ${m.encontrado ? "" : "disabled"}>Encender</button>`;
  return `<div class="model-row engine-row" data-motor="${n}">
    <div class="grow"><b>${esc(m.nombre)}</b> <span class="chip ${m.encendido ? "on" : "off"}">${m.encendido ? "Encendido" : "Apagado"}</span>
      ${m.lanzado_por_plasmo ? `<span class="muted small">· lo encendió Plasmo</span>` : ""}
      <div class="muted small"><code>${esc(m.encontrado ? m.descripcion : missing)}</code></div></div>${action}</div>`;
}

async function refreshEngineRows() {
  const box = $("#engine-rows");
  if (!box) return;
  const mot = await api("/api/motores");
  box.innerHTML = Object.keys(ENGINE_NAMES).map((n) => engineRow(n, mot[n])).join("");
  hydrateIcons(box);
  bindEngineRows();
}

function bindEngineRows() {
  $$("[data-motor-on]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    b.innerHTML = `<span class="spinner"></span> Encendiendo…`;
    await api("/api/config", { method: "PUT", body: engineFields() });
    await startEngines([b.dataset.motorOn]);
    refreshEngineRows();
  }));
  $$("[data-motor-off]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    b.innerHTML = `<span class="spinner"></span> Apagando…`;
    await api("/api/motores/apagar", { method: "POST", body: { motores: [b.dataset.motorOff] } });
    toast(`${ENGINE_NAMES[b.dataset.motorOff]} apagado`);
    await refreshEngineRows();
    refreshStatus();
  }));
}

async function downloadJob(btn, url, body) {
  btn.disabled = true;
  const row = btn.closest(".model-row");
  const bar = document.createElement("div");
  bar.className = "bar mini-bar";
  bar.innerHTML = "<i></i>";
  btn.replaceWith(bar);
  try {
    const job = await api(url, { method: "POST", body });
    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const j = await api(`/api/trabajos/${job.id}`);
      bar.firstChild.style.width = `${j.porcentaje}%`;
      bar.title = j.mensaje;
      if (j.estado === "listo") { bar.replaceWith(Object.assign(document.createElement("span"), { className: "chip", textContent: "Descargado" })); break; }
      if (j.estado === "error") throw new Error(j.error);
    }
    refreshStatus();
  } catch (e) {
    toast(e.message, "err");
    row.querySelector(".mini-bar")?.remove();
  }
}

async function saveSettings() {
  const num = (id) => Number($(id).value);
  const data = {
    ollama_url: $("#s-ollama").value.trim(),
    vision_model: $("#s-vision").value,
    generator: $("#s-gen").value,
    comfy_url: $("#s-comfy").value.trim(),
    comfy_checkpoint: $("#s-ckpt").value,
    comfy_t2i_checkpoint: $("#s-t2i").value,
    hunyuan_api_url: $("#s-hy").value.trim(),
    bg_model: $("#s-bg").value,
    depth_model: $("#s-depth").value,
    output_dir: $("#s-out").value.trim(),
    ...engineFields(),
    printer: { bed_x: num("#s-bx"), bed_y: num("#s-by"), bed_z: num("#s-bz"), nozzle: num("#s-noz"), kind: $("#s-kind").value, material: $("#s-mat").value },
  };
  state.config = await api("/api/config", { method: "PUT", body: data });
  $("#settings").close();
  viewer?.setBed(data.printer.bed_x, data.printer.bed_y);
  toast("Ajustes guardados", "ok");
  refreshStatus();
  loadProjects();
}

// ------------------------------------------------------------ tutorial
async function createExample() {
  showWorkspace();
  $("#img-preview").src = "/ejemplo.png";
  setCutting(true, "Preparando el ejemplo…");
  renderAnalysis(null);
  try {
    const meta = await api("/api/ejemplo", { method: "POST" });
    await loadProjects();
    setCutting(false);
    await openProject(meta.id, { reveal: true });
  } catch (e) {
    setCutting(false);
    throw e;
  }
}

function engineStatus() {
  const st = state.status;
  if (!st) return "";
  const row = (ok, text) => `<span><span class="dot ${ok ? "ok" : "err"}"></span>${text}</span>`;
  const o = st.ollama;
  const g = st.generador;
  return `<div class="tour-status">
    ${row(o.ok && o.modelo_listo, o.ok ? (o.modelo_listo ? `Ollama listo con <b>${esc(o.modelo_activo)}</b>` : `Ollama abierto, falta <b>${esc(o.modelo_activo)}</b>`) : "Ollama no responde")}
    ${row(g.ok, g.ok ? `${esc(g.nombre)} listo${g.gpu ? ` en ${esc(g.gpu.replace(/^cuda:\d+\s*/, ""))}` : ""}` : `${esc(g.nombre)} sin conexión`)}
  </div>${!(o.ok && g.ok) ? `<p class="muted small" style="margin:8px 0 0">Sin ellos igual puedes hacer relieves, litofanías, llaveros, imanes, pixel art y cortadores.</p>` : ""}`;
}

function buildTourSteps() {
  const ollamaReady = () => state.status?.ollama?.ok && state.status?.ollama?.modelo_listo;
  return [
    {
      title: "Bienvenido a Plasmo",
      wide: true,
      extra: `<div class="tour-hero"><div class="tour-logo" id="tour-logo"></div></div>`,
      text: `<p>Plasmo convierte una foto en una pieza lista para tu impresora 3D, todo en tu PC.</p>
             <p>En unos dos minutos te muestro cómo, con un gatito de ejemplo. Usa <kbd>→</kbd> para avanzar y <kbd>Esc</kbd> para salir.</p>`,
      after: async (card) => { const el = card.querySelector("#tour-logo"); if (el) el.innerHTML = await logoSVG("tour"); },
    },
    {
      target: ".status", placement: "bottom", title: "Tus motores de IA",
      text: `<p><b>Ollama</b> mira la foto y te aconseja. <b>ComfyUI</b> con Hunyuan3D genera la figura 3D.</p><p>Verde es listo. Si están apagados, el botón <b>Encender IA</b> los arranca sin abrir ventanas; si no los encuentra, indícale dónde están en Ajustes.</p>`,
      extra: engineStatus,
      before: async () => { showWelcome(); await refreshStatus(); },
    },
    {
      target: "#btn-settings", placement: "bottom", title: "Ajustes",
      text: `<p>Aquí conectas los motores, descargas los modelos recomendados (como <b>qwen3.5:9b</b>) y pones las medidas de tu cama.</p>`,
    },
    {
      target: "#drop", placement: "bottom", title: "Sube tu foto",
      text: `<p>Arrástrala aquí, haz clic para elegirla o pégala con <kbd>Ctrl</kbd>+<kbd>V</kbd>.</p><p>Sale mejor con <b>un solo objeto</b>, entero, bien iluminado y con fondo simple.</p>`,
      before: showWelcome,
    },
    {
      target: ".idea", placement: "bottom", title: "O describe tu idea",
      text: `<p>Escribe lo que quieres y Ollama prepara el prompt para que ComfyUI dibuje la imagen. Para esto hace falta un checkpoint SDXL en Ajustes.</p>`,
    },
    {
      title: "Probemos con un ejemplo",
      text: `<p>Voy a abrir un proyecto con un gatito de juguete. Todo lo que sigue lo haces igual con tus fotos.</p>`,
      action: { label: "Usar el gatito de ejemplo", busy: "Recortando…", run: createExample },
    },
    {
      target: () => $(".col-left .card"), placement: "right", title: "El recorte",
      text: `<p>Plasmo separó al gatito del fondo combinando dos modelos (IS-Net y BiRefNet), en tu CPU. Debajo ves la <b>nota del recorte</b> y sus avisos.</p><p>Si algo quedó mal, cambia el método y pulsa <b>Recortar</b>. Con <b>Original</b> ves la foto entera.</p>`,
    },
    {
      target: "#card-analysis", placement: "right", title: "Lo que ve la IA",
      text: () => ollamaReady()
        ? `<p>Ollama describe la foto y sugiere la forma, el tamaño, los soportes y el material.</p>`
        : `<p>Cuando Ollama esté conectado, aquí verás qué hay en la foto, la forma sugerida, el tamaño y consejos de impresión.</p>`,
      action: { label: "Analizar ahora", busy: "Ollama está mirando…", run: async () => { if (!ollamaReady()) throw new Error("Ollama no está listo; conéctalo en Ajustes cuando quieras."); await analyze(); } },
    },
    {
      target: "#forms", placement: "right", title: "Elige la forma",
      text: `<ul class="tour-list"><li><b>Figura 3D</b> con IA (necesita ComfyUI)</li><li><b>Relieve</b> y <b>Litofanía</b> (plana, curva o lámpara)</li>
             <li><b>Llavero</b> e <b>Imán</b> de nevera</li><li><b>Pixel art</b> multicolor y <b>Cortador</b> de galletas</li></ul>
             <p style="margin-top:8px">La que sugiere la IA lleva la etiqueta <b>sugerida</b>.</p>`,
      before: () => { $("#card-form").scrollIntoView({ block: "center" }); },
    },
    {
      target: "#params-box", placement: "right", title: "Parámetros",
      text: `<p>Elegí <b>Llavero</b> para el ejemplo. Cada forma tiene sus parámetros: tamaño, grosor, relieve, argolla…</p><p>Pasa el mouse por un nombre para ver su ayuda.</p>`,
      before: () => { selectForm("llavero"); $("#params-box").open = true; },
    },
    {
      target: "#print-box", placement: "right", title: "Preparación para imprimir",
      text: `<p>Gira la pieza o divídela si no cabe en tu cama. En la <b>Figura 3D</b> también eliges el tamaño, una base y cuánto aplanar abajo.</p>`,
      before: () => { $("#print-box").open = true; },
    },
    {
      target: ".generate", placement: "right", title: "Genera la pieza",
      text: `<p>La barra muestra el <b>% total</b> y cada fase, mientras la boquilla imprime la pieza capa por capa.</p>`,
      action: { label: "Generar el llavero", busy: "Generando…", run: async () => {
        $("#print-box").open = false;
        const ok = await generate();
        if (!ok) throw new Error("No se pudo generar esta vez.");
      } },
    },
    {
      target: ".viewer-card", placement: "left", title: "Tu pieza en 3D",
      text: `<p>Arrastra para girar y usa la rueda para el zoom. Arriba a la izquierda cambias entre versiones; a la derecha tienes alambre, cama y encuadre.</p>`,
    },
    {
      target: "#card-report", placement: "left", title: "¿Se puede imprimir?",
      text: `<p>Plasmo revisa si la malla está cerrada, los voladizos, la estabilidad, si cabe en tu cama y cuánto pesa.</p><p>Después descarga el <b>3MF</b> o el <b>STL</b>, o ábrelo directo en tu slicer.</p>`,
      before: () => { $("#card-report").scrollIntoView({ block: "center" }); },
    },
    {
      target: "#btn-chat", placement: "bottom", title: "Tu asistente",
      text: `<p>Pregúntale por temperaturas, soportes o ajustes del slicer. Conoce la pieza que tienes abierta.</p>`,
    },
    {
      target: "#btn-help", placement: "bottom", title: "Cuando quieras, otra vez",
      text: `<p>Este botón vuelve a abrir el tutorial.</p>`,
    },
    {
      title: "¡Listo para imprimir!",
      wide: true,
      extra: `<div class="tour-hero"><div class="tour-logo" id="tour-logo-end"></div></div>`,
      text: `<p>Ya sabes todo lo básico. Sube tu primera foto y a imprimir.</p><p class="muted small">La guía para instalar la IA en tu PC está en <b>docs/INSTALAR-IA.md</b>.</p>`,
      after: async (card) => {
        const el = card.querySelector("#tour-logo-end");
        if (el) el.innerHTML = await logoSVG("tourend");
        const r = card.getBoundingClientRect();
        celebrate(r.left + r.width / 2, r.top + 60);
      },
    },
  ];
}

function startTour() {
  $("#chat").classList.add("hidden");
  new Tour(buildTourSteps(), {
    onEnd: () => {
      if (!state.config?.tutorial_visto) api("/api/config", { method: "PUT", body: { tutorial_visto: true } }).then((c) => { state.config = c; }).catch(() => {});
    },
  }).start();
}

// --------------------------------------------------------------- inicio
function renderFeatures() {
  $("#features").innerHTML = state.forms.map((f, i) => `<div class="feature" style="--k:${i}"><b>${icon(f.clave)}${esc(f.nombre)}</b><span>${esc(f.descripcion)}</span></div>`).join("");
}

async function fromIdea() {
  const idea = $("#idea").value.trim();
  if (!idea) return toast("Escribe una idea primero.");
  try {
    const job = await api("/api/texto-a-imagen", { method: "POST", body: { idea } });
    const res = await watchJob(job, { fixed: true });
    await loadProjects();
    await openProject(res.proyecto);
    analyze();
  } catch (e) {
    toast(e.message, "err");
  }
}

function bind() {
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => createProject(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "dragleave" && e.relatedTarget) return; drop.classList.remove("over"); }));
  document.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) createProject(f); });
  document.addEventListener("paste", (e) => {
    if (e.target.closest("input, textarea")) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) createProject(item.getAsFile());
  });
  $("#btn-new").addEventListener("click", showWelcome);
  $("#btn-idea").addEventListener("click", fromIdea);
  $("#idea").addEventListener("keydown", (e) => { if (e.key === "Enter") fromIdea(); });
  $$("#img-tabs button").forEach((b) => b.addEventListener("click", () => { state.imgView = b.dataset.v; renderImage(); }));
  $("#btn-recut").addEventListener("click", async () => {
    if (!state.project) return;
    const btn = $("#btn-recut");
    btn.disabled = true;
    setCutting(true, "Recortando de nuevo…");
    try {
      await api(`/api/proyectos/${encodeURIComponent(state.project.id)}/recorte`, { method: "POST", body: { modelo: $("#cut-model").value, objetos: $("#cut-objects").value } });
      state.project = await api(`/api/proyectos/${encodeURIComponent(state.project.id)}`);
      state.imgView = "recorte";
      setCutting(false);
      renderImage(true);
      renderCutQuality();
      loadProjects();
    } catch (e) { toast(e.message, "err"); } finally { setCutting(false); btn.disabled = false; }
  });
  $("#btn-analyze").addEventListener("click", analyze);
  $("#btn-generate").addEventListener("click", generate);
  $("#pg-cancel").addEventListener("click", () => state.job && api(`/api/trabajos/${state.job.id}/cancelar`, { method: "POST" }));
  $("#v-wire").addEventListener("click", (e) => e.currentTarget.classList.toggle("on", viewer?.toggleWire()));
  $("#v-bed").addEventListener("click", (e) => e.currentTarget.classList.toggle("on", viewer?.toggleBed()));
  $("#v-reset").addEventListener("click", () => viewer?.frame());
  $("#btn-help").addEventListener("click", startTour);
  $("#btn-power").addEventListener("click", () => startEngines());
  $("#btn-chat").addEventListener("click", () => $("#chat").classList.toggle("hidden"));
  $("#chat-close").addEventListener("click", () => $("#chat").classList.add("hidden"));
  $("#chat-clear").addEventListener("click", () => { state.chat = []; $("#chat-log").innerHTML = ""; addMsg("assistant", "Listo, empezamos de nuevo."); });
  $("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendChat($("#chat-input").value); });
  $("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat($("#chat-input").value); } });
  $$("#chat-suggest button").forEach((b) => b.addEventListener("click", () => sendChat(b.textContent)));
  $("#btn-settings").addEventListener("click", () => openSettings().catch((e) => toast(e.message, "err")));
  $("#pill-ollama").addEventListener("click", () => openSettings());
  $("#pill-gen").addEventListener("click", () => openSettings());
  $("#set-close").addEventListener("click", () => $("#settings").close());
  $("#set-cancel").addEventListener("click", () => $("#settings").close());
  $("#set-save").addEventListener("click", () => saveSettings().catch((e) => toast(e.message, "err")));
}

async function init() {
  const hideSplash = await splashReady();
  hydrateIcons();
  bind();
  logoSVG("brand").then((svg) => { $("#brand-logo").innerHTML = svg; }).catch(() => {});
  logoSVG("hero").then((svg) => { $("#hero-logo").innerHTML = svg; }).catch(() => {});
  try {
    const [cfg, forms] = await Promise.all([api("/api/config"), api("/api/formas")]);
    state.config = cfg;
    state.forms = forms.formas;
    renderFeatures();
    await loadProjects();
    await refreshStatus();
    setInterval(refreshStatus, 15000);
    if (cfg.encender_al_abrir && enginesOff().length) startEngines();
    const linked = new URLSearchParams(location.hash.slice(1)).get("p");
    if (linked) await openProject(linked).catch(() => showWelcome());
  } finally {
    hideSplash().then(() => {
      // la primera vez, el tutorial arranca solo al terminar la pantalla de inicio
      if (state.config && !state.config.tutorial_visto && !location.hash) setTimeout(startTour, 450);
    });
  }
}

init().catch((e) => toast(e.message, "err"));
