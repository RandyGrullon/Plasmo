"""Orquesta un trabajo de punta a punta: imagen → pieza imprimible exportada."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from plasmo import generators
from plasmo.config import Settings
from plasmo.imaging import processing as proc
from plasmo.jobs import Job
from plasmo.mesh import color as paint
from plasmo.mesh import export, forms
from plasmo.mesh import printability as pr
from plasmo.ollama_client import OllamaClient
from plasmo.projects import ProjectStore

FORM_LABELS = {
    "figura": "Figura 3D (IA)",
    "relieve": "Relieve",
    "litofania": "Litofanía",
    "llavero": "Llavero",
    "iman": "Imán de nevera",
    "pixel": "Pixel art multicolor",
    "cortador": "Cortador de galletas",
}
THIN_BY_DESIGN = {"litofania", "cortador"}


def thin_policy(form: str) -> str:
    if form in THIN_BY_DESIGN:
        return "omitir"
    return "error" if form == "figura" else "aviso"
USES_DEPTH = {"relieve", "llavero", "iman"}


def phases_for(form: str) -> list[tuple[str, str, float]]:
    if form == "figura":
        return [
            ("preparar", "Preparar imagen", 4),
            ("liberar_vram", "Liberar VRAM de Ollama", 1),
            ("generar", "Generar forma 3D con IA", 62),
            ("convertir", "Convertir y orientar", 4),
            ("reparar", "Reparar malla", 11),
            ("imprimir", "Escalar, base y aplanado", 8),
            ("revisar", "Revisar imprimibilidad", 6),
            ("exportar", "Exportar STL / 3MF", 4),
        ]
    return [
        ("preparar", "Preparar imagen", 8),
        ("profundidad", "Calcular profundidad", 14),
        ("construir", f"Construir {FORM_LABELS.get(form, form).lower()}", 32),
        ("simplificar", "Optimizar malla", 18),
        ("imprimir", "Preparar para imprimir", 6),
        ("revisar", "Revisar imprimibilidad", 14),
        ("exportar", "Exportar STL / 3MF", 8),
    ]


def load_source(store: ProjectStore, pid: str, settings: Settings) -> forms.Source:
    folder = store.path(pid)
    img = proc.load_image(folder / "original.png")
    mask_path = folder / "mascara.png"
    if mask_path.exists():
        mask = np.asarray(Image.open(mask_path).convert("L").resize(img.size), dtype=np.float32) / 255.0
    else:
        mask = np.ones((img.height, img.width), dtype=np.float32)
    return forms.Source(img, mask, settings.depth_model)


def segment_project(store: ProjectStore, pid: str, model_key: str, objects: str = "todos") -> dict:
    """Recorta el sujeto y guarda máscara + recorte. 'ninguno' usa la imagen completa.

    `objects`: "todos" conserva todo lo que se detecte; "principal" deja solo el protagonista.
    """
    folder = store.path(pid)
    img = proc.load_image(folder / "original.png")
    info: dict = {}
    if model_key == "ninguno":
        mask = np.ones((img.height, img.width), dtype=np.float32)
    else:
        mask, info = proc.segment_detail(img, model_key, objects=objects)
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(folder / "mascara.png")
    cut = proc.cutout(img, mask)
    cut.save(folder / "recorte.png")
    quality = proc.mask_quality(mask, info) if model_key != "ninguno" else {"nota": 100, "calidad": "completa", "objetos": 1, "avisos": []}
    result = {"modelo": model_key, "cobertura": round(float((mask > 0.5).mean()), 3), "tamano": list(cut.size), **info, **quality}
    result.pop("pixeles", None)
    store.update_meta(pid, recorte=result)
    return result


def _next_version(store: ProjectStore, pid: str, form: str) -> tuple[str, Path]:
    meta = store.meta(pid)
    n = len(meta.get("versiones") or []) + 1
    name = f"v{n}-{form}"
    path = store.path(pid) / name
    path.mkdir(exist_ok=True)
    return name, path


def _prepare_figure(mesh: trimesh.Trimesh, opts: dict, src: forms.Source) -> tuple[list[forms.Part], list[str]]:
    notes: list[str] = []
    pr.place_on_bed(mesh)
    pr.scale_to(mesh, float(opts.get("tamano_mm") or 100), opts.get("eje", "alto"))
    pr.place_on_bed(mesh)
    flatten = float(opts.get("aplanar_mm", 1.0) or 0)
    if flatten > 0 and mesh.is_volume:
        mesh = pr.flatten_bottom(mesh, flatten)
        notes.append(f"Aplané {flatten:.1f} mm abajo para que pegue a la cama.")
    base = opts.get("base", "ninguna")
    base_top = None
    if base in ("disco", "rectangular") and mesh.is_volume:
        base_mm = float(opts.get("base_mm") or 3)
        mesh = pr.add_base(mesh, base, base_mm)
        base_top = base_mm + 0.05
        notes.append(f"Base {base} de {base_mm:.0f} mm.")
    vertex_colors = None
    if opts.get("color", True):
        # se pinta mientras la figura todavía mira al frente (antes de girarla)
        try:
            body = mesh.vertices if base_top is None else mesh.vertices[mesh.vertices[:, 2] >= base_top]
            frame = paint.fit_frame(body, src.cut_mask)
            vertex_colors = paint.project_colors(mesh, src.cut, frame, base_top)
            notes.append("Pinté la figura con los colores de la foto; la espalda toma un promedio del frente.")
        except Exception as exc:
            notes.append(f"No pude pasarle los colores de la foto: {exc}")
    pr.rotate_z(mesh, float(opts.get("rotacion_z") or 0))
    pr.place_on_bed(mesh)
    return [forms.Part("figura", mesh, vertex_colors=vertex_colors)], notes


def _maybe_split(parts: list[forms.Part], settings: Settings, opts: dict) -> tuple[list[forms.Part], list[str]]:
    p = settings.printer
    bed = (p.bed_x, p.bed_y, p.bed_z)
    mesh = export.combined(parts)
    if not opts.get("dividir") or len(parts) != 1 or np.all(mesh.extents <= np.array(bed)):
        return parts, []
    if not mesh.is_volume:
        return parts, ["No pude dividirla porque la malla no es un volumen cerrado."]
    pieces, offsets = pr.split_for_bed(mesh, bed, with_offsets=True)
    colors = parts[0].vertex_colors
    tree = cKDTree(mesh.vertices) if colors is not None else None
    out = []
    for i, (m, off) in enumerate(zip(pieces, offsets)):
        vc = colors[tree.query(m.vertices - off)[1]] if tree is not None else None  # colores del vértice original más cercano
        out.append(forms.Part(f"parte_{i + 1}", m, parts[0].color, vertex_colors=vc))
    return out, [f"Dividida en {len(out)} partes para que quepan en la cama ({bed[0]:.0f}×{bed[1]:.0f}×{bed[2]:.0f} mm)."]


def run_generation(job: Job, store: ProjectStore, settings: Settings, pid: str, form: str, params: dict, opts: dict) -> dict:
    if form != "figura" and form not in forms.FORMS:
        raise ValueError(f"Forma desconocida: {form}")
    params = params or {}
    opts = opts or {}
    notes: list[str] = []
    version, folder = _next_version(store, pid, form)

    job.start("preparar")
    src = load_source(store, pid, settings)
    job.done("preparar")

    if form == "figura":
        job.start("liberar_vram")
        if settings.vision_model:
            OllamaClient(settings.ollama_url).unload(settings.vision_model)
        job.done("liberar_vram")

        job.start("generar", "Generando con IA…")
        square = proc.square_on_background(src.cut, 1024)
        square.save(folder / "entrada-ia.png")
        gen = generators.get(settings)
        glb = gen.generate(square, job, "generar", params)
        (folder / "ia-crudo.glb").write_bytes(glb)
        job.done("generar")

        job.start("convertir")
        loaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
        mesh = pr.y_up_to_z_up(loaded)
        job.done("convertir")

        job.start("reparar")
        if opts.get("reparar", True):
            mesh, log = pr.repair(mesh, voxel_res=int(opts.get("voxeles", 384)))
            notes += log
        level = opts.get("pulido", "suave")
        if level in ("suave", "fuerte"):
            pr.polish(mesh, level)
            notes.append(f"Pulido {level} de la superficie (Taubin, sin encoger la pieza).")
        mesh = pr.simplify(mesh, int(opts.get("caras_max", 500_000)))
        job.done("reparar")

        job.start("imprimir")
        parts, more = _prepare_figure(mesh, opts, src)
        notes += more
    else:
        job.start("profundidad")
        if form in USES_DEPTH and params.get("fuente", "profundidad") == "profundidad":
            src.relief_map("profundidad")
            if src.depth_model == "luminancia":
                notes.append("Sin modelo de profundidad: usé la luminancia de la foto.")
            job.done("profundidad")
        else:
            job.skip("profundidad")

        job.start("construir")
        result = forms.FORMS[form](src, params)
        parts = result.parts
        notes += result.notes
        job.done("construir")

        job.start("simplificar")
        for i, part in enumerate(parts):
            part.mesh = pr.simplify(part.mesh, int(opts.get("caras_max", 250_000)))
            job.update("simplificar", (i + 1) / len(parts))
        job.done("simplificar")

        job.start("imprimir")
        if opts.get("rotacion_z"):
            for part in parts:
                pr.rotate_z(part.mesh, float(opts["rotacion_z"]))

    parts, more = _maybe_split(parts, settings, opts)
    notes += more
    job.done("imprimir")

    job.start("revisar")
    report = pr.analyze(export.combined(parts), settings.printer.__dict__, thin=thin_policy(form))
    if form in THIN_BY_DESIGN:
        report["nota_paredes"] = "Pieza fina a propósito: no se evalúan paredes finas."
    job.done("revisar")

    job.start("exportar")
    meta = store.meta(pid)
    stem = (meta.get("analisis") or {}).get("nombre_archivo") or meta.get("nombre") or "modelo"
    files = export.write_all(parts, folder, f"{stem}-{form}", stem)
    info = {
        "version": version,
        "forma": form,
        "forma_nombre": FORM_LABELS.get(form, form),
        "parametros": params,
        "impresion": opts,
        "archivos": files,
        "piezas": [{"nombre": p.name, "color": "#%02x%02x%02x" % p.color} for p in parts],
        "reporte": report,
        "notas": notes,
    }
    (folder / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    versions = meta.get("versiones") or []
    versions.append(info)
    store.update_meta(pid, versiones=versions)
    job.done("exportar")
    return {"proyecto": pid, **info}


def run_reanalysis(store: ProjectStore, settings: Settings, pid: str, version: str) -> dict:
    """Vuelve a revisar una versión (por ejemplo tras cambiar de impresora)."""
    folder = store.path(pid) / version
    info = json.loads((folder / "info.json").read_text(encoding="utf-8-sig"))
    mesh = trimesh.load(folder / info["archivos"]["stl"], force="mesh")
    info["reporte"] = pr.analyze(mesh, settings.printer.__dict__, thin=thin_policy(info["forma"]))
    (folder / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
