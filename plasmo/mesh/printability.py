"""Análisis y preparación para imprimir: reparar, orientar, escalar, base y corte.

Las métricas siguen las de "Analyze Printability" de Meshy (cerrada, volumen,
aristas no manifold, caras degeneradas, agujeros, paredes finas) y se suman
voladizos, estabilidad, cama y peso estimado.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

MIN_WALL_MM = {"fdm": 1.0, "sla": 0.8}
DENSITY = {"PLA": 1.24, "PETG": 1.27, "ABS": 1.04, "ASA": 1.07, "TPU": 1.21, "Resina": 1.12}


# ------------------------------------------------------------------ manifold3d

def to_manifold(mesh: trimesh.Trimesh):
    import manifold3d as mf

    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))
    if m.status() != mf.Error.NoError:
        raise ValueError(f"la malla no es manifold ({m.status()})")
    return m


def from_manifold(m) -> trimesh.Trimesh:
    out = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(out.vert_properties)[:, :3], faces=np.asarray(out.tri_verts), process=True)


# --------------------------------------------------------------------- análisis

def edge_stats(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """(aristas no manifold, agujeros = lazos de borde)."""
    edges = mesh.edges_sorted.astype(np.int64)
    if len(edges) == 0:
        return 0, 0
    keys = edges[:, 0] * (len(mesh.vertices) + 1) + edges[:, 1]
    uniq, first, counts = np.unique(keys, return_index=True, return_counts=True)
    non_manifold = int((counts > 2).sum())
    boundary = edges[first[counts == 1]]
    holes = 0
    if len(boundary):
        holes = len(trimesh.graph.connected_components(boundary, min_len=1))
    return non_manifold, holes


def thin_wall_ratio(mesh: trimesh.Trimesh, min_wall: float, samples: int = 1500) -> float | None:
    """Fracción de superficie con grosor local menor que `min_wall` (rayos hacia adentro)."""
    if not mesh.is_watertight or len(mesh.faces) == 0:
        return None
    try:
        pts, fidx = trimesh.sample.sample_surface(mesh, samples, seed=11)
        normals = mesh.face_normals[fidx]
        origins = pts - normals * 1e-3
        locs, ray_idx, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    except Exception:
        return None
    if len(ray_idx) == 0:
        return None
    dist = np.linalg.norm(locs - origins[ray_idx], axis=1)
    return float((dist < min_wall).sum() / samples)


def overhang_ratio(mesh: trimesh.Trimesh, angle_deg: float = 45.0, samples: int = 2000) -> float:
    """Fracción de la superficie que mira hacia abajo y no tiene nada debajo.

    Las caras apoyadas sobre otra pieza (p. ej. los píxeles sobre la placa)
    no cuentan: se tira un rayo hacia abajo y si choca enseguida, está apoyada.
    """
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return 0.0
    nz = mesh.face_normals[:, 2]
    cz = mesh.triangles_center[:, 2]
    down = np.nonzero((nz < -math.cos(math.radians(angle_deg))) & (cz > mesh.bounds[0][2] + 0.5))[0]
    if len(down) == 0:
        return 0.0
    areas = mesh.area_faces[down]
    rng = np.random.default_rng(5)
    pick = rng.choice(down, size=min(samples, len(down)), p=areas / areas.sum())
    origins = mesh.triangles_center[pick] - np.array([0, 0, 0.01])
    dirs = np.tile([0.0, 0.0, -1.0], (len(pick), 1))
    try:
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    except Exception:
        return float(areas.sum() / mesh.area)
    supported = np.zeros(len(pick), bool)
    if len(ray_idx):
        dist = origins[ray_idx, 2] - locs[:, 2]
        # chocar con una cara que mira hacia abajo = el punto ya estaba dentro de otra pieza
        inside = mesh.face_normals[tri_idx, 2] < 0
        supported[ray_idx[(dist < 0.6) | inside]] = True
    return float(areas.sum() / mesh.area * (~supported).mean())


def stability(mesh: trimesh.Trimesh, contact_mm: float = 0.6) -> dict:
    zmin = mesh.bounds[0][2]
    contact = mesh.vertices[mesh.vertices[:, 2] < zmin + contact_mm][:, :2]
    com = mesh.center_mass if mesh.is_volume else mesh.centroid
    result = {"apoyo_mm2": 0.0, "estable": False}
    if len(contact) < 3:
        return result
    try:
        hull = ConvexHull(contact)
    except QhullError:
        return result
    inside = bool(np.all(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2] <= 1e-6))
    result.update(apoyo_mm2=round(float(hull.volume), 1), estable=inside)
    return result


def analyze(mesh: trimesh.Trimesh, printer: dict | None = None, thin: str = "error") -> dict:
    """`thin`: cómo tratar las paredes finas — "error" (figuras), "aviso" (placas
    y relieves: lo fino suele ser un detalle del contorno) u "omitir" (piezas
    finas a propósito, como la litofanía o el filo del cortador)."""
    printer = printer or {}
    check_thin = thin != "omitir"
    kind = printer.get("kind", "fdm")
    min_wall = MIN_WALL_MM.get(kind, 1.0)
    bed = np.array([printer.get("bed_x", 256), printer.get("bed_y", 256), printer.get("bed_z", 256)], float)
    extents = mesh.extents if len(mesh.vertices) else np.zeros(3)
    non_manifold, holes = edge_stats(mesh)
    degenerate = int((~mesh.nondegenerate_faces()).sum()) if len(mesh.faces) else 0
    watertight = bool(mesh.is_watertight)
    volume_mm3 = float(mesh.volume) if watertight else 0.0
    area = float(mesh.area)

    overhang = overhang_ratio(mesh)
    thin_ratio = thin_wall_ratio(mesh, min_wall) if check_thin else None
    stab = stability(mesh) if len(mesh.vertices) else {"apoyo_mm2": 0.0, "estable": False}

    material = printer.get("material", "PLA")
    density = DENSITY.get(material, 1.24)
    shell_cm3 = min(volume_mm3, area * 0.8) / 1000  # ~2 perímetros de 0.4 mm
    infill_cm3 = max(volume_mm3 / 1000 - shell_cm3, 0) * 0.15
    grams = (shell_cm3 + infill_cm3) * density

    errors, warnings = [], []
    if not watertight:
        errors.append("La malla no está cerrada (no es watertight).")
    if non_manifold:
        errors.append(f"{non_manifold} aristas no manifold.")
    if watertight and volume_mm3 <= 0:
        errors.append("Volumen nulo o negativo (normales invertidas).")
    if degenerate:
        warnings.append(f"{degenerate} caras degeneradas.")
    if holes:
        warnings.append(f"{holes} agujeros abiertos.")
    if thin_ratio is not None and thin_ratio > 0.05:
        text = f"{thin_ratio:.0%} de la superficie tiene paredes de menos de {min_wall} mm."
        (errors if thin == "error" else warnings).append(text)
    fits = bool(np.all(extents <= bed + 1e-6))
    if not fits:
        warnings.append("No cabe en la cama: escálala o divídela en partes.")
    if overhang > 0.03:
        warnings.append(f"Voladizos en {overhang:.0%} de la superficie: activa soportes.")
    if not stab["estable"]:
        warnings.append("Puede caerse: agrégale una base o aplana la parte de abajo.")

    status = "error" if errors else ("aviso" if warnings else "sana")
    return {
        "estado": status,
        "errores": errors,
        "avisos": warnings,
        "metricas": {
            "cerrada": watertight,
            "volumen_cm3": round(volume_mm3 / 1000, 2),
            "area_cm2": round(area / 100, 1),
            "aristas_no_manifold": non_manifold,
            "caras_degeneradas": degenerate,
            "agujeros": holes,
            "caras": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "componentes": int(mesh.body_count) if len(mesh.faces) else 0,
        },
        "dimensiones_mm": [round(float(v), 1) for v in extents],
        "cabe_en_cama": fits,
        "paredes_finas": None if thin_ratio is None else round(thin_ratio, 3),
        "pared_minima_mm": min_wall,
        "voladizos": round(overhang, 3),
        "necesita_soportes": overhang > 0.03,
        "estabilidad": stab,
        "peso_g": round(grams, 1),
        "material": material,
    }


# --------------------------------------------------------------------- reparar

def repair(mesh: trimesh.Trimesh, min_part_ratio: float = 0.02, voxel_res: int = 256) -> tuple[trimesh.Trimesh, list[str]]:
    log: list[str] = []
    m = mesh.copy()
    m.merge_vertices()
    before = len(m.faces)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) < before:
        log.append(f"Quité {before - len(m.faces)} caras degeneradas o duplicadas.")

    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= areas.max() * min_part_ratio]
        if len(keep) < len(parts):
            log.append(f"Eliminé {len(parts) - len(keep)} fragmentos sueltos.")
        m = trimesh.util.concatenate(keep)

    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    m.fix_normals()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        if m.is_watertight:
            log.append("Tapé los agujeros.")

    non_manifold, _ = edge_stats(m)
    if not m.is_watertight or non_manifold:
        m = voxel_remesh(m, voxel_res)
        log.append(f"Rehíce la malla por vóxeles ({voxel_res}³) para dejarla cerrada.")
    if m.is_watertight and m.volume < 0:
        m.invert()
    return m, log or ["La malla ya estaba sana."]


def voxel_remesh(mesh: trimesh.Trimesh, resolution: int = 256) -> trimesh.Trimesh:
    pitch = float(mesh.extents.max()) / resolution
    vox = mesh.voxelized(pitch).fill()
    out = vox.marching_cubes
    out.apply_transform(vox.transform)
    trimesh.smoothing.filter_taubin(out, iterations=8)
    out.fix_normals()
    return out


def simplify(mesh: trimesh.Trimesh, max_faces: int = 250_000) -> trimesh.Trimesh:
    """Reduce caras (sobre todo zonas planas) sin abrir la malla; si la abre, no toca nada."""
    if len(mesh.faces) <= max_faces:
        return mesh
    target = max(max_faces, int(len(mesh.faces) * 0.2))
    try:
        out = mesh.simplify_quadric_decimation(face_count=target, aggression=7)
    except Exception:
        return mesh
    # el decimador a veces deja caras duplicadas que abren aristas
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    if mesh.is_watertight and not (out.is_watertight and out.is_volume):
        return mesh
    return out


# ------------------------------------------------------------- orientar/escalar

def y_up_to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return mesh


def place_on_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    b = mesh.bounds
    mesh.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
    return mesh


def scale_to(mesh: trimesh.Trimesh, size_mm: float, axis: str = "alto") -> trimesh.Trimesh:
    ext = mesh.extents
    ref = ext[2] if axis == "alto" else ext.max()
    if ref > 0:
        mesh.apply_scale(size_mm / ref)
    return mesh


def rotate_z(mesh: trimesh.Trimesh, degrees: float) -> trimesh.Trimesh:
    if degrees:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(degrees), [0, 0, 1]))
    return mesh


def flatten_bottom(mesh: trimesh.Trimesh, depth_mm: float) -> trimesh.Trimesh:
    """Corta una rebanada de abajo para que apoye plano en la cama."""
    if depth_mm <= 0:
        return mesh
    z = mesh.bounds[0][2] + depth_mm
    cut = to_manifold(mesh).trim_by_plane((0.0, 0.0, 1.0), float(z))
    return place_on_bed(from_manifold(cut))


def add_base(mesh: trimesh.Trimesh, kind: str = "disco", height_mm: float = 3.0, margin_mm: float = 3.0) -> trimesh.Trimesh:
    """Une un pedestal bajo el modelo (disco o rectangular)."""
    if kind not in ("disco", "rectangular"):
        return mesh
    place_on_bed(mesh)
    low = mesh.vertices[mesh.vertices[:, 2] < mesh.extents[2] * 0.12][:, :2]
    if len(low) < 3:
        low = mesh.vertices[:, :2]
    center = (low.min(axis=0) + low.max(axis=0)) / 2
    overlap = min(0.8, mesh.extents[2] * 0.02)
    if kind == "disco":
        radius = float(np.linalg.norm(low - center, axis=1).max()) + margin_mm
        base = trimesh.creation.cylinder(radius=radius, height=height_mm + overlap, sections=128)
    else:
        size = low.max(axis=0) - low.min(axis=0) + 2 * margin_mm
        base = trimesh.creation.box(extents=[size[0], size[1], height_mm + overlap])
    base.apply_translation([center[0], center[1], (height_mm + overlap) / 2 - height_mm])
    joined = to_manifold(mesh) + to_manifold(base)
    return place_on_bed(from_manifold(joined))


def split_for_bed(mesh: trimesh.Trimesh, bed: tuple[float, float, float], margin_mm: float = 4.0, gap_mm: float = 8.0) -> list[trimesh.Trimesh]:
    """Corta en partes iguales por cada eje que no quepa en la cama."""
    pieces = [to_manifold(mesh)]
    lo, hi = mesh.bounds
    for axis in range(3):
        usable = bed[axis] - margin_mm
        n = int(math.ceil((hi[axis] - lo[axis]) / usable)) if usable > 0 else 1
        if n <= 1:
            continue
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        step = (hi[axis] - lo[axis]) / n
        new = []
        for piece in pieces:
            rest = piece
            for k in range(1, n):
                offset = float(lo[axis] + step * k)
                below = rest.trim_by_plane([-v for v in normal], -offset)
                rest = rest.trim_by_plane(normal, offset)
                if not below.is_empty():
                    new.append(below)
            if not rest.is_empty():
                new.append(rest)
        pieces = new
    out = []
    x = 0.0
    for piece in pieces:
        m = place_on_bed(from_manifold(piece))
        m.apply_translation([x + m.extents[0] / 2, 0, 0])
        x += m.extents[0] + gap_mm
        out.append(m)
    if out:
        shift = -x / 2 + gap_mm / 2
        for m in out:
            m.apply_translation([shift, 0, 0])
    return out
