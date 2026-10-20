"""Formas imprimibles que salen directo de la imagen, sin IA 3D.

Todas devuelven piezas cerradas en milímetros, con Z hacia arriba y apoyadas
en la cama (z = 0). La "Figura 3D" con IA vive en `plasmo.pipeline`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from plasmo.imaging import processing as proc
from plasmo.mesh.heightfield import bend_cylinder, heightfield_solid

MAX_PIXELS = 700_000


@dataclass
class Part:
    name: str
    mesh: trimesh.Trimesh
    color: tuple[int, int, int] = (230, 126, 80)


@dataclass
class FormResult:
    parts: list[Part]
    notes: list[str] = field(default_factory=list)


class Source:
    """La imagen del usuario con su recorte; la profundidad se calcula una sola vez."""

    def __init__(self, image: Image.Image, mask: np.ndarray, depth_model: str = "depth-anything-v2-small"):
        self.image = image
        self.mask = mask
        self.cut = proc.cutout(image, mask)
        self.cut_mask = np.asarray(self.cut.getchannel("A"), dtype=np.float32) / 255.0
        self.depth_model = depth_model
        self._depth: np.ndarray | None = None

    def relief_map(self, source: str) -> np.ndarray:
        """Altura relativa [0,1] sobre el recorte: profundidad IA o luminancia."""
        if source == "luminancia" or self.depth_model == "luminancia":
            rgb = Image.new("RGB", self.cut.size, (255, 255, 255))
            rgb.paste(self.cut, mask=self.cut.getchannel("A"))
            return proc.luminance(rgb)
        if self._depth is None:
            try:
                self._depth = proc.depth(self.cut.convert("RGB"), self.depth_model)
            except Exception:  # sin modelo o sin red: no frenamos la pieza
                self.depth_model = "luminancia"
                return self.relief_map("luminancia")
        return self._depth


# ------------------------------------------------------------------ utilidades

def _p(params: dict, key: str, default):
    value = (params or {}).get(key, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default
    return value


def fit_grid(aspect_wh: float, longest_mm: float, res_mm: float) -> tuple[int, int, float]:
    """Tamaño de la grilla (w, h) y resolución final, sin pasarse de MAX_PIXELS."""
    if aspect_wh >= 1:
        w_mm, h_mm = longest_mm, longest_mm / aspect_wh
    else:
        w_mm, h_mm = longest_mm * aspect_wh, longest_mm
    res = max(res_mm, math.sqrt(w_mm * h_mm / MAX_PIXELS))
    return max(2, round(w_mm / res)), max(2, round(h_mm / res)), res


def resample(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32))
    return np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32)


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) <= radius_px


def erode(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(mask) > radius_px


def disk(shape: tuple[int, int], cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def solid_silhouette(mask: np.ndarray) -> np.ndarray:
    m = proc.largest_component(mask > 0.5)
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def badge_mask(shape: str, w: int, h: int) -> np.ndarray:
    """Máscara de una placa: circulo, rectangulo, hexagono, escudo, estrella, corazon."""
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 1
    if shape == "circulo":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif shape == "hexagono":
        d.polygon([(cx + r * math.cos(math.radians(60 * k + 30)), cy + r * math.sin(math.radians(60 * k + 30))) for k in range(6)], fill=255)
    elif shape == "estrella":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.5
            a = math.radians(-90 + 36 * k)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=255)
    elif shape == "escudo":
        top, bottom, left, right = 1, h - 2, 1, w - 2
        mid = top + (bottom - top) * 0.55
        pts = [(left, top), (right, top), (right, mid)]
        for k in range(1, 20):
            t = k / 20
            pts.append((right - (right - cx) * t, mid + (bottom - mid) * math.sin(t * math.pi / 2)))
        pts.append((cx, bottom))
        for k in range(19, 0, -1):
            t = k / 20
            pts.append((left + (cx - left) * t, mid + (bottom - mid) * math.sin(t * math.pi / 2)))
        pts.append((left, mid))
        d.polygon(pts, fill=255)
    elif shape == "corazon":
        pts = []
        for k in range(200):
            t = 2 * math.pi * k / 200
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + x / 17 * r, cy - y / 17 * r - r * 0.05))
        d.polygon(pts, fill=255)
    else:  # rectangulo
        d.rounded_rectangle([1, 1, w - 2, h - 2], radius=min(w, h) * 0.08, fill=255)
    return np.asarray(img) > 127


def _subject_layers(src: Source, w: int, h: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Máscara del sujeto y relieve normalizado [0,1] a tamaño de grilla."""
    subj = solid_silhouette(resample(src.cut_mask, w, h))
    rel = resample(src.relief_map(_p(params, "fuente", "profundidad")), w, h)
    inside = rel[subj]
    if inside.size:
        lo, hi = np.percentile(inside, [2, 98])
        rel = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)
    if _p(params, "invertir", False):
        rel = 1 - rel
    rel = proc.apply_curve(rel, _p(params, "curva", "linear"), _p(params, "gamma", 1.0))
    sigma = _p(params, "suavizado", 1.0)
    if sigma > 0:
        rel = ndimage.gaussian_filter(rel, sigma)
    return subj, rel * subj


def _place_in(badge: np.ndarray, src: Source, params: dict, border_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Mete el sujeto centrado dentro de una placa, respetando su proporción."""
    h, w = badge.shape
    inner = erode(badge, border_px * 1.6)
    ys, xs = np.nonzero(inner)
    subj = np.zeros_like(badge)
    rel = np.zeros(badge.shape, dtype=np.float32)
    if len(xs) == 0:
        return subj, rel
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    aspect = src.cut.width / src.cut.height
    pw, ph = (bw, max(2, int(bw / aspect))) if bw / bh < aspect else (max(2, int(bh * aspect)), bh)
    pw, ph = int(pw * 0.92), int(ph * 0.92)
    s, r = _subject_layers(src, pw, ph, params)
    x0 = int(xs.min() + (bw - pw) / 2)
    y0 = int(ys.min() + (bh - ph) / 2)
    subj[y0 : y0 + ph, x0 : x0 + pw] = s
    rel[y0 : y0 + ph, x0 : x0 + pw] = r
    subj &= inner
    return subj, rel * subj


def _outline_or_badge(src: Source, params: dict, size_default: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Contorno de la pieza (silueta o placa) + sujeto + relieve, con su resolución."""
    size = _p(params, "tamano_mm", size_default)
    res_mm = _p(params, "resolucion_mm", 0.2)
    border = _p(params, "borde_mm", 2.0)
    shape = _p(params, "contorno", "silueta")
    if shape == "silueta":
        w, h, res = fit_grid(src.cut.width / src.cut.height, max(size - 2 * border, 5), res_mm)
        pad = int(math.ceil(border / res)) + 2
        subj, rel = _subject_layers(src, w, h, params)
        subj = np.pad(subj, pad)
        rel = np.pad(rel, pad)
        outline = dilate(subj, border / res)
        return outline, subj, rel, res, border
    aspect = src.cut.width / src.cut.height if shape == "rectangulo" else 1.0
    w, h, res = fit_grid(aspect, size, res_mm)
    outline = badge_mask(shape, w, h)
    subj, rel = _place_in(outline, src, params, border / res)
    return outline, subj, rel, res, border


def _rim(outline: np.ndarray, border_px: float) -> np.ndarray:
    return outline & ~erode(outline, border_px)


def _solid(top, mask, res, bottom=0.0) -> trimesh.Trimesh:
    mesh = heightfield_solid(top, mask, res, bottom=bottom)
    # centra en XY para que el slicer la reciba en el medio
    b = mesh.bounds
    mesh.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
    return mesh


# --------------------------------------------------------------------- formas

def relieve(src: Source, params: dict) -> FormResult:
    base = _p(params, "base_mm", 2.0)
    relief = _p(params, "relieve_mm", 3.0)
    outline, subj, rel, res, border = _outline_or_badge(src, params, 80.0)
    top = base + relief * rel
    if _p(params, "contorno", "silueta") != "silueta" and _p(params, "aro", True):
        top = np.where(_rim(outline, border / res), base + relief * 0.6, top)
    mesh = _solid(top, outline, res)
    notes = [f"Relieve de {relief:.1f} mm sobre base de {base:.1f} mm ({'profundidad IA' if src.depth_model != 'luminancia' else 'luminancia'})."]
    return FormResult([Part("relieve", mesh)], notes)


def llavero(src: Source, params: dict) -> FormResult:
    base = _p(params, "base_mm", 2.5)
    relief = _p(params, "relieve_mm", 1.5)
    ring_d = _p(params, "argolla_mm", 9.0)
    hole_d = _p(params, "agujero_mm", 4.5)
    p = dict(params or {})
    p.setdefault("tamano_mm", 50.0)
    outline, subj, rel, res, border = _outline_or_badge(src, p, 50.0)
    r_out, r_in = ring_d / 2 / res, hole_d / 2 / res
    pad = int(math.ceil(r_out * 2)) + 2
    outline, subj, rel = (np.pad(a, ((pad, 0), (0, 0))) for a in (outline, subj, rel))
    rows = np.nonzero(outline.any(axis=1))[0]
    top_row = rows.min()
    probe = min(top_row + int(r_out * 0.5), rows.max())
    cols = np.nonzero(outline[probe])[0]
    cx = cols.mean() if len(cols) else outline.shape[1] / 2
    cy = top_row - r_out * 0.25
    ring = disk(outline.shape, cy, cx, r_out)
    hole = disk(outline.shape, cy, cx, r_in)
    mask = (outline | ring) & ~hole
    subj &= ~dilate(hole, 1.2 / res)
    top = base + relief * rel * subj
    if _p(params, "contorno", "silueta") != "silueta" and _p(params, "aro", True):
        top = np.where(_rim(outline, border / res) & mask, base + relief * 0.7, top)
    mesh = _solid(top, mask, res)
    return FormResult([Part("llavero", mesh)], [f"Argolla de {ring_d:.0f} mm con agujero de {hole_d:.1f} mm para la cadenita."])


def iman(src: Source, params: dict) -> FormResult:
    magnet_d = _p(params, "iman_diametro_mm", 10.2)
    magnet_h = _p(params, "iman_alto_mm", 3.2)
    base = max(_p(params, "base_mm", 3.0), magnet_h + 0.8)
    relief = _p(params, "relieve_mm", 2.0)
    p = dict(params or {})
    p.setdefault("tamano_mm", 60.0)
    outline, subj, rel, res, border = _outline_or_badge(src, p, 60.0)
    dist = ndimage.distance_transform_edt(outline)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    notes = [f"Bolsillo trasero de {magnet_d:.1f} × {magnet_h:.1f} mm para el imán."]
    r_px = magnet_d / 2 / res
    if dist[cy, cx] < r_px + 1.0 / res:
        notes = ["La silueta es muy estrecha para el imán: se omitió el bolsillo; pégalo con silicona."]
        pocket = np.zeros_like(outline)
    else:
        pocket = disk(outline.shape, cy, cx, r_px)
    bottom = np.where(pocket, magnet_h, 0.0)
    top = base + relief * rel
    if _p(params, "contorno", "silueta") != "silueta" and _p(params, "aro", True):
        top = np.where(_rim(outline, border / res), base + relief * 0.6, top)
    mesh = _solid(top, outline, res, bottom=bottom)
    return FormResult([Part("iman", mesh)], notes)


def litofania(src: Source, params: dict) -> FormResult:
    form = _p(params, "forma", "plana")
    t_min = _p(params, "grosor_min_mm", 0.8)
    t_max = _p(params, "grosor_max_mm", 3.0)
    frame = _p(params, "marco_mm", 3.0)
    res_mm = _p(params, "resolucion_mm", 0.2)
    img = Image.new("RGB", src.image.size, (255, 255, 255))
    img.paste(src.image, mask=src.image.getchannel("A"))
    aspect = img.width / img.height
    if form == "cilindro":
        diameter = _p(params, "diametro_mm", 80.0)
        width = math.pi * diameter
    else:
        width = _p(params, "ancho_mm", 100.0)
    height = width / aspect
    longest = max(width, height)
    w, h, res = fit_grid(aspect, longest, res_mm)
    lum = proc.luminance(img.resize((w, h), Image.LANCZOS))
    if _p(params, "invertir", False):
        lum = 1 - lum
    lum = ndimage.gaussian_filter(lum, _p(params, "suavizado", 0.6))
    thick = t_min + (t_max - t_min) * (1 - lum)
    fp = int(round(frame / res))
    if fp > 0:
        border = np.zeros((h, w), bool)
        border[:fp, :] = border[-fp:, :] = True
        if form != "cilindro":
            border[:, :fp] = border[:, -fp:] = True
        thick = np.where(border, t_max + 0.4, thick)
    if form == "cilindro":
        flat = heightfield_solid(thick, None, res, wrap_x=True)
        mesh = bend_cylinder(flat, w * res, 360.0)
        note = f"Cilindro de {diameter:.0f} mm de diámetro para lámpara; imprímelo de pie, sin soportes."
    elif form == "curva":
        angle = _p(params, "angulo", 120.0)
        mesh = bend_cylinder(heightfield_solid(thick, None, res), w * res, angle)
        note = f"Litofanía curva de {angle:.0f}°; imprímela de pie."
    else:
        mesh = heightfield_solid(thick, None, res)
        note = "Litofanía plana: imprímela de pie o acostada al 100 % de relleno."
    mesh.fix_normals()
    b = mesh.bounds
    mesh.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
    return FormResult([Part("litofania", mesh, (245, 240, 230))], [note, "Usa filamento blanco y 100 % de relleno."])


def pixel_art(src: Source, params: dict) -> FormResult:
    longest = int(_p(params, "pixeles", 32))  # píxeles en el lado más largo
    n_colors = int(_p(params, "colores", 4))
    size = _p(params, "tamano_mm", 100.0)
    height = _p(params, "alto_mm", 2.0)
    plate = _p(params, "placa_mm", 1.2)
    with_bg = _p(params, "con_fondo", False)
    tight = src.cut if with_bg else src.cut.crop(proc.bbox(src.cut_mask))  # sin margen: el tamaño es el del sujeto
    aspect = tight.width / tight.height
    cols, rows = (longest, max(2, round(longest / aspect))) if aspect >= 1 else (max(2, round(longest * aspect)), longest)
    cell = size / longest
    small = tight.resize((cols, rows), Image.BOX)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(small.getchannel("A"), dtype=np.float32) / 255
    cells = np.ones((rows, cols), bool) if with_bg else alpha > 0.5
    if not cells.any():
        cells = np.ones((rows, cols), bool)
    colors = rgb[cells]
    k = max(1, min(n_colors, len(np.unique(colors.round(), axis=0))))
    try:
        centroids, labels = kmeans2(colors, k, minit="++", rng=7)
    except TypeError:  # SciPy < 1.15
        centroids, labels = kmeans2(colors, k, minit="++", seed=7)
    label_grid = np.full((rows, cols), -1)
    label_grid[cells] = labels
    parts: list[Part] = []
    z0 = plate if plate > 0 else 0.0
    order = np.argsort(centroids.sum(axis=1))
    for idx, ci in enumerate(order):
        m = label_grid == ci
        if not m.any():
            continue
        top = np.full(m.shape, z0 + height)
        # "split": un píxel que toca a otro del mismo color solo por la esquina no cambia de color
        mesh = heightfield_solid(top, m, cell, bottom=z0, pinch="split")
        color = tuple(int(c) for c in np.clip(centroids[ci], 0, 255))
        parts.append(Part(f"color_{idx + 1}", mesh, color))
    if plate > 0:
        plate_mask = dilate(cells, 0.5)
        parts.insert(0, Part("placa", heightfield_solid(np.full(cells.shape, plate), plate_mask, cell), (30, 30, 30)))
    allb = np.vstack([p.mesh.bounds for p in parts])
    shift = [-(allb[:, 0].min() + allb[:, 0].max()) / 2, -(allb[:, 1].min() + allb[:, 1].max()) / 2, 0]
    for p in parts:
        p.mesh.apply_translation(shift)
    notes = [f"{cols}×{rows} píxeles de {cell:.1f} mm en {len(parts) - (1 if plate > 0 else 0)} colores; cada color es un objeto del 3MF para asignarle su filamento."]
    return FormResult(parts, notes)


def cortador(src: Source, params: dict) -> FormResult:
    size = _p(params, "tamano_mm", 80.0)
    wall = _p(params, "pared_mm", 1.0)
    blade_h = _p(params, "alto_mm", 14.0)
    flange = _p(params, "pestana_mm", 4.0)
    flange_h = _p(params, "pestana_alto_mm", 2.0)
    res_mm = _p(params, "resolucion_mm", 0.2)
    w, h, res = fit_grid(src.cut.width / src.cut.height, size, res_mm)
    subj = solid_silhouette(resample(src.cut_mask, w, h))
    subj = ndimage.binary_opening(subj, iterations=max(1, int(1.5 / res)))
    pad = int(math.ceil((wall + flange) / res)) + 2
    subj = np.pad(subj, pad)
    blade = dilate(subj, wall / res) & ~subj
    outer = dilate(subj, (wall + flange) / res) & ~subj
    top = np.where(blade, blade_h, flange_h)
    mesh = _solid(top, outer, res)
    return FormResult([Part("cortador", mesh, (240, 200, 120))], [f"Filo de {wall:.1f} mm y {blade_h:.0f} mm de alto; pestaña de {flange:.0f} mm para empujar."])


FORMS: dict[str, Callable[[Source, dict], FormResult]] = {
    "relieve": relieve,
    "litofania": litofania,
    "llavero": llavero,
    "iman": iman,
    "pixel": pixel_art,
    "cortador": cortador,
}
