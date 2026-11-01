"""Colores de la foto sobre la figura 3D (para la vista previa y el OBJ a color).

La figura de Hunyuan3D mira al frente (eje -Y una vez parada sobre la cama),
así que se proyecta la foto de frente: el lado que mira a la cámara recibe la
imagen nítida y lo que mira hacia atrás un promedio difuminado (la espalda de
una cabeza queda del color de la cabeza, sin los ojos pintados detrás).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from plasmo.imaging import processing as proc

BASE_COLOR = (70, 74, 82)


@dataclass
class Frame:
    """Cómo se mapea la figura (X, Z) a la foto (px, py), con el ajuste fino aplicado."""

    xmin: float
    xmax: float
    zmin: float
    zmax: float
    box: tuple[float, float, float, float]  # caja del sujeto en la foto

    def to_pixels(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (x - self.xmin) / max(self.xmax - self.xmin, 1e-9)
        t = (self.zmax - z) / max(self.zmax - self.zmin, 1e-9)
        x0, y0, x1, y1 = self.box
        return x0 + u * (x1 - x0), y0 + t * (y1 - y0)


def _bilinear(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    px = np.clip(px, 0, w - 1.001)
    py = np.clip(py, 0, h - 1.001)
    x0, y0 = px.astype(int), py.astype(int)
    fx, fy = (px - x0)[:, None], (py - y0)[:, None]
    c00, c10 = img[y0, x0], img[y0, x0 + 1]
    c01, c11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def fit_frame(vertices: np.ndarray, alpha: np.ndarray, grid: int = 96, samples: int = 20000) -> Frame:
    """Alinea la silueta frontal de la figura con la del recorte (escala y corrimiento finos)."""
    return fit_score(vertices, alpha, grid, samples)[0]


def fit_score(vertices: np.ndarray, alpha: np.ndarray, grid: int = 96, samples: int = 20000) -> tuple[Frame, float]:
    """Como `fit_frame`, y además cuánto se parece la silueta frontal a la de la foto (IoU 0–1)."""
    x, z = vertices[:, 0], vertices[:, 2]
    base = Frame(float(x.min()), float(x.max()), float(z.min()), float(z.max()), tuple(float(v) for v in proc.bbox(alpha)))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(vertices), size=min(samples, len(vertices)), replace=False)
    h, w = alpha.shape
    target = proc.resize_float((alpha > 0.5).astype(np.float32), (grid, grid)) > 0.5
    best, best_iou = base, -1.0
    x0, y0, x1, y1 = base.box
    bw, bh = x1 - x0, y1 - y0
    for s in (0.96, 0.98, 1.0, 1.02, 1.04):
        for dx in (-0.03, -0.015, 0.0, 0.015, 0.03):
            for dy in (-0.03, -0.015, 0.0, 0.015, 0.03):
                cx, cy = (x0 + x1) / 2 + dx * bw, (y0 + y1) / 2 + dy * bh
                box = (cx - bw * s / 2, cy - bh * s / 2, cx + bw * s / 2, cy + bh * s / 2)
                fr = Frame(base.xmin, base.xmax, base.zmin, base.zmax, box)
                px, py = fr.to_pixels(x[pick], z[pick])
                gx = np.clip((px / w * grid).astype(int), 0, grid - 1)
                gy = np.clip((py / h * grid).astype(int), 0, grid - 1)
                occ = np.zeros((grid, grid), bool)
                occ[gy, gx] = True
                occ = ndimage.binary_closing(occ, iterations=1)
                iou = (occ & target).sum() / max((occ | target).sum(), 1)
                if iou > best_iou:
                    best, best_iou = fr, iou
    return best, float(best_iou)


def _region_colors(img: Image.Image, small: int = 72, size: int = 23) -> np.ndarray:
    """Color de cada zona sin los rasgos chicos (ojos, boca): mediana en baja resolución.

    Es lo que recibe la espalda: el promedio dejaría la sombra de los ojos.
    """
    w, h = img.size
    s = small / max(w, h)
    lowres = np.asarray(img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), dtype=np.float32)
    med = ndimage.median_filter(lowres, size=(size, size, 1), mode="nearest")
    med = np.stack([ndimage.gaussian_filter(med[..., c], 1.5) for c in range(3)], axis=-1)
    return np.stack([proc.resize_float(med[..., c], (w, h)) for c in range(3)], axis=-1)


def project_colors(mesh: trimesh.Trimesh, cut: Image.Image, frame: Frame, base_top: float | None = None) -> np.ndarray:
    """Color RGB (N, 3) uint8 para cada vértice de `mesh`."""
    rgba = cut.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    # sin halo: lo transparente toma el color del sujeto más cercano (si no, el difuminado se oscurece)
    clean = proc.decontaminate(rgba, alpha, solid=0.5)
    rgb = np.asarray(clean, dtype=np.float32)
    soft = _region_colors(clean)
    v = mesh.vertices
    px, py = frame.to_pixels(v[:, 0], v[:, 2])
    sharp = _bilinear(rgb, px, py)
    blurred = _bilinear(soft, px, py)
    facing = -mesh.vertex_normals[:, 1]  # 1 = mira a la cámara
    t = np.clip((facing + 0.15) / 0.55, 0, 1)
    wgt = (t * t * (3 - 2 * t))[:, None]
    colors = sharp * wgt + blurred * (1 - wgt)
    if base_top is not None:
        colors[v[:, 2] < base_top] = BASE_COLOR
    return np.clip(colors, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- IDOL: giro y colores propios

def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Colores lineales (los COLOR_0 del GLB de ComfyUI) → sRGB 0–255, como los de la foto."""
    c = np.clip(np.asarray(rgb, dtype=np.float32)[..., :3] / 255.0, 0, 1)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(s * 255 + 0.5, 0, 255).astype(np.uint8)


def _by_height(mask: np.ndarray, grid: int) -> np.ndarray:
    """Silueta escalada a `grid` de alto y centrada en un lienzo de 2·grid de ancho (conserva la proporción)."""
    x0, y0, x1, y1 = proc.bbox(mask.astype(np.float32))
    crop = mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1].astype(np.float32)
    h, w = crop.shape
    width = max(1, min(2 * grid, round(grid * w / max(h, 1))))
    small = proc.resize_float(crop, (width, grid)) > 0.5
    canvas = np.zeros((grid, 2 * grid), bool)
    left = (2 * grid - width) // 2
    canvas[:, left:left + width] = small
    return canvas


def _projected(xz: np.ndarray, grid: int) -> np.ndarray:
    """Silueta de puntos X-Z vistos desde -Y, con la misma normalización que `_by_height`."""
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    scale = (grid - 1) / max(hi[1] - lo[1], 1e-9)
    cx = (lo[0] + hi[0]) / 2
    gx = np.clip(np.round((xz[:, 0] - cx) * scale + grid).astype(int), 0, 2 * grid - 1)
    gy = np.clip(np.round((hi[1] - xz[:, 1]) * scale).astype(int), 0, grid - 1)
    occ = np.zeros((grid, 2 * grid), bool)
    occ[gy, gx] = True
    occ = ndimage.binary_closing(occ, iterations=2)
    return ndimage.binary_fill_holes(occ)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _rot_z(deg: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])[:3, :3]


def _front_color_error(mesh: trimesh.Trimesh, colors: np.ndarray, cut: Image.Image, yaw: float) -> float:
    """Qué tan distinto es el frente de la figura (girada `yaw`) de los colores de la foto."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1]))
    rgba = cut.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    frame = fit_frame(m.vertices, alpha)
    front = -m.vertex_normals[:, 1] > 0.5
    if front.sum() < 20:
        return 255.0
    px, py = frame.to_pixels(m.vertices[front, 0], m.vertices[front, 2])
    inside = _bilinear(alpha[..., None], px, py)[:, 0] > 0.5
    if inside.sum() < 20:
        return 255.0
    photo = _bilinear(np.asarray(rgba.convert("RGB"), dtype=np.float32), px[inside], py[inside])
    return float(np.abs(photo - colors[front][inside].astype(np.float32)).mean())


def orient_to_photo(mesh: trimesh.Trimesh, colors: np.ndarray | None, cut: Image.Image, grid: int = 72,
                    samples: int = 30000) -> tuple[float, float]:
    """Giro sobre Z (grados) que deja la figura mirando a la cámara (-Y) como en la foto, y el parecido (IoU).

    IDOL decide solo hacia dónde mira el objeto, así que se prueban los giros y gana el
    que calca la silueta del recorte. El frente y la espalda dan siluetas en espejo: si
    se parecen igual, deciden los colores del frente contra los de la foto.
    """
    alpha = np.asarray(cut.getchannel("A"), dtype=np.float32) / 255.0 > 0.5
    if not alpha.any() or len(mesh.vertices) < 4:
        return 0.0, 0.0
    target = _by_height(alpha, grid)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(mesh.vertices), size=min(samples, len(mesh.vertices)), replace=False)
    pts = mesh.vertices[pick] - np.r_[mesh.vertices[pick, :2].mean(axis=0), 0.0]

    def score(deg: float) -> float:
        p = pts @ _rot_z(deg).T
        return _iou(_projected(p[:, [0, 2]], grid), target)

    step = 3.0
    angles = np.arange(0.0, 360.0, step)
    scores = np.array([score(deg) for deg in angles])
    n = len(angles)

    def plateau(i0: int) -> tuple[float, float]:
        """Centro del tramo de giros casi igual de buenos: cerca del óptimo la silueta casi no cambia."""
        best, left, right = scores[i0], i0, i0
        while scores[(left - 1) % n] >= best - 0.004 and i0 - left < n // 4:
            left -= 1
        while scores[(right + 1) % n] >= best - 0.004 and right - i0 < n // 4:
            right += 1
        return float((angles[i0] + (left + right - 2 * i0) / 2 * step) % 360.0), float(best)

    a, iou_a = plateau(int(np.argmax(scores)))
    near_mirror = np.abs((angles - a - 180.0 + 180.0) % 360.0 - 180.0) <= 30.0  # la espalda da la silueta en espejo
    idx = np.nonzero(near_mirror)[0]
    b, iou_b = plateau(int(idx[np.argmax(scores[idx])]))
    if colors is None:
        return (a, iou_a) if iou_a >= iou_b else (b, iou_b)
    if abs(iou_a - iou_b) < 0.04 and _front_color_error(mesh, colors, cut, b) < _front_color_error(mesh, colors, cut, a):
        a, iou_a = b, iou_b
    elif iou_b > iou_a:
        a, iou_a = b, iou_b
    # ajuste fino: cerca del frente la silueta casi no cambia, pero los rasgos (ojos, manchas) sí
    near = [(deg % 360.0, score(deg % 360.0)) for deg in np.arange(a - 24.0, a + 24.5, 3.0)]
    near = [(deg, s) for deg, s in near if s >= iou_a - 0.02]
    errors = {deg: _front_color_error(mesh, colors, cut, deg) for deg, _ in near}
    coarse_best = min(errors, key=errors.get)
    for deg in np.arange(coarse_best - 2.0, coarse_best + 2.5, 1.0):  # y de a 1°
        errors.setdefault(deg % 360.0, _front_color_error(mesh, colors, cut, deg % 360.0))
    best = min(errors, key=errors.get)
    return float(best), score(best)


def _normalized(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return (v - center) / max(float(hi[2] - lo[2]), 1e-9)


def transfer_colors(points: np.ndarray, colors: np.ndarray, mesh: trimesh.Trimesh, base_top: float | None = None) -> np.ndarray:
    """Lleva los colores propios del modelo (IDOL, todo el contorno) a la malla final.

    La malla final es la misma figura reparada, escalada, aplanada y quizá con base:
    se comparan las dos con la altura normalizada y cada vértice toma el color del
    punto original más cercano.
    """
    from scipy.spatial import cKDTree

    v = mesh.vertices
    body = np.ones(len(v), bool) if base_top is None else v[:, 2] >= base_top
    ref = v[body] if body.any() else v
    idx = cKDTree(_normalized(points, points)).query(_normalized(v, ref))[1]
    out = np.asarray(colors, dtype=np.uint8)[idx].copy()
    out[~body] = BASE_COLOR
    return out
