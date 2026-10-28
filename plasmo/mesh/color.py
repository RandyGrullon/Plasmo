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
    return best


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
