"""Sólidos cerrados (watertight) a partir de mapas de altura por píxel.

Cada píxel dentro de la máscara es una columna entre `bottom` y `top` (mm).
Las caras superior e inferior comparten vértices en las esquinas de los
píxeles y las paredes salen solo donde la máscara termina, así que el
resultado es una malla manifold lista para el slicer, sin booleanas.
"""

from __future__ import annotations

import numpy as np
import trimesh

MIN_THICKNESS_MM = 0.2


def clean_mask(mask: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """Rellena los contactos solo-por-esquina (patrones en diagonal 2x2).

    Dos píxeles que se tocan solo en una esquina generarían una arista con
    cuatro caras (no manifold); rellenar el hueco lo evita.
    """
    m = mask.astype(bool).copy()
    for _ in range(max_iter):
        a = m[:-1, :-1]
        b = m[:-1, 1:]
        c = m[1:, :-1]
        d = m[1:, 1:]
        diag1 = a & d & ~b & ~c
        diag2 = b & c & ~a & ~d
        if not (diag1.any() or diag2.any()):
            break
        fill = diag1 | diag2
        # rellena las dos celdas vacías del patrón
        m[:-1, 1:] |= fill
        m[1:, :-1] |= fill
    return m


def _corner_average(values: np.ndarray, weights: np.ndarray, wrap_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """Promedia `values` de los píxeles que tocan cada esquina (H+1, W+1)."""
    if wrap_x:
        vp = np.pad(values * weights, ((1, 1), (0, 0)))
        wp = np.pad(weights, ((1, 1), (0, 0)))
        vp = np.concatenate([vp[:, -1:], vp], axis=1)
        wp = np.concatenate([wp[:, -1:], wp], axis=1)
    else:
        vp = np.pad(values * weights, 1)
        wp = np.pad(weights, 1)
    s = vp[:-1, :-1] + vp[:-1, 1:] + vp[1:, :-1] + vp[1:, 1:]
    n = wp[:-1, :-1] + wp[:-1, 1:] + wp[1:, :-1] + wp[1:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(n > 0, s / np.maximum(n, 1e-12), 0.0)
    return avg, n


def heightfield_solid(
    top: np.ndarray,
    mask: np.ndarray | None = None,
    pixel_mm: float = 0.2,
    bottom: np.ndarray | float = 0.0,
    wrap_x: bool = False,
) -> trimesh.Trimesh:
    """Construye un sólido cerrado a partir de alturas por píxel.

    top, bottom: alturas (mm) por píxel, forma (H, W). La fila 0 de la imagen
    queda arriba (+Y). `wrap_x` une el borde izquierdo con el derecho (para
    doblarlo luego en cilindro); en ese caso la máscara debe ser completa.
    """
    top = np.asarray(top, dtype=np.float64)
    h, w = top.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    mask = clean_mask(mask)
    if wrap_x and not mask.all():
        raise ValueError("wrap_x requiere una máscara completa")
    bottom = np.broadcast_to(np.asarray(bottom, dtype=np.float64), (h, w))
    top = np.maximum(top, bottom + MIN_THICKNESS_MM)

    weights = mask.astype(np.float64)
    top_c, _ = _corner_average(top, weights, wrap_x)
    bot_c, _ = _corner_average(bottom, weights, wrap_x)

    cols = w if wrap_x else w + 1
    n_corners = (h + 1) * cols

    jj, ii = np.meshgrid(np.arange(cols), np.arange(h + 1))
    x = jj * pixel_mm
    y = (h - ii) * pixel_mm
    verts = np.concatenate(
        [
            np.stack([x, y, top_c], axis=-1).reshape(-1, 3),
            np.stack([x, y, bot_c], axis=-1).reshape(-1, 3),
        ]
    )

    def corner(r, c):
        return r * cols + (c % cols if wrap_x else c)

    rr, cc = np.nonzero(mask)
    a = corner(rr, cc)          # arriba-izquierda
    b = corner(rr, cc + 1)      # arriba-derecha
    c3 = corner(rr + 1, cc + 1)  # abajo-derecha
    d = corner(rr + 1, cc)      # abajo-izquierda
    off = n_corners

    faces = [
        np.stack([d, c3, b], axis=1),
        np.stack([d, b, a], axis=1),
        np.stack([d + off, b + off, c3 + off], axis=1),
        np.stack([d + off, a + off, b + off], axis=1),
    ]

    padded = np.pad(mask, 1)
    up_out = ~padded[rr, cc + 1]
    down_out = ~padded[rr + 2, cc + 1]
    if wrap_x:
        left_out = np.zeros_like(up_out)
        right_out = np.zeros_like(up_out)
    else:
        left_out = ~padded[rr + 1, cc]
        right_out = ~padded[rr + 1, cc + 2]

    def wall(sel, p, q):
        p, q = p[sel], q[sel]
        return [np.stack([p + off, p, q], axis=1), np.stack([p + off, q, q + off], axis=1)]

    faces += wall(up_out, a, b)
    faces += wall(down_out, c3, d)
    faces += wall(left_out, d, a)
    faces += wall(right_out, b, c3)

    mesh = trimesh.Trimesh(vertices=verts, faces=np.concatenate(faces), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def bend_cylinder(mesh: trimesh.Trimesh, width_mm: float, angle_deg: float = 360.0, radius_mm: float | None = None) -> trimesh.Trimesh:
    """Dobla un sólido plano (X = ancho, Y = alto, Z = grosor) en arco o cilindro.

    El grosor queda hacia afuera y la altura de la imagen pasa a ser el eje Z,
    para imprimir la pieza de pie (litofanía tipo lámpara).
    """
    angle = np.radians(angle_deg)
    if radius_mm is None:
        radius_mm = width_mm / angle
    v = mesh.vertices
    theta = v[:, 0] / width_mm * angle
    r = radius_mm + v[:, 2]
    out = np.column_stack([r * np.cos(theta), r * np.sin(theta), v[:, 1]])
    bent = trimesh.Trimesh(vertices=out, faces=mesh.faces.copy(), process=False)
    if angle_deg >= 359.999:
        bent.merge_vertices()
    return bent
