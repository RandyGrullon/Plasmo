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


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Rellena los contactos solo-por-esquina (patrones en diagonal 2x2).

    Dos píxeles que se tocan solo en una esquina generarían una arista con
    cuatro caras (no manifold); rellenar el hueco lo evita. Cada vuelta solo
    agrega celdas, así que siempre converge; los patrones finos (ojos,
    bigotes) pueden pedir muchas vueltas.
    """
    m = mask.astype(bool).copy()
    for _ in range(m.size):
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
    pinch: str = "fill",
) -> trimesh.Trimesh:
    """Construye un sólido cerrado a partir de alturas por píxel.

    top, bottom: alturas (mm) por píxel, forma (H, W). La fila 0 de la imagen
    queda arriba (+Y). `wrap_x` une el borde izquierdo con el derecho (para
    doblarlo luego en cilindro); en ese caso la máscara debe ser completa.
    `pinch` decide qué hacer con dos píxeles que se tocan solo por una esquina:
    "fill" rellena la esquina (pieza más firme) y "split" duplica ese vértice
    para que cada lado quede cerrado sin cambiar la forma (pixel art).
    """
    top = np.asarray(top, dtype=np.float64)
    h, w = top.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    split = pinch == "split" and not wrap_x
    mask = mask.astype(bool) if split else clean_mask(mask)
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
    off = n_corners
    # índices arriba (t) y abajo (b) de las cuatro esquinas de cada píxel
    a_t = corner(rr, cc)            # arriba-izquierda
    b_t = corner(rr, cc + 1)        # arriba-derecha
    c_t = corner(rr + 1, cc + 1)    # abajo-derecha
    d_t = corner(rr + 1, cc)        # abajo-izquierda
    a_b, b_b, c_b, d_b = a_t + off, b_t + off, c_t + off, d_t + off

    if split:
        pix = np.full((h, w), -1, dtype=np.int64)
        pix[rr, cc] = np.arange(len(rr))
        A, B, C, D = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
        extra = []
        n_v = len(verts)
        # (i,j) y (i+1,j+1) comparten la esquina (i+1,j+1): el segundo usa una copia
        # (i,j+1) e (i+1,j) comparten la misma esquina: el segundo usa una copia
        for keep, other, field in (
            (np.nonzero(A & D & ~B & ~C), (1, 1), "a"),
            (np.nonzero(B & C & ~A & ~D), (1, 0), "b"),
        ):
            i, j = keep
            if not len(i):
                continue
            ki, kj = (i, j) if field == "a" else (i, j + 1)
            oi, oj = i + other[0], j + other[1]
            ci = corner(i + 1, j + 1)
            verts[ci, 2] = top[ki, kj]
            verts[ci + off, 2] = bottom[ki, kj]
            dup_t = n_v + np.arange(len(i))
            dup_b = dup_t + len(i)
            extra.append(np.column_stack([verts[ci, 0], verts[ci, 1], top[oi, oj]]))
            extra.append(np.column_stack([verts[ci, 0], verts[ci, 1], bottom[oi, oj]]))
            n_v += 2 * len(i)
            k = pix[oi, oj]
            if field == "a":
                a_t[k], a_b[k] = dup_t, dup_b
            else:
                b_t[k], b_b[k] = dup_t, dup_b
        if extra:
            verts = np.concatenate([verts, *extra])

    faces = [
        np.stack([d_t, c_t, b_t], axis=1),
        np.stack([d_t, b_t, a_t], axis=1),
        np.stack([d_b, b_b, c_b], axis=1),
        np.stack([d_b, a_b, b_b], axis=1),
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

    def wall(sel, pt, pb, qt, qb):
        pt, pb, qt, qb = pt[sel], pb[sel], qt[sel], qb[sel]
        return [np.stack([pb, pt, qt], axis=1), np.stack([pb, qt, qb], axis=1)]

    faces += wall(up_out, a_t, a_b, b_t, b_b)
    faces += wall(down_out, c_t, c_b, d_t, d_b)
    faces += wall(left_out, d_t, d_b, a_t, a_b)
    faces += wall(right_out, b_t, b_b, c_t, c_b)

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
