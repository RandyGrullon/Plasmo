"""Banco de pruebas del recorte: fotos sintéticas difíciles con su máscara real.

Mide IoU (cuánto coincide la máscara), F1 de borde (qué tan fino queda el
contorno, tolerancia 2 px) y objetos perdidos, para comparar métodos:

    uv run python tools/banco_recorte.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plasmo.imaging import processing as proc  # noqa: E402

S = 640


def _shade(mask: Image.Image, color) -> Image.Image:
    m = np.asarray(mask, np.float32) / 255
    ys, xs = np.nonzero(m > 0.5)
    yy, xx = np.mgrid[: S, : S]
    cx, cy = xs.min() + (xs.max() - xs.min()) * 0.35, ys.min() + (ys.max() - ys.min()) * 0.3
    k = 1.15 - 0.5 * np.clip(np.hypot(xx - cx, yy - cy) / max(np.ptp(xs), np.ptp(ys), 1), 0, 1)
    rgb = np.clip(np.array(color, np.float32) * k[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), "RGBA")


def _layer(draw_fn, color) -> Image.Image:
    m = Image.new("L", (S, S), 0)
    draw_fn(ImageDraw.Draw(m))
    return _shade(m.filter(ImageFilter.GaussianBlur(0.8)), color)


def subject(kind: str) -> Image.Image:
    """Sujeto en RGBA con fondo transparente (el alfa es la verdad)."""
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if kind in ("gato", "dos"):
        out.alpha_composite(_layer(lambda d: d.rounded_rectangle([230, 270, 410, 520], radius=60, fill=255), (60, 120, 200)))
        out.alpha_composite(_layer(lambda d: [d.rectangle([250, 500, 300, 590], fill=255), d.rectangle([340, 500, 390, 590], fill=255)], (40, 70, 140)))
        out.alpha_composite(_layer(lambda d: [d.polygon([(235, 140), (250, 50), (300, 110)], fill=255), d.polygon([(405, 140), (390, 50), (340, 110)], fill=255),
                                              d.ellipse([220, 90, 420, 290], fill=255)], (235, 120, 60)))
        if kind == "dos":
            out.alpha_composite(_layer(lambda d: d.ellipse([470, 450, 590, 570], fill=255), (210, 40, 60)))
    elif kind == "robot":  # partes finas: antenas y brazos delgados
        out.alpha_composite(_layer(lambda d: [d.rectangle([250, 200, 390, 470], fill=255), d.rectangle([270, 120, 370, 200], fill=255),
                                              d.line([(300, 120), (280, 50)], fill=255, width=5), d.line([(340, 120), (365, 45)], fill=255, width=5),
                                              d.line([(250, 260), (160, 380)], fill=255, width=9), d.line([(390, 260), (480, 380)], fill=255, width=9),
                                              d.rectangle([270, 470, 300, 590], fill=255), d.rectangle([340, 470, 370, 590], fill=255)], (150, 160, 175)))
    elif kind == "taza":  # agujero real (el asa) que no se debe rellenar
        out.alpha_composite(_layer(lambda d: [d.rounded_rectangle([200, 220, 400, 520], radius=30, fill=255),
                                              d.ellipse([360, 280, 500, 440], fill=255)], (240, 240, 235)))
        hole = Image.new("L", (S, S), 0)
        ImageDraw.Draw(hole).ellipse([395, 315, 465, 405], fill=255)
        a = np.asarray(out.getchannel("A")).copy()
        a[np.asarray(hole) > 0] = 0
        out.putalpha(Image.fromarray(a))
    return out


def background(kind: str, rng) -> Image.Image:
    yy, xx = np.mgrid[: S, : S].astype(np.float32) / S
    if kind == "liso":
        base = np.full((S, S, 3), 244, np.float32)
    elif kind == "degradado":
        base = np.stack([200 + 40 * xx, 190 + 30 * yy, 170 + 50 * (1 - xx)], -1)
    elif kind == "parecido":  # fondo del mismo tono que el sujeto
        base = np.stack([215 + 20 * yy, 150 + 20 * xx, 100 + 10 * yy], -1)
    elif kind == "oscuro":
        base = np.stack([40 + 30 * xx, 38 + 25 * yy, 45 + 20 * xx], -1)
    else:  # recargado: mesa con objetos y textura
        base = np.stack([150 + 60 * yy, 120 + 50 * yy, 90 + 40 * xx], -1)
        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(14):
            x, y, r = rng.integers(0, S, 2).tolist() + [int(rng.integers(15, 60))]
            d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c) for c in rng.integers(40, 230, 3)))
        base = np.asarray(img, np.float32)
    base = base + rng.normal(0, 6, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def make_case(subj: str, bg: str, seed: int = 3) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    obj = subject(subj)
    alpha = np.asarray(obj.getchannel("A"))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sh = Image.fromarray((alpha > 128).astype(np.uint8) * 90).filter(ImageFilter.GaussianBlur(10))
    shadow.putalpha(sh)
    img = background(bg, rng).convert("RGBA")
    img.alpha_composite(shadow, (12, 14))
    img.alpha_composite(obj)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=72)  # como una foto de celular comprimida
    return proc.load_image(buf.getvalue()), alpha > 127


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    def edges(m):
        return m & ~ndimage.binary_erosion(m)
    pe, ge = edges(pred), edges(gt)
    if not pe.any() or not ge.any():
        return 0.0
    dg = ndimage.distance_transform_edt(~ge)
    dp = ndimage.distance_transform_edt(~pe)
    precision = float((dg[pe] <= tol).mean())
    recall = float((dp[ge] <= tol).mean())
    return 2 * precision * recall / max(precision + recall, 1e-9)


def lost_objects(pred: np.ndarray, gt: np.ndarray) -> int:
    labels, n = ndimage.label(gt)
    return sum(1 for i in range(1, n + 1) if (pred[labels == i]).mean() < 0.5)


SUBJECTS = ["gato", "robot", "taza", "dos"]
BACKGROUNDS = ["liso", "degradado", "parecido", "oscuro", "recargado"]
CACHE = Path(tempfile.gettempdir()) / "plasmo-banco"


def predictions(img: Image.Image, key: str) -> dict[str, np.ndarray]:
    """Cada modelo corre una vez por foto; se guarda para comparar fusiones al instante."""
    path = CACHE / f"{key}.npz"
    if path.exists():
        return dict(np.load(path))
    out = {
        "simple": proc.segment_simple(img),
        "isnet_viejo": proc._predict(img, "isnet-general-use", False, letterbox=False),
        "isnet": proc._predict(img, "isnet-general-use", False),
        "isnet_espejo": proc._predict(img, "isnet-general-use", True),
        "biref": proc._predict(img, "birefnet-lite", False),
        "biref_espejo": proc._predict(img, "birefnet-lite", True),
    }
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def methods(p: dict[str, np.ndarray], img: Image.Image) -> dict[str, np.ndarray]:
    """Lo de antes contra lo de ahora (misma fusión que `processing.segment_detail`)."""
    isnet = (p["isnet"] + p["isnet_espejo"]) / 2
    biref = (p["biref"] + p["biref_espejo"]) / 2
    w = proc.AUTO_WEIGHTS
    fused = proc.calibrate((w["isnet-general-use"] * isnet + w["birefnet-lite"] * biref) / (w["isnet-general-use"] + w["birefnet-lite"]))
    return {
        "por color (antes)": proc.largest_component(p["simple"] > 0.5).astype(np.float32),
        "IS-Net (antes)": p["isnet_viejo"],
        "máxima calidad": proc.refine_mask(fused, img),
        "solo el principal": proc.refine_mask(proc.keep_main(fused, isnet), img),
    }


def example_case() -> tuple[Image.Image, np.ndarray]:
    """La imagen de ejemplo del tutorial: fondo liso, así que el corte por color es la verdad."""
    img = proc.load_image(Path(__file__).resolve().parents[1] / "plasmo" / "web" / "ejemplo.png")
    return img, proc.segment_simple(img) > 0.5


def main() -> None:
    results: dict[str, dict[str, list]] = {"limpias": {}, "mesa recargada": {}, "gatito de ejemplo": {}}
    t0 = time.time()
    cases = [(s, b) for s in SUBJECTS for b in BACKGROUNDS] + [("ejemplo", "tutorial")]
    for s, b in cases:
        img, gt = example_case() if s == "ejemplo" else make_case(s, b)
        group = "gatito de ejemplo" if s == "ejemplo" else "mesa recargada" if b == "recargado" else "limpias"
        preds = predictions(img, f"{s}-{b}")
        for name, prob in methods(preds, img).items():
            pred = prob > 0.5
            iou = (pred & gt).sum() / max((pred | gt).sum(), 1)
            results[group].setdefault(name, []).append((iou, boundary_f1(pred, gt), lost_objects(pred, gt)))
        print(f"{s}-{b} listo ({time.time() - t0:.0f} s)", flush=True)
    for group, by_method in results.items():
        n = len(next(iter(by_method.values())))
        print(f"\n{group} ({n} foto{'s' if n > 1 else ''})")
        print(f"  {'método':20s} {'IoU':>6s} {'peor':>6s} {'borde F1':>9s} {'objetos perdidos':>17s}")
        for name, rows in by_method.items():
            a = np.array(rows)
            print(f"  {name:20s} {a[:, 0].mean():6.3f} {a[:, 0].min():6.3f} {a[:, 1].mean():9.3f} {int(a[:, 2].sum()):17d}")


if __name__ == "__main__":
    main()
