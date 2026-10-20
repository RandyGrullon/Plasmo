"""Dibuja la imagen de ejemplo del tutorial (plasmo/web/ejemplo.png): un gatito de juguete."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
S = 768


def shaded(mask: Image.Image, base: tuple[int, int, int], light=(0.35, 0.3)) -> Image.Image:
    """Rellena una máscara con un sombreado esférico suave (da volumen al relieve)."""
    m = np.asarray(mask, dtype=np.float32) / 255
    ys, xs = np.nonzero(m > 0.5)
    if not len(xs):
        return Image.new("RGBA", mask.size)
    cx = xs.min() + (xs.max() - xs.min()) * light[0]
    cy = ys.min() + (ys.max() - ys.min()) * light[1]
    r = max(xs.max() - xs.min(), ys.max() - ys.min())
    yy, xx = np.mgrid[: mask.height, : mask.width]
    d = np.clip(np.hypot(xx - cx, yy - cy) / r, 0, 1)
    k = 1.18 - 0.5 * d
    rgb = np.clip(np.array(base, np.float32) * k[..., None], 0, 255).astype(np.uint8)
    out = Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), "RGBA")
    return out


def shape(draw_fn) -> Image.Image:
    m = Image.new("L", (S, S), 0)
    draw_fn(ImageDraw.Draw(m))
    return m.filter(ImageFilter.GaussianBlur(1.2))


def main() -> None:
    img = Image.new("RGBA", (S, S), (247, 244, 239, 255))
    s = S / 640
    parts = [
        (lambda d: d.rounded_rectangle([228 * s, 268 * s, 412 * s, 528 * s], radius=70 * s, fill=255), (64, 126, 210)),  # cuerpo
        (lambda d: [d.rounded_rectangle([248 * s, 495 * s, 302 * s, 598 * s], radius=20 * s, fill=255),
                    d.rounded_rectangle([338 * s, 495 * s, 392 * s, 598 * s], radius=20 * s, fill=255)], (44, 76, 150)),  # piernas
        (lambda d: d.ellipse([282 * s, 330 * s, 358 * s, 452 * s], fill=255), (252, 232, 204)),  # panza
        (lambda d: [d.polygon([(232 * s, 150 * s), (248 * s, 48 * s), (306 * s, 112 * s)], fill=255),
                    d.polygon([(408 * s, 150 * s), (392 * s, 48 * s), (334 * s, 112 * s)], fill=255)], (232, 112, 56)),  # orejas
        (lambda d: d.ellipse([214 * s, 84 * s, 426 * s, 296 * s], fill=255), (238, 124, 62)),  # cabeza
    ]
    for fn, color in parts:
        layer = shaded(shape(fn), color)
        img.alpha_composite(layer)
    d = ImageDraw.Draw(img)
    for cx in (282, 358):  # ojos con brillo
        d.ellipse([(cx - 17) * s, 168 * s, (cx + 17) * s, 204 * s], fill=(28, 28, 32))
        d.ellipse([(cx - 7) * s, 174 * s, (cx + 1) * s, 184 * s], fill=(255, 255, 255))
    d.polygon([(312 * s, 214 * s), (328 * s, 214 * s), (320 * s, 225 * s)], fill=(210, 80, 90))  # nariz
    d.arc([296 * s, 214 * s, 320 * s, 240 * s], 20, 160, fill=(70, 40, 30), width=int(3 * s))
    d.arc([320 * s, 214 * s, 344 * s, 240 * s], 20, 160, fill=(70, 40, 30), width=int(3 * s))
    for cy in (226, 238):  # bigotes
        d.line([(232 * s, cy * s), (282 * s, (cy - 4) * s)], fill=(120, 60, 30), width=int(2 * s))
        d.line([(358 * s, (cy - 4) * s), (408 * s, cy * s)], fill=(120, 60, 30), width=int(2 * s))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([210 * s, 588 * s, 430 * s, 614 * s], fill=(0, 0, 0, 40))
    out = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)), img)
    final = Image.new("RGB", (S, S), (247, 244, 239))
    final.paste(out, mask=out.getchannel("A"))
    final.save(ROOT / "plasmo" / "web" / "ejemplo.png", optimize=True)
    print(ROOT / "plasmo" / "web" / "ejemplo.png")


if __name__ == "__main__":
    main()
