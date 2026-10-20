"""Dibuja el ícono de Plasmo (gota con capas de impresión) en assets/plasmo.ico."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def drop_mask(size: int) -> Image.Image:
    s = size / 64
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([10 * s, 14 * s, 54 * s, 58 * s], fill=255)
    d.polygon([(32 * s, 3 * s), (51 * s, 27 * s), (13 * s, 27 * s)], fill=255)
    return img.filter(ImageFilter.GaussianBlur(size / 160))


def render(size: int) -> Image.Image:
    big = size * 4
    grad = Image.new("RGB", (big, big))
    px = grad.load()
    stops = [(0.0, (255, 179, 71)), (0.55, (255, 122, 69)), (1.0, (229, 72, 122))]
    for y in range(big):
        for x in range(big):
            t = (x + y) / (2 * big)
            for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
                if t <= t1:
                    k = (t - t0) / (t1 - t0)
                    px[x, y] = tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
                    break
    d = ImageDraw.Draw(grad)
    for row in (30, 38, 46, 54):
        y = row * big / 64
        d.line([(0, y), (big, y)], fill=(170, 70, 50), width=max(1, big // 30))
    shine = Image.new("L", (big, big), 0)
    ImageDraw.Draw(shine).ellipse([19 * big / 64, 16 * big / 64, 29 * big / 64, 30 * big / 64], fill=90)
    grad.paste((255, 255, 255), mask=shine)
    # el color llena todo el lienzo y la gota va solo en el alfa: así no queda halo al reducir
    out = grad.convert("RGBA")
    out.putalpha(drop_mask(big))
    return out.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    target = ROOT / "assets" / "plasmo.ico"
    target.parent.mkdir(exist_ok=True)
    img = render(256)
    img.save(target, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(ROOT / "assets" / "plasmo.png")
    print(target)
