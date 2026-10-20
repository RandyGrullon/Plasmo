"""Genera el logo de Plasmo desde una sola geometría.

La marca es una gota de plasma "rebanada" en capas, como la ve un slicer:
discos apilados en perspectiva con degradado de calor y una chispa arriba.

Salidas:
  plasmo/web/logo.svg     marca (favicon, barra superior, animaciones)
  assets/plasmo-app.svg   ícono de app (marca sobre baldosa oscura)
  assets/plasmo.png       256 px para el README
  assets/plasmo.ico       ícono del .exe (16–256 px), renderizado con Edge headless
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# (centro y, radio x) de cada capa, de arriba hacia abajo, en un lienzo de 64
LAYERS = [(15.0, 2.8), (19.9, 6.6), (24.8, 10.2), (29.7, 13.4), (34.6, 15.9), (39.5, 17.4), (44.4, 17.6), (49.3, 15.9), (54.0, 11.4)]
THICK = 2.4
SQUASH = 0.34  # alto/ancho de la elipse (perspectiva)
STOPS = [(0.0, (255, 214, 102)), (0.35, (255, 138, 76)), (0.7, (240, 80, 122)), (1.0, (156, 72, 220))]


def mix(t: float) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return tuple(round(a + (b - a) * k) for a, b in zip(c0, c1))
    return STOPS[-1][1]


def hexc(c, k: float = 1.0) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * k))) for v in c)


def mark(cx: float = 32.0, dy: float = 0.0, scale: float = 1.0, ids: str = "p") -> tuple[str, str]:
    """Devuelve (defs, cuerpo) de la marca. Cada capa es un <g class="capa">."""
    defs = [
        f'<radialGradient id="{ids}-spark" cx="50%" cy="50%" r="50%">'
        '<stop offset="0" stop-color="#fffbe8"/><stop offset="0.35" stop-color="#ffe08a"/>'
        '<stop offset="1" stop-color="#ffb347" stop-opacity="0"/></radialGradient>'
    ]
    body = []
    n = len(LAYERS)
    for i, (cy, rx) in enumerate(reversed(LAYERS)):  # de abajo hacia arriba (pintor)
        idx = n - 1 - i
        t = idx / (n - 1)
        col = mix(t)
        cy = cy * scale + dy
        rx = rx * scale
        ry = rx * SQUASH
        th = THICK * scale
        gid = f"{ids}-top{idx}"
        defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{hexc(col, 1.12)}"/><stop offset="1" stop-color="{hexc(col, 0.92)}"/></linearGradient>'
        )
        side = (
            f"M{cx - rx:.2f},{cy:.2f} A{rx:.2f},{ry:.2f} 0 0 0 {cx + rx:.2f},{cy:.2f} "
            f"L{cx + rx:.2f},{cy + th:.2f} A{rx:.2f},{ry:.2f} 0 0 1 {cx - rx:.2f},{cy + th:.2f} Z"
        )
        body.append(
            f'<g class="capa" style="--i:{idx}">'
            f'<path d="{side}" fill="{hexc(col, 0.62)}"/>'
            f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" fill="url(#{gid})"/>'
            f'<ellipse cx="{cx - rx * 0.28:.2f}" cy="{cy - ry * 0.32:.2f}" rx="{rx * 0.42:.2f}" ry="{ry * 0.34:.2f}" fill="#fff" fill-opacity="0.22"/>'
            "</g>"
        )
    sy = LAYERS[0][0] * scale + dy - 5.4 * scale
    body.append(
        f'<g class="chispa"><circle cx="{cx}" cy="{sy:.2f}" r="{5.4 * scale:.2f}" fill="url(#{ids}-spark)"/>'
        f'<circle cx="{cx}" cy="{sy:.2f}" r="{1.6 * scale:.2f}" fill="#fffdf2"/></g>'
    )
    return "".join(defs), "".join(body)


def logo_svg() -> str:
    defs, body = mark()
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Plasmo">'
        f"<defs>{defs}</defs>{body}</svg>\n"
    )


def app_svg() -> str:
    defs, body = mark(dy=1.4, scale=0.86, cx=32, ids="a")
    # centra la marca escalada en la baldosa
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        "<defs>"
        '<linearGradient id="tile" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1f2534"/><stop offset="1" stop-color="#0d1017"/></linearGradient>'
        '<radialGradient id="halo" cx="50%" cy="62%" r="55%"><stop offset="0" stop-color="#ff7a45" stop-opacity="0.32"/><stop offset="1" stop-color="#ff7a45" stop-opacity="0"/></radialGradient>'
        f"{defs}</defs>"
        '<rect x="1" y="1" width="62" height="62" rx="15" fill="url(#tile)"/>'
        '<rect x="1" y="1" width="62" height="62" rx="15" fill="url(#halo)"/>'
        '<rect x="1.5" y="1.5" width="61" height="61" rx="14.5" fill="none" stroke="#ffffff" stroke-opacity="0.07"/>'
        f'<g transform="translate(4.5 3.2)">{body}</g>'
        "</svg>\n"
    )


def find_edge() -> str | None:
    for p in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")


def render_png(svg: str, size: int, out: Path) -> None:
    """Renderiza el SVG con Edge/Chrome headless y fondo transparente."""
    browser = find_edge()
    if not browser:
        raise SystemExit("Necesito Edge o Chrome para renderizar el ícono.")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "icono.html"
        page.write_text(
            f'<!doctype html><html><body style="margin:0;background:transparent">'
            f'<div style="width:{size}px;height:{size}px">{svg.replace("<svg ", f"<svg width=\"{size}\" height=\"{size}\" ", 1)}</div></body></html>',
            encoding="utf-8",
        )
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}\\perfil",
             "--default-background-color=00000000", f"--window-size={size},{size}", f"--screenshot={out}", page.as_uri()],
            check=True, capture_output=True, timeout=60,
        )


if __name__ == "__main__":
    (ROOT / "plasmo" / "web" / "logo.svg").write_text(logo_svg(), encoding="utf-8")
    app = app_svg()
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "assets" / "plasmo-app.svg").write_text(app, encoding="utf-8")
    big = ROOT / "assets" / "plasmo.png"
    render_png(app, 256, big)
    img = Image.open(big).convert("RGBA")
    img.save(ROOT / "assets" / "plasmo.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("logo.svg, plasmo-app.svg, plasmo.png, plasmo.ico")
