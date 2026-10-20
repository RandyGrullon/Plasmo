"""Imagen de prueba: un muñeco de colores sobre fondo claro."""

from PIL import Image, ImageDraw


def sample_image(size: int = 640) -> Image.Image:
    img = Image.new("RGB", (size, size), (246, 244, 240))
    d = ImageDraw.Draw(img)
    s = size / 640
    d.ellipse([220 * s, 90 * s, 420 * s, 290 * s], fill=(235, 120, 60))        # cabeza
    d.polygon([(235 * s, 140 * s), (250 * s, 50 * s), (300 * s, 110 * s)], fill=(235, 120, 60))  # orejas
    d.polygon([(405 * s, 140 * s), (390 * s, 50 * s), (340 * s, 110 * s)], fill=(235, 120, 60))
    d.ellipse([270 * s, 160 * s, 300 * s, 190 * s], fill=(30, 30, 30))          # ojos
    d.ellipse([340 * s, 160 * s, 370 * s, 190 * s], fill=(30, 30, 30))
    d.rounded_rectangle([230 * s, 270 * s, 410 * s, 520 * s], radius=60 * s, fill=(60, 120, 200))  # cuerpo
    d.ellipse([280 * s, 330 * s, 360 * s, 450 * s], fill=(250, 230, 200))        # panza
    d.rectangle([250 * s, 500 * s, 300 * s, 590 * s], fill=(40, 70, 140))        # piernas
    d.rectangle([340 * s, 500 * s, 390 * s, 590 * s], fill=(40, 70, 140))
    return img
