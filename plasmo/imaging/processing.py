"""Carga de imagen, recorte del sujeto (quitar fondo) y mapas de profundidad."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from plasmo.imaging import downloads

MAX_SIDE = 2048
_sessions: dict[str, object] = {}
_sessions_lock = threading.Lock()


def load_image(source: bytes | str | Path) -> Image.Image:
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def _session(key: str):
    import onnxruntime as ort

    with _sessions_lock:
        if key not in _sessions:
            path = downloads.download(key)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            _sessions[key] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return _sessions[key]


def _normalize(img: Image.Image, mean, std, size) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


# (mean, std, tamaño, aplicar sigmoide) — mismos valores que usa rembg
_SEG_PARAMS = {
    "isnet-general-use": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1024, 1024), False),
    "birefnet-lite": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024), True),
    "u2netp": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320), False),
}


def has_transparency(img: Image.Image) -> bool:
    alpha = np.asarray(img.getchannel("A"))
    return bool((alpha < 250).mean() > 0.02)


def segment(img: Image.Image, model_key: str = "isnet-general-use") -> np.ndarray:
    """Devuelve la máscara del sujeto (H, W) en [0, 1]."""
    if has_transparency(img):
        return np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
    if model_key not in _SEG_PARAMS:
        return segment_simple(img)
    mean, std, size, sigmoid = _SEG_PARAMS[model_key]
    sess = _session(model_key)
    inp = {sess.get_inputs()[0].name: _normalize(img, mean, std, size)}
    pred = sess.run(None, inp)[0][:, 0, :, :][0]
    if sigmoid:
        pred = 1.0 / (1.0 + np.exp(-pred))
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-6)
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.LANCZOS)
    return np.asarray(mask, dtype=np.float32) / 255.0


def segment_simple(img: Image.Image) -> np.ndarray:
    """Respaldo sin modelo: separa del color de fondo estimado en los bordes."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=-1)
    border_dist = np.linalg.norm(border - bg, axis=-1)
    thr = max(float(np.percentile(border_dist, 95)) * 1.5, 24.0)
    mask = dist > thr
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.gaussian_filter(mask.astype(np.float32), 1.0)


def largest_component(mask: np.ndarray, keep_ratio: float = 0.08) -> np.ndarray:
    """Se queda con la pieza más grande y las que pasen `keep_ratio` de su área."""
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= sizes.max() * keep_ratio)[0] + 1
    return np.isin(labels, keep)


def bbox(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def cutout(img: Image.Image, mask: np.ndarray, margin: float = 0.06) -> Image.Image:
    """Sujeto recortado en RGBA, ajustado a su caja con un margen."""
    rgba = img.copy()
    rgba.putalpha(Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)))
    x0, y0, x1, y1 = bbox(mask)
    pad = int(max(x1 - x0, y1 - y0) * margin)
    w, h = img.size
    return rgba.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def square_on_background(cut: Image.Image, size: int = 1024, fill: float = 0.85, color=(255, 255, 255)) -> Image.Image:
    """Centra el recorte en un lienzo cuadrado (lo que esperan los generadores 3D)."""
    canvas = Image.new("RGB", (size, size), color)
    scale = size * fill / max(cut.size)
    obj = cut.resize((max(1, int(cut.width * scale)), max(1, int(cut.height * scale))), Image.LANCZOS)
    canvas.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
    return canvas


def luminance(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def depth(img: Image.Image, model_key: str = "depth-anything-v2-small") -> np.ndarray:
    """Profundidad relativa en [0, 1] (1 = más cerca). Cae a luminancia si no hay modelo."""
    if model_key not in downloads.CATALOG:
        return luminance(img)
    sess = _session(model_key)
    size = (518, 518)
    inp = {sess.get_inputs()[0].name: _imagenet(img, size)}
    pred = sess.run(None, inp)[0]
    pred = np.squeeze(pred).astype(np.float32)
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-6)
    out = Image.fromarray((pred * 65535).astype(np.uint16)).resize(img.size, Image.BICUBIC)
    return np.asarray(out, dtype=np.float32) / 65535.0


def _imagenet(img: Image.Image, size) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def apply_curve(values: np.ndarray, curve: str = "linear", param: float = 1.0) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    if curve == "gamma":
        return v ** max(param, 1e-3)
    if curve == "s-curve":
        return v * v * (3 - 2 * v)
    return v
