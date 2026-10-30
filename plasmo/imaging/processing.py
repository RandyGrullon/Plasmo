"""Carga de imagen, recorte del sujeto (quitar fondo) y mapas de profundidad.

La prioridad es la calidad, no la velocidad: el recorte combina dos modelos
con doble pasada (normal y espejo), respeta la proporción de la foto y ajusta
los bordes con un filtro guiado por los colores de la imagen.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
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
            # sin "arena": BiRefNet pide bloques enormes de una vez y falla con «bad allocation»
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            _sessions[key] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return _sessions[key]


def resize_float(arr: np.ndarray, size: tuple[int, int], resample=Image.BILINEAR) -> np.ndarray:
    return np.asarray(Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32)).resize(size, resample), dtype=np.float32)


# ------------------------------------------------------------------ recorte

# (mean, std, lado de entrada, aplicar sigmoide) — mismos valores que usa rembg
_SEG_PARAMS = {
    "isnet-general-use": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), 1024, False),
    "birefnet-lite": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1024, True),
    "u2netp": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 320, False),
}
# "auto" = máxima calidad: los dos mejores modelos combinados. BiRefNet pesa más
# porque toma el sujeto entero (IS-Net a veces se queda solo con lo más llamativo).
AUTO_MODELS = ("isnet-general-use", "birefnet-lite")
AUTO_WEIGHTS = {"isnet-general-use": 0.3, "birefnet-lite": 0.7}


def has_transparency(img: Image.Image) -> bool:
    alpha = np.asarray(img.getchannel("A"))
    return bool((alpha < 250).mean() > 0.02)


def _letterbox(img: Image.Image, side: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Encaja la foto en un cuadrado sin deformarla; el relleno imita el fondo."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    s = side / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    arr = np.asarray(rgb)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    fill = tuple(int(c) for c in np.median(border, axis=0))
    canvas = Image.new("RGB", (side, side), fill)
    x0, y0 = (side - nw) // 2, (side - nh) // 2
    canvas.paste(rgb.resize((nw, nh), Image.LANCZOS), (x0, y0))
    return canvas, (x0, y0, nw, nh)


def _predict(img: Image.Image, model_key: str, flip: bool = False, letterbox: bool = True) -> np.ndarray:
    """Probabilidad de sujeto (H, W) en [0, 1] con un modelo."""
    mean, std, side, sigmoid = _SEG_PARAMS[model_key]
    if letterbox:
        canvas, (x0, y0, nw, nh) = _letterbox(img, side)
    else:  # como rembg: estira a cuadrado
        canvas, (x0, y0, nw, nh) = img.convert("RGB").resize((side, side), Image.LANCZOS), (0, 0, side, side)
    if flip:
        canvas = ImageOps.mirror(canvas)
    arr = np.asarray(canvas, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
    sess = _session(model_key)
    pred = sess.run(None, {sess.get_inputs()[0].name: arr.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    if sigmoid:
        pred = 1.0 / (1.0 + np.exp(-pred))
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-6)
    if flip:
        pred = pred[:, ::-1]
    pred = pred[y0 : y0 + nh, x0 : x0 + nw]
    return np.clip(resize_float(pred, img.size), 0.0, 1.0)


def segment(img: Image.Image, model_key: str = "auto", tta: bool = True, refine: bool = True, letterbox: bool = True,
            objects: str = "todos") -> np.ndarray:
    """Máscara del sujeto (H, W) en [0, 1]. Ver `segment_detail`."""
    return segment_detail(img, model_key, tta, refine, letterbox, objects)[0]


def segment_detail(img: Image.Image, model_key: str = "auto", tta: bool = True, refine: bool = True,
                   letterbox: bool = True, objects: str = "todos") -> tuple[np.ndarray, dict]:
    """Máscara del sujeto y un diagnóstico (modelos usados, cuánto discrepan).

    "auto" combina IS-Net y BiRefNet con los pesos de `AUTO_WEIGHTS`; con
    `tta` cada modelo mira también la foto en espejo. Si un modelo no está,
    se usa el que haya. `objects="principal"` deja solo el protagonista.
    """
    if has_transparency(img):
        return np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0, {"modelos": ["transparencia"]}
    if model_key == "simple" or (model_key not in _SEG_PARAMS and model_key != "auto"):
        prob = segment_simple(img)
        return (refine_mask(prob, img) if refine else prob), {"modelos": ["simple"]}
    keys = AUTO_MODELS if model_key == "auto" else (model_key,)
    per_model: dict[str, np.ndarray] = {}
    errors = []
    for key in keys:
        try:
            views = [_predict(img, key, False, letterbox)]
            if tta:
                views.append(_predict(img, key, True, letterbox))
            per_model[key] = np.mean(views, axis=0)
        except Exception as exc:  # sin red o sin modelo: seguimos con el resto
            errors.append(exc)
    if not per_model:
        raise RuntimeError(f"No pude usar el modelo de recorte: {errors[0]}") if errors else RuntimeError("Sin modelos de recorte")
    if len(per_model) > 1:
        weights = np.array([AUTO_WEIGHTS[k] for k in per_model])
        prob = np.tensordot(weights / weights.sum(), np.stack(list(per_model.values())), axes=1)
    else:
        prob = next(iter(per_model.values()))
    prob = calibrate(prob)
    info = {"modelos": list(per_model), "objetos_modo": objects}
    if len(per_model) > 1:
        a, b = (m > 0.5 for m in per_model.values())
        info["discrepancia"] = round(float((a ^ b).sum() / max((a | b).sum(), 1)), 3)
    if objects == "principal":
        prob = keep_main(prob, per_model.get("isnet-general-use"))
    return (refine_mask(prob, img) if refine else prob), info


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
    mask = keep_components(mask)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.gaussian_filter(mask.astype(np.float32), 1.0)


def keep_components(mask: np.ndarray, min_fraction: float = 0.002) -> np.ndarray:
    """Quita solo las motas: conserva todo objeto con al menos `min_fraction` de la foto."""
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= max(mask.size * min_fraction, 1))[0] + 1
    if not len(keep):
        keep = [int(np.argmax(sizes)) + 1]
    return np.isin(labels, keep)


def largest_component(mask: np.ndarray, keep_ratio: float = 0.08) -> np.ndarray:
    """Se queda con la pieza más grande y las que pasen `keep_ratio` de su área."""
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= sizes.max() * keep_ratio)[0] + 1
    return np.isin(labels, keep)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return ndimage.uniform_filter(x, size=2 * r + 1, mode="reflect")


def guided_filter(guide_rgb: np.ndarray, p: np.ndarray, r: int, eps: float = 1e-3, max_side: int = 900) -> np.ndarray:
    """Filtro guiado en color (He et al.), rápido: coeficientes en baja, aplicación en alta.

    Hace que los bordes de `p` (máscara o profundidad) sigan los bordes reales
    de la foto, aunque el sujeto y el fondo tengan la misma luminancia.
    """
    h, w = p.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        I = np.stack([resize_float(guide_rgb[..., c], size) for c in range(3)], axis=-1)
        ps = resize_float(p, size)
        rs = max(1, round(r * s))
    else:
        I, ps, rs = guide_rgb, p, r
    mean_I = np.stack([_box(I[..., c], rs) for c in range(3)], axis=-1)
    mean_p = _box(ps, rs)
    cov_Ip = np.stack([_box(I[..., c] * ps, rs) for c in range(3)], axis=-1) - mean_I * mean_p[..., None]
    var = np.empty(ps.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(I[..., i] * I[..., j], rs) - mean_I[..., i] * mean_I[..., j]
            var[..., i, j] = var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mean_p - (a * mean_I).sum(-1)
    mean_a = np.stack([_box(a[..., c], rs) for c in range(3)], axis=-1)
    mean_b = _box(b, rs)
    if s < 1.0:
        mean_a = np.stack([resize_float(mean_a[..., c], (w, h)) for c in range(3)], axis=-1)
        mean_b = resize_float(mean_b, (w, h))
    return (mean_a * guide_rgb).sum(-1) + mean_b


def calibrate(prob: np.ndarray, low: float = 0.3, high: float = 0.6) -> np.ndarray:
    """Pasa la probabilidad combinada a opacidad: lo decidido como sujeto queda opaco.

    Si BiRefNet dice 1 e IS-Net 0, la mezcla 70/30 da 0.7: es sujeto, pero sin
    esto quedaría al 70 % de opacidad (cuerpo deslavado en el recorte y en la
    entrada de Hunyuan3D). Solo los bordes de verdad quedan suaves.
    """
    t = np.clip((prob - low) / (high - low), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def keep_main(prob: np.ndarray, salient: np.ndarray | None = None, min_overlap: float = 0.25) -> np.ndarray:
    """Solo el objeto principal: las piezas que el modelo de saliencia (IS-Net) marca como protagonista.

    Sirve para fotos con otras cosas alrededor (una mesa con objetos). Sin ese
    modelo, se queda con la pieza más grande.
    """
    labels, n = ndimage.label(prob > 0.5)
    if n <= 1:
        return prob
    ids = range(1, n + 1)
    keep = [k for k in ids if salient is not None and (salient[labels == k] > 0.5).mean() >= min_overlap]
    if not keep:
        keep = [int(np.argmax(ndimage.sum(np.ones_like(labels), labels, ids))) + 1]
    return np.where(np.isin(labels, keep), prob, np.minimum(prob, 0.49)).astype(np.float32)


def refine_mask(prob: np.ndarray, img: Image.Image) -> np.ndarray:
    """Limpia motas y huecos chicos y ajusta los bordes a la foto."""
    p = np.clip(prob, 0, 1).astype(np.float32)
    hard = p > 0.5
    if not hard.any():
        return p
    kept = keep_components(hard)
    p = np.where(hard & ~kept, 0.0, p)
    filled = ndimage.binary_fill_holes(kept)
    holes, n = ndimage.label(filled & ~kept)
    if n:
        sizes = ndimage.sum(np.ones_like(holes), holes, index=np.arange(1, n + 1))
        small = np.nonzero(sizes < kept.sum() * 0.0015)[0] + 1  # los huecos grandes (entre las piernas) quedan
        p = np.where(np.isin(holes, small), 1.0, p)
    h, w = p.shape
    r = max(2, round(min(h, w) / 160))
    guide = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    q = np.clip(guided_filter(guide, p, r, 1e-4), 0, 1)
    # solo se tocan los bordes: el interior seguro y el fondo seguro no cambian
    band = ndimage.binary_dilation((p > 0.03) & (p < 0.97), iterations=r) | (ndimage.binary_dilation(p > 0.5, iterations=r) & ~ndimage.binary_erosion(p > 0.5, iterations=r))
    return np.where(band, q, p).astype(np.float32)


def mask_quality(mask: np.ndarray, info: dict | None = None) -> dict:
    """Nota del recorte (0-100) con avisos legibles."""
    hard = mask > 0.5
    total = mask.size
    cover = float(hard.mean())
    labels, n = ndimage.label(keep_components(hard))
    # ancho medio del borde suave, comparado con el que deja el ajuste de bordes (radio r)
    edge = (mask > 0.15) & (mask < 0.85)
    perimeter = hard & ~ndimage.binary_erosion(hard)
    r = max(2, round(min(mask.shape) / 160))
    uncertain = float(edge.sum() / max(perimeter.sum(), 1) / (2.5 * r))
    border = np.concatenate([hard[0], hard[-1], hard[:, 0], hard[:, -1]])
    touches = float(border.mean())
    score = 100.0
    notes = []
    if cover < 0.01:
        score -= 60
        notes.append("Casi no encontré sujeto: prueba otra foto o «Usar la imagen completa».")
    elif cover > 0.85:
        score -= 40
        notes.append("El sujeto ocupa casi toda la foto: puede que el fondo no se haya separado.")
    if touches > 0.02:
        score -= min(30, touches * 300)
        notes.append("El sujeto toca el borde de la foto: puede estar cortado.")
    if uncertain > 1.0:
        score -= min(30, (uncertain - 1.0) * 30)
        notes.append("Hay bordes dudosos (pelo, transparencias o fondo parecido al sujeto).")
    if n > 1:
        extra = " Si solo quieres al protagonista, elige «Solo el principal»." if n > 2 and (info or {}).get("objetos_modo") != "principal" else ""
        notes.append(f"Detecté {n} objetos separados; se conservan todos.{extra}")
    disagreement = (info or {}).get("discrepancia")
    if disagreement is not None and disagreement > 0.15:
        # suele ser IS-Net quedándose corto: se usa lo de BiRefNet, que ve el sujeto entero
        score -= min(10, (disagreement - 0.15) * 40)
        hint = "" if (info or {}).get("objetos_modo") == "principal" else " Si se coló algo de alrededor, prueba «Solo el principal»."
        notes.append("Un modelo vio más que el otro; Plasmo se quedó con lo más completo." + hint)
    score = max(0.0, min(100.0, score))
    label = "excelente" if score >= 90 else "buena" if score >= 75 else "regular" if score >= 50 else "mala"
    return {"nota": round(score), "calidad": label, "objetos": int(n), "cobertura": round(cover, 3),
            "bordes_dudosos": round(uncertain, 3), "avisos": notes, "pixeles": total}


def decontaminate(img: Image.Image, mask: np.ndarray, solid: float = 0.9) -> Image.Image:
    """Quita el halo del fondo: cada píxel toma el color del punto sólido del sujeto más cercano."""
    rgb = np.asarray(img.convert("RGB"))
    sure = mask >= solid
    if not sure.any():
        return img.convert("RGB")
    _, (iy, ix) = ndimage.distance_transform_edt(~sure, return_indices=True)
    out = rgb[iy, ix]
    return Image.fromarray(np.where(sure[..., None], rgb, out).astype(np.uint8))


def bbox(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def cut_box(mask: np.ndarray, margin: float = 0.06) -> tuple[int, int, int, int]:
    """Caja del sujeto con margen, dentro de la foto."""
    x0, y0, x1, y1 = bbox(mask)
    pad = int(max(x1 - x0, y1 - y0) * margin)
    h, w = mask.shape
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def cutout(img: Image.Image, mask: np.ndarray, margin: float = 0.06) -> Image.Image:
    """Sujeto recortado en RGBA, sin halo de fondo, ajustado a su caja con un margen."""
    rgba = decontaminate(img, mask).convert("RGBA")
    rgba.putalpha(Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)))
    return rgba.crop(cut_box(mask, margin))


def enhance_for_ai(cut: Image.Image) -> Image.Image:
    """Realza la foto antes de mandarla al generador: contraste automático y nitidez suave."""
    alpha = cut.getchannel("A") if cut.mode == "RGBA" else None
    rgb = ImageOps.autocontrast(cut.convert("RGB"), cutoff=0.5)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb


def square_on_background(cut: Image.Image, size: int = 1024, fill: float = 0.85, color=(255, 255, 255)) -> Image.Image:
    """Centra el recorte en un lienzo cuadrado (lo que esperan los generadores 3D)."""
    canvas = Image.new("RGB", (size, size), color)
    scale = size * fill / max(cut.size)
    obj = cut.resize((max(1, int(cut.width * scale)), max(1, int(cut.height * scale))), Image.LANCZOS)
    canvas.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
    return canvas


def luminance(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


# --------------------------------------------------------------- profundidad

DEPTH_LONG_SIDE = 728  # múltiplo de 14 (el paso de parche de Depth Anything)


def _depth_input(img: Image.Image, flip: bool, long_side: int = DEPTH_LONG_SIDE) -> np.ndarray:
    w, h = img.size
    s = long_side / max(w, h)
    size = (max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14))
    rgb = img.convert("RGB").resize(size, Image.BICUBIC)
    if flip:
        rgb = ImageOps.mirror(rgb)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def depth(img: Image.Image, model_key: str = "depth-anything-v2-small", tta: bool = True, smooth: bool = True,
          long_side: int = DEPTH_LONG_SIDE) -> np.ndarray:
    """Profundidad relativa en [0, 1] (1 = más cerca), con la proporción de la foto.

    Promedia la foto y su espejo y suaviza el ruido siguiendo los bordes de la
    imagen. Cae a luminancia si no hay modelo.
    """
    if model_key not in downloads.CATALOG:
        return luminance(img)
    sess = _session(model_key)
    name = sess.get_inputs()[0].name
    preds = []
    for flip in ((False, True) if tta else (False,)):
        pred = np.squeeze(sess.run(None, {name: _depth_input(img, flip, long_side)})[0]).astype(np.float32)
        preds.append(pred[:, ::-1] if flip else pred)
    pred = np.mean(preds, axis=0)
    lo, hi = np.percentile(pred, [0.5, 99.5])
    pred = np.clip((pred - lo) / max(float(hi - lo), 1e-6), 0, 1)
    out = resize_float(pred, img.size, Image.BICUBIC)
    if smooth:
        guide = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        r = max(2, round(min(img.size) / 120))
        out = guided_filter(guide, out, r, 2e-3)
    return np.clip(out, 0.0, 1.0)


def high_detail(img: Image.Image, sigma: float = 3.0) -> np.ndarray:
    """Detalle fino de la foto (ojos, texturas) en [-1, 1], para sumarlo a la profundidad."""
    lum = luminance(img)
    hp = lum - ndimage.gaussian_filter(lum, sigma)
    scale = float(np.percentile(np.abs(hp), 99)) or 1.0
    return np.clip(hp / scale, -1.0, 1.0)


def local_contrast(values: np.ndarray, amount: float) -> np.ndarray:
    """Contraste local (CLAHE): saca relieve de las zonas planas sin quemar los extremos."""
    if amount <= 0:
        return values
    from skimage import exposure

    v = np.clip(values, 0.0, 1.0)
    eq = exposure.equalize_adapthist(v, kernel_size=max(8, min(v.shape) // 8), clip_limit=0.012).astype(np.float32)
    return (1 - amount) * v + amount * eq


def sharpen(values: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    """Máscara de enfoque sobre un mapa en [0, 1]."""
    if amount <= 0:
        return values
    v = np.asarray(values, dtype=np.float32)
    return np.clip(v + amount * (v - ndimage.gaussian_filter(v, sigma)), 0.0, 1.0)


def apply_curve(values: np.ndarray, curve: str = "linear", param: float = 1.0) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    if curve == "gamma":
        return v ** max(param, 1e-3)
    if curve == "s-curve":
        return v * v * (3 - 2 * v)
    return v
