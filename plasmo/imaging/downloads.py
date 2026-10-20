"""Catálogo y descarga de los modelos ONNX livianos (corren en CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from plasmo.config import MODELS_DIR

REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"


@dataclass(frozen=True)
class OnnxModel:
    key: str
    label: str
    url: str
    filename: str
    size_mb: int
    kind: str  # fondo | profundidad


CATALOG: dict[str, OnnxModel] = {
    m.key: m
    for m in [
        OnnxModel("isnet-general-use", "IS-Net (recomendado)", REMBG + "isnet-general-use.onnx", "isnet-general-use.onnx", 179, "fondo"),
        OnnxModel("birefnet-lite", "BiRefNet lite (bordes finos)", REMBG + "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", "birefnet-general-lite.onnx", 224, "fondo"),
        OnnxModel("u2netp", "U2-Net P (rápido, 5 MB)", REMBG + "u2netp.onnx", "u2netp.onnx", 5, "fondo"),
        OnnxModel(
            "depth-anything-v2-small",
            "Depth Anything V2 Small",
            "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
            "depth-anything-v2-small.onnx",
            99,
            "profundidad",
        ),
    ]
}


def model_path(key: str) -> Path:
    return MODELS_DIR / CATALOG[key].filename


def is_downloaded(key: str) -> bool:
    return key in CATALOG and model_path(key).is_file()


def download(key: str, on_progress: Callable[[int, int], None] | None = None) -> Path:
    model = CATALOG[key]
    target = model_path(key)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", model.url, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with partial.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    partial.replace(target)
    return target
