"""Generadores de forma 3D por HTTP (no importan torch: corren en su propio proceso).

- ComfyUI (recomendado): nodos nativos de Hunyuan3D 2.0 / 2.1, progreso por WebSocket.
- Hunyuan3D-2 `api_server.py` (o el paquete WinPortable): POST /generate → GLB.
"""

from __future__ import annotations

import base64
import io
import json
import random
import time
import uuid
from contextlib import ExitStack
from urllib.parse import urlparse

import httpx
from PIL import Image

from plasmo.config import Settings
from plasmo.jobs import Cancelled, Job

# Ajustes de las plantillas oficiales de Comfy-Org por checkpoint
PRESETS = {
    "2.1": {"resolution": 4096, "steps": 30, "cfg": 5.0},
    "2.0": {"resolution": 3072, "steps": 20, "cfg": 8.0},
    "mini": {"resolution": 512, "steps": 30, "cfg": 5.0},
}

CHECKPOINTS = [
    {
        "archivo": "hunyuan_3d_v2.1.safetensors",
        "nombre": "Hunyuan3D 2.1 (recomendado, ~10 GB VRAM)",
        "gb": 7.4,
        "url": "https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors",
    },
    {
        "archivo": "hunyuan3d-dit-v2_fp16.safetensors",
        "nombre": "Hunyuan3D 2.0 (respaldo, ~6 GB VRAM)",
        "gb": 4.9,
        "url": "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors",
    },
]


def preset_for(checkpoint: str) -> dict:
    name = checkpoint.lower()
    if "mini" in name:
        return PRESETS["mini"]
    if "2.1" in name or "v2_1" in name or "2-1" in name:
        return PRESETS["2.1"]
    return PRESETS["2.0"]


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def hunyuan_workflow(image_name: str, checkpoint: str, params: dict) -> dict:
    """Grafo en formato API: imagen → Hunyuan3D (solo forma) → GLB."""
    preset = preset_for(checkpoint)
    steps = int(params.get("pasos") or 0) or preset["steps"]
    cfg = float(params.get("guia") or 0) or preset["cfg"]
    seed = int(params.get("semilla") or 0) or random.randint(1, 2**31 - 1)
    octree = int(params.get("octree") or 256)
    algorithm = "basic" if params.get("algoritmo") == "rapido" else "surface net"
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"}},
        "4": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["3", 0]}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
        "6": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": preset["resolution"], "batch_size": 1}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["6", 0],
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["7", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": octree}},
        "9": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["8", 0], "algorithm": algorithm, "threshold": 0.6}},
        "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": "3d/plasmo"}},
    }


def text_to_image_workflow(prompt: str, checkpoint: str, seed: int | None = None) -> dict:
    """SD/SDXL de un archivo con nodos básicos; imagen 1024² pensada para pasar a 3D."""
    negative = "multiple objects, text, watermark, cropped, cut off, busy background, shadow, blurry, lowres"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed or random.randint(1, 2**31 - 1), "steps": 25, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "plasmo/idea"}},
    }


class GeneratorError(RuntimeError):
    pass


class ComfyUIGenerator:
    kind = "comfyui"
    REQUIRED_NODES = ["ImageOnlyCheckpointLoader", "Hunyuan3Dv2Conditioning", "EmptyLatentHunyuan3Dv2", "VAEDecodeHunyuan3D", "VoxelToMesh", "SaveGLB"]

    def __init__(self, url: str, checkpoint: str, t2i_checkpoint: str = ""):
        self.url = url.rstrip("/")
        self.checkpoint = checkpoint
        self.t2i_checkpoint = t2i_checkpoint

    def _client(self, timeout: float = 30.0) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=httpx.Timeout(timeout, connect=4.0))

    # ----------------------------------------------------------------- estado
    def status(self) -> dict:
        info = {"tipo": self.kind, "nombre": "ComfyUI", "url": self.url, "checkpoint": self.checkpoint, "ok": False, "catalogo": CHECKPOINTS}
        try:
            with self._client(4) as c:
                stats = c.get("/system_stats").json()
                ckpts = c.get("/models/checkpoints").json()
                missing = [n for n in self.REQUIRED_NODES if not c.get(f"/object_info/{n}").json()]
        except (httpx.HTTPError, ValueError):
            info["mensaje"] = f"ComfyUI no responde en {self.url}. Ábrelo (run_nvidia_gpu.bat) y vuelve a intentar."
            return info
        dev = (stats.get("devices") or [{}])[0]
        info.update(
            version=(stats.get("system") or {}).get("comfyui_version"),
            gpu=dev.get("name"),
            vram_gb=round((dev.get("vram_total") or 0) / 2**30, 1),
            vram_libre_gb=round((dev.get("vram_free") or 0) / 2**30, 1),
            checkpoints=ckpts,
        )
        if missing:
            info["mensaje"] = "Tu ComfyUI no tiene los nodos de Hunyuan3D (" + ", ".join(missing) + "). Actualízalo."
        elif self.checkpoint not in ckpts:
            info["mensaje"] = f"Falta {self.checkpoint} en ComfyUI/models/checkpoints."
        else:
            info.update(ok=True, mensaje=f"Listo con {self.checkpoint}")
        info["texto_a_imagen"] = bool(self.t2i_checkpoint and self.t2i_checkpoint in ckpts)
        return info

    def supports_text_to_image(self) -> bool:
        return bool(self.t2i_checkpoint)

    # ------------------------------------------------------------- ejecución
    def _upload(self, data: bytes, name: str) -> str:
        with self._client(60) as c:
            r = c.post("/upload/image", files={"image": (name, data, "image/png")}, data={"overwrite": "true"})
            r.raise_for_status()
            up = r.json()
        return f"{up['subfolder']}/{up['name']}" if up.get("subfolder") else up["name"]

    def _queue(self, workflow: dict, client_id: str) -> str:
        with self._client(30) as c:
            r = c.post("/prompt", json={"prompt": workflow, "client_id": client_id})
        if r.status_code >= 400:
            try:
                err = r.json()
                detail = err.get("error", {}).get("message") or json.dumps(err.get("node_errors"))[:400]
            except ValueError:
                detail = r.text[:400]
            raise GeneratorError(f"ComfyUI rechazó el flujo: {detail}")
        return r.json()["prompt_id"]

    def _interrupt(self, prompt_id: str) -> None:
        try:
            with self._client(5) as c:
                c.post("/interrupt", json={"prompt_id": prompt_id})
        except httpx.HTTPError:
            pass

    def free_vram(self) -> None:
        try:
            with self._client(10) as c:
                c.post("/free", json={"unload_models": True, "free_memory": True})
        except httpx.HTTPError:
            pass

    def _run(self, workflow: dict, job: Job, phase: str, stages: dict[str, tuple[float, str]], sampler: str, span: tuple[float, float]) -> dict:
        """Encola, sigue el progreso por WebSocket y devuelve las salidas de /history."""
        from websockets.sync.client import connect

        client_id = uuid.uuid4().hex
        parsed = urlparse(self.url)
        ws_url = f"{'wss' if parsed.scheme == 'https' else 'ws'}://{parsed.netloc}/ws?clientId={client_id}"
        lo, hi = span
        with ExitStack() as stack:
            try:
                ws = stack.enter_context(connect(ws_url, open_timeout=5, max_size=None))
            except Exception:
                ws = None  # sin WebSocket: se consulta /history y la barra se estima
            prompt_id = self._queue(workflow, client_id)
            if ws is None:
                return self._poll(prompt_id, job, phase)
            while True:
                if job.cancel_event.is_set():
                    self._interrupt(prompt_id)
                    raise Cancelled()
                try:
                    raw = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    continue  # vistas previas binarias
                msg = json.loads(raw)
                data = msg.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                kind = msg.get("type")
                if kind == "progress" and str(data.get("node")) == sampler and data.get("max"):
                    frac = data["value"] / data["max"]
                    job.update(phase, lo + (hi - lo) * frac, f"Muestreando {data['value']}/{data['max']}")
                elif kind == "executing":
                    node = data.get("node")
                    if node is None:
                        break
                    if str(node) in stages:
                        frac, label = stages[str(node)]
                        job.update(phase, frac, label)
                elif kind == "execution_error":
                    raise GeneratorError(f"ComfyUI falló en {data.get('node_type')}: {data.get('exception_message', '').strip()}")
                elif kind == "execution_interrupted":
                    raise Cancelled()
                elif kind == "execution_success":
                    break
        return self._history(prompt_id)

    def _poll(self, prompt_id: str, job: Job, phase: str, timeout_s: float = 1800) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            if job.cancel_event.is_set():
                self._interrupt(prompt_id)
                raise Cancelled()
            hist = self._history(prompt_id, allow_empty=True)
            if hist:
                return hist
            time.sleep(1.0)
        raise GeneratorError("ComfyUI tardó demasiado.")

    def _history(self, prompt_id: str, allow_empty: bool = False) -> dict:
        with self._client(30) as c:
            data = c.get(f"/history/{prompt_id}").json()
        entry = data.get(prompt_id)
        if not entry:
            if allow_empty:
                return {}
            raise GeneratorError("ComfyUI no devolvió resultados.")
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
            detail = msgs[0][1].get("exception_message", "") if msgs else ""
            raise GeneratorError(f"ComfyUI falló: {detail.strip()}")
        return entry.get("outputs") or {}

    def _fetch(self, ref: dict) -> bytes:
        with self._client(120) as c:
            r = c.get("/view", params={"filename": ref["filename"], "subfolder": ref.get("subfolder", ""), "type": ref.get("type", "output")})
            r.raise_for_status()
            return r.content

    def generate(self, image: Image.Image, job: Job, phase: str, params: dict) -> bytes:
        name = self._upload(png_bytes(image), f"plasmo-{uuid.uuid4().hex[:8]}.png")
        job.update(phase, 0.03, "Cargando Hunyuan3D en la GPU…")
        workflow = hunyuan_workflow(name, self.checkpoint, params)
        stages = {"8": (0.8, "Decodificando el volumen…"), "9": (0.9, "Extrayendo la superficie…"), "10": (0.97, "Guardando GLB…")}
        outputs = self._run(workflow, job, phase, stages, sampler="7", span=(0.05, 0.78))
        refs = (outputs.get("10") or {}).get("3d") or []
        if not refs:
            raise GeneratorError("ComfyUI terminó pero no guardó el GLB.")
        glb = self._fetch(refs[0])
        if params.get("liberar_vram", True):
            self.free_vram()
        return glb

    def text_to_image(self, prompt: str, job: Job, phase: str) -> bytes:
        workflow = text_to_image_workflow(prompt, self.t2i_checkpoint)
        outputs = self._run(workflow, job, phase, {"6": (0.92, "Decodificando imagen…")}, sampler="5", span=(0.05, 0.9))
        refs = (outputs.get("7") or {}).get("images") or []
        if not refs:
            raise GeneratorError("ComfyUI no devolvió la imagen.")
        data = self._fetch(refs[0])
        self.free_vram()
        return data


class HunyuanApiGenerator:
    """`api_server.py` de Tencent-Hunyuan/Hunyuan3D-2 (usa 2mini-turbo, solo forma con texture=false)."""

    kind = "hunyuan_api"

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def status(self) -> dict:
        info = {"tipo": self.kind, "nombre": "Hunyuan3D API", "url": self.url, "ok": False}
        try:
            r = httpx.get(self.url + "/docs", timeout=4)
            info["ok"] = r.status_code < 500
            info["mensaje"] = "Servidor de Hunyuan3D listo" if info["ok"] else f"HTTP {r.status_code}"
        except httpx.HTTPError:
            info["mensaje"] = f"No responde en {self.url}. Arranca api_server.py (puerto 8081)."
        return info

    def supports_text_to_image(self) -> bool:
        return False

    def generate(self, image: Image.Image, job: Job, phase: str, params: dict) -> bytes:
        payload = {
            "image": base64.b64encode(png_bytes(image)).decode("ascii"),
            "seed": int(params.get("semilla") or 0) or random.randint(1, 2**31 - 1),
            "octree_resolution": int(params.get("octree") or 256),
            "num_inference_steps": int(params.get("pasos") or 0) or 5,
            "guidance_scale": float(params.get("guia") or 0) or 5.0,
            "texture": False,
            "face_count": 200000,
            "type": "glb",
        }
        job.update(phase, 0.02, "Hunyuan3D trabajando…")
        p = job._phase(phase)
        p.measured = False  # sin pasos: la barra se estima por tiempo
        with httpx.Client(timeout=httpx.Timeout(1800, connect=5)) as c:
            r = c.post(self.url + "/generate", json=payload)
        if r.status_code >= 400 or r.headers.get("content-type", "").startswith("application/json"):
            raise GeneratorError(f"Hunyuan3D devolvió un error: {r.text[:300]}")
        return r.content

    def text_to_image(self, prompt: str, job: Job, phase: str) -> bytes:
        raise GeneratorError("El servidor de Hunyuan3D no genera imágenes.")


def get(settings: Settings):
    if settings.generator == "hunyuan_api":
        return HunyuanApiGenerator(settings.hunyuan_api_url)
    return ComfyUIGenerator(settings.comfy_url, settings.comfy_checkpoint, settings.comfy_t2i_checkpoint)
