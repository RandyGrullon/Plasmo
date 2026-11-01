"""Generadores de forma 3D por HTTP (no importan torch: corren en su propio proceso).

- ComfyUI (recomendado): nodos nativos de Hunyuan3D 2.0 / 2.1, progreso por WebSocket.
  También IDOL (speridlabs) sobre los nodos nativos de TripoSplat: foto → splat → malla a color.
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

from plasmo import idol
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


def png_bytes(img: Image.Image, alpha: bool = False) -> bytes:
    buf = io.BytesIO()
    img.convert("RGBA" if alpha else "RGB").save(buf, "PNG")
    return buf.getvalue()


# calidad → (detalle de la malla, multiplicador de pasos sobre la plantilla oficial)
QUALITY = {"maxima": (512, 5 / 3), "alta": (384, 4 / 3), "rapida": (256, 1.0)}
# lo que aceptan los nodos de ComfyUI (cualquier otro valor vuelve al de la plantilla)
SAMPLERS = {"euler", "heun", "dpmpp_2m", "uni_pc"}
SCHEDULERS = {"normal", "simple", "beta", "sgm_uniform"}


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def hunyuan_workflow(image_name: str, checkpoint: str, params: dict) -> dict:
    """Grafo en formato API: imagen → Hunyuan3D (solo forma) → GLB."""
    preset = preset_for(checkpoint)
    octree_q, steps_mult = QUALITY.get(params.get("calidad", "maxima"), QUALITY["maxima"])
    steps = int(params.get("pasos") or 0) or round(preset["steps"] * steps_mult)
    cfg = float(params.get("guia") or 0) or preset["cfg"]
    seed = int(params.get("semilla") or 0) or random.randint(1, 2**31 - 1)
    octree = _clamp(int(params.get("octree") or 0) or octree_q, 64, 512)
    latents = _clamp(int(params.get("latentes") or 0) or preset["resolution"], 256, 8192)
    threshold = _clamp(float(params.get("umbral") or 0.6), -1.0, 1.0)
    shift = _clamp(float(params.get("shift") or 1.0), 0.1, 10.0)
    sampler = params.get("muestreador") if params.get("muestreador") in SAMPLERS else "euler"
    scheduler = params.get("programador") if params.get("programador") in SCHEDULERS else "normal"
    algorithm = "basic" if params.get("algoritmo") == "rapido" else "surface net"
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"}},
        "4": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["3", 0]}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift}},
        "6": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": latents, "batch_size": 1}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["6", 0],
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["7", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": octree}},
        "9": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["8", 0], "algorithm": algorithm, "threshold": threshold}},
        "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": "3d/plasmo"}},
    }


# calidad → (resolución de la malla que sale del splat, precisión del muestreo, pasos)
IDOL_QUALITY = {"maxima": (512, "fp32, fp16 attention", 50), "alta": (384, "default", 50), "rapida": (256, "default", 30)}
IDOL_PRECISIONS = {"default", "fp32, fp16 attention", "fp32"}


def idol_workflow(image_name: str, params: dict) -> dict:
    """Grafo en formato API: recorte con alfa → IDOL (TripoSplat) → splat → malla a color (GLB) + splat (PLY).

    Ajustes de IDOL según su paquete para ComfyUI: euler, programador simple, 50 pasos y
    cfg 4 (la guía 3 de IDOL escrita como la cuenta ComfyUI, 1 + w); el shift 3 viene en el modelo.
    """
    res_q, precision_q, steps_q = IDOL_QUALITY.get(params.get("calidad", "maxima"), IDOL_QUALITY["maxima"])
    steps = int(params.get("pasos") or 0) or steps_q
    cfg = float(params.get("guia") or 0) or 4.0
    seed = int(params.get("semilla") or 0) or random.randint(1, 2**31 - 1)
    mode = params.get("modo_foto") if params.get("modo_foto") in ("real", "synthetic") else "real"
    precision = params.get("precision_idol") if params.get("precision_idol") in IDOL_PRECISIONS else precision_q
    resolution = _clamp(int(params.get("resolucion_malla") or 0) or res_q, 64, 768) // 16 * 16
    gaussians = _clamp(int(params.get("gaussianas") or 262_144), 32_768, 1_048_576) // 32 * 32
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "InvertMask", "inputs": {"mask": ["1", 1]}},  # LoadImage da 1 - alfa
        "3": {"class_type": "IdolPreprocessImage", "inputs": {"image": ["1", 0], "mask": ["2", 0], "mode": mode}},
        "4": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dino_v3_vit_h.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "6": {"class_type": "TripoSplatConditioning", "inputs": {"clip_vision": ["4", 0], "vae": ["5", 0], "image": ["3", 0]}},
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": "idol_fp32.safetensors", "weight_dtype": "default"}},
        "8": {"class_type": "IdolPrecision", "inputs": {"model": ["7", 0], "precision": precision}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["6", 2],
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "idol_vae_decoder_fp32.safetensors"}},
        "11": {"class_type": "VAEDecodeTripoSplat", "inputs": {"samples": ["9", 0], "vae": ["10", 0], "num_gaussians": gaussians, "seed": seed}},
        "12": {
            "class_type": "SplatToMesh",
            "inputs": {
                "splat": ["11", 0], "resolution": resolution, "kernel": 5, "smooth": 0,
                "level": _clamp(float(params.get("nivel_superficie") or 0.4), 0.05, 2.0), "min_component": 500,
                "min_opacity": 0.02, "color_sharpen": _clamp(float(params.get("nitidez_color") or 2.0), 1.0, 8.0),
            },
        },
        "13": {"class_type": "SaveGLB", "inputs": {"mesh": ["12", 0], "filename_prefix": "3d/plasmo-idol"}},
        "14": {"class_type": "SplatToFile3D", "inputs": {"splat": ["11", 0], "format": "ply"}},
        "15": {"class_type": "SaveGLB", "inputs": {"mesh": ["14", 0], "filename_prefix": "3d/plasmo-idol-splat"}},
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
        self.extras: dict[str, bytes] = {}  # archivos de más del último trabajo (el splat .ply de IDOL)

    def _client(self, timeout: float = 30.0) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=httpx.Timeout(timeout, connect=4.0))

    # ----------------------------------------------------------------- estado
    def status(self) -> dict:
        info = {"tipo": self.kind, "nombre": "ComfyUI", "url": self.url, "checkpoint": self.checkpoint, "ok": False, "responde": False, "catalogo": CHECKPOINTS}
        try:
            with self._client(4) as c:
                stats = c.get("/system_stats").json()
                ckpts = c.get("/models/checkpoints").json()
                missing = [n for n in self.REQUIRED_NODES if not c.get(f"/object_info/{n}").json()]
                info["idol"] = idol.status(c)
        except (httpx.HTTPError, ValueError):
            info["mensaje"] = f"ComfyUI no responde en {self.url}. Ábrelo (run_nvidia_gpu.bat) y vuelve a intentar."
            info["idol"] = {"responde": False, "listo": False, "mensaje": "ComfyUI no responde: enciéndelo para ver o instalar IDOL."}
            return info
        dev = (stats.get("devices") or [{}])[0]
        info.update(
            responde=True,
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
        self.extras = {}
        if params.get("modelo_3d") == "idol":
            return self._generate_idol(image, job, phase, params)
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

    def _generate_idol(self, image: Image.Image, job: Job, phase: str, params: dict) -> bytes:
        """IDOL recibe el recorte con su transparencia: el encuadre lo hace su propio nodo."""
        name = self._upload(png_bytes(image, alpha=True), f"plasmo-{uuid.uuid4().hex[:8]}.png")
        job.update(phase, 0.03, "Cargando IDOL en la GPU…")
        workflow = idol_workflow(name, params)
        stages = {"11": (0.82, "Decodificando las gaussianas…"), "12": (0.88, "Sacando la malla del splat…"),
                  "13": (0.96, "Guardando la malla…"), "15": (0.98, "Guardando el splat…")}
        outputs = self._run(workflow, job, phase, stages, sampler="9", span=(0.06, 0.8))
        refs = (outputs.get("13") or {}).get("3d") or []
        if not refs:
            raise GeneratorError("ComfyUI terminó pero no guardó la malla de IDOL.")
        glb = self._fetch(refs[0])
        splat = (outputs.get("15") or {}).get("3d") or []
        if splat:
            try:
                self.extras["splat.ply"] = self._fetch(splat[0])
            except httpx.HTTPError:
                pass  # el splat es un extra: la pieza sale igual
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
        info = {"tipo": self.kind, "nombre": "Hunyuan3D API", "url": self.url, "ok": False,
                "idol": {"listo": False, "responde": False, "mensaje": "IDOL corre en ComfyUI: elígelo como generador en Ajustes."}}
        try:
            r = httpx.get(self.url + "/docs", timeout=4)
            info["ok"] = info["responde"] = r.status_code < 500
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
            "octree_resolution": int(params.get("octree") or 0) or QUALITY.get(params.get("calidad", "maxima"), QUALITY["maxima"])[0],
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
