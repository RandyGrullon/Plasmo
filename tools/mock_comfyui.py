"""ComfyUI simulado para probar Plasmo sin GPU.

Imita las rutas que usa Plasmo (/system_stats, /models/*, /object_info,
/internal/folder_paths, /upload/image, /prompt, /history, /view, /ws,
/interrupt, /free). En vez de Hunyuan3D, "infla" la silueta de la imagen subida
y la guarda como GLB Y-arriba normalizado a [-1, 1], igual que la salida real.

IDOL: es un ComfyUI 0.43 (con los nodos nativos de TripoSplat) cuyas carpetas
viven en ROOT. Los nodos de ComfyUI-Idol aparecen cuando su carpeta está en
custom_nodes, y /descargas sirve el paquete y modelos de mentira para probar el
instalador. La malla de IDOL sale a color (lineal, como ComfyUI) y girada
IDOL_YAW grados, porque IDOL decide solo hacia dónde mira el objeto.

    uv run python tools/mock_comfyui.py --port 8188
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image
from scipy import ndimage

from plasmo import idol
from plasmo.mesh.heightfield import heightfield_solid

app = FastAPI()
UPLOADS: dict[str, bytes] = {}
OUTPUTS: dict[str, bytes] = {}
HISTORY: dict[str, dict] = {}
SOCKETS: dict[str, WebSocket] = {}
CHECKPOINTS = ["hunyuan_3d_v2.1.safetensors", "hunyuan3d-dit-v2_fp16.safetensors", "sd_xl_base_1.0.safetensors"]
NODES = {"ImageOnlyCheckpointLoader", "Hunyuan3Dv2Conditioning", "EmptyLatentHunyuan3Dv2", "VAEDecodeHunyuan3D", "VoxelToMesh", "SaveGLB", "KSampler",
         *idol.NATIVE_NODES, "LoadImage"}
STEP_DELAY = 0.02
ROOT = Path(os.environ.get("PLASMO_MOCK_COMFY_DIR") or Path(tempfile.gettempdir()) / "plasmo-mock-comfy")
IDOL_YAW = 70.0  # hacia dónde deja mirando el objeto el IDOL simulado
FAKE_SIZE = int(os.environ.get("PLASMO_MOCK_FAKE_KB", "256")) * 1024  # bytes de cada modelo de mentira
DOWNLOAD_DELAY = float(os.environ.get("PLASMO_MOCK_DELAY", "0"))  # segundos por trozo de 64 KB (el demo lo sube)


def model_folders() -> dict[str, Path]:
    return {name: ROOT / "models" / name for name in ("checkpoints", "diffusion_models", "vae", "clip_vision")}


def idol_pack_installed() -> bool:
    return (ROOT / "custom_nodes" / idol.PACK_DIR / "__init__.py").is_file()


def fake_models(base_url: str) -> list[idol.ModelFile]:
    """Los modelos de IDOL, pero servidos por este simulador y del tamaño FAKE_SIZE."""
    return [idol.ModelFile(m.folder, m.name, f"{base_url}/descargas/{m.name}", FAKE_SIZE) for m in idol.MODELS]


@app.get("/system_stats")
def system_stats():
    return {"system": {"comfyui_version": "0.43.0-mock", "os": "nt"},
            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5070 (simulada)", "type": "cuda", "vram_total": 12 * 2**30, "vram_free": 11 * 2**30}]}


@app.get("/models/checkpoints")
def checkpoints():
    return CHECKPOINTS


@app.get("/models/{folder}")
def models(folder: str):
    path = model_folders().get(folder)
    if path is None or not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_file() and not p.name.endswith(".part"))


@app.get("/internal/folder_paths")
def folder_paths():
    out = {name: [str(path)] for name, path in model_folders().items()}
    out["custom_nodes"] = [str(ROOT / "custom_nodes")]
    return out


@app.get("/object_info/{cls}")
def object_info(cls: str):
    known = cls in NODES or (cls in idol.PACK_NODES and idol_pack_installed())
    return {cls: {"input": {}, "output": []}} if known else {}


def pack_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ComfyUI-Idol-mock/__init__.py", "from .nodes import comfy_entrypoint\n")
        zf.writestr("ComfyUI-Idol-mock/nodes.py", "# nodos de IDOL (simulados)\n")
    return buf.getvalue()


@app.get("/descargas/{name}")
async def descarga(name: str, request: Request):
    if name == "ComfyUI-Idol.zip":
        return Response(pack_zip(), media_type="application/zip")
    data = (name.encode() * (FAKE_SIZE // len(name) + 1))[:FAKE_SIZE]
    start = 0
    rng = request.headers.get("range", "")
    if rng.startswith("bytes="):
        start = int(rng[6:].split("-")[0] or 0)
        if start >= len(data):
            return Response(status_code=416)

    async def chunks():
        for i in range(start, len(data), 64 * 1024):
            if DOWNLOAD_DELAY:
                await asyncio.sleep(DOWNLOAD_DELAY)
            yield data[i:i + 64 * 1024]

    headers = {"Content-Length": str(len(data) - start), "Accept-Ranges": "bytes"}
    if start:
        headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
    return StreamingResponse(chunks(), status_code=206 if start else 200, headers=headers, media_type="application/octet-stream")


@app.post("/upload/image")
async def upload(image: UploadFile = File(...), overwrite: str = Form("false")):
    UPLOADS[image.filename] = await image.read()
    return {"name": image.filename, "subfolder": "", "type": "input"}


def inflate(png: bytes, colored: bool = False, yaw: float = 0.0) -> bytes:
    src = Image.open(io.BytesIO(png))
    if src.mode == "RGBA" and np.asarray(src.getchannel("A")).min() < 255:  # recorte con transparencia (IDOL)
        src.thumbnail((160, 160))
        mask = np.asarray(src.getchannel("A")) > 127
        img = src.convert("RGB")
    else:
        img = src.convert("RGB").resize((160, 160))
        rgb = np.asarray(img, dtype=np.float32)
        mask = np.linalg.norm(rgb - 255, axis=-1) > 30
    mask = ndimage.binary_fill_holes(ndimage.binary_opening(mask, iterations=1))
    if mask.sum() < 50:
        mask = np.zeros(mask.shape, bool)
        mask[mask.shape[0] // 4:-mask.shape[0] // 4, mask.shape[1] // 4:-mask.shape[1] // 4] = True
    dist = ndimage.distance_transform_edt(mask)
    puff = np.sqrt(dist) * 2.0 + 0.5
    mesh = heightfield_solid(puff, mask, 1.0, bottom=-puff)
    if colored:  # el color de la foto en cada vértice, guardado lineal como hace ComfyUI
        h, w = mask.shape
        rows = np.clip(np.round(h - mesh.vertices[:, 1]).astype(int), 0, h - 1)
        cols = np.clip(np.round(mesh.vertices[:, 0]).astype(int), 0, w - 1)
        srgb = np.asarray(img, dtype=np.float32)[rows, cols] / 255.0
        back = mesh.vertices[:, 2] < 0  # la espalda que «inventa» IDOL: del mismo tono, más apagada
        srgb[back] = srgb[back] * 0.45 + 0.3
        lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[np.round(lin * 255), np.full(len(lin), 255)].astype(np.uint8))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(2.0 / mesh.extents.max())
    if yaw:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0]))
    return mesh.export(file_type="glb")


def fake_ply() -> bytes:
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    return header.encode() + np.zeros(3, np.float32).tobytes()


def text_png(prompt: str) -> bytes:
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.ellipse([140, 90, 370, 320], fill=(220, 110, 60))
    d.rounded_rectangle([170, 280, 340, 470], radius=50, fill=(70, 120, 210))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def send(client_id: str, kind: str, data: dict):
    ws = SOCKETS.get(client_id)
    if ws:
        try:
            await ws.send_text(json.dumps({"type": kind, "data": data}))
        except Exception:
            pass


async def execute(prompt_id: str, client_id: str, workflow: dict):
    await send(client_id, "execution_start", {"prompt_id": prompt_id})
    sampler = next(k for k, v in workflow.items() if v["class_type"] == "KSampler")
    steps = int(workflow[sampler]["inputs"]["steps"])
    await send(client_id, "executing", {"node": sampler, "prompt_id": prompt_id})
    for i in range(1, steps + 1):
        await asyncio.sleep(STEP_DELAY)
        await send(client_id, "progress", {"value": i, "max": steps, "prompt_id": prompt_id, "node": sampler})
    outputs = {}
    for node, spec in workflow.items():
        if spec["class_type"] == "SaveGLB":
            load = next(v for v in workflow.values() if v["class_type"] == "LoadImage")
            feeder = workflow[spec["inputs"]["mesh"][0]]["class_type"]
            await send(client_id, "executing", {"node": "8", "prompt_id": prompt_id})
            await send(client_id, "executing", {"node": "9", "prompt_id": prompt_id})
            if feeder == "SplatToFile3D":
                name = f"plasmo-idol-splat_{len(OUTPUTS) + 1:05d}_.ply"
                OUTPUTS[name] = fake_ply()
            else:
                name = f"plasmo_{len(OUTPUTS) + 1:05d}_.glb"
                idol_mesh = feeder == "SplatToMesh"
                OUTPUTS[name] = inflate(UPLOADS[load["inputs"]["image"]], colored=idol_mesh, yaw=IDOL_YAW if idol_mesh else 0.0)
            outputs[node] = {"3d": [{"filename": name, "subfolder": "3d", "type": "output"}]}
        elif spec["class_type"] == "SaveImage":
            name = f"idea_{len(OUTPUTS) + 1:05d}_.png"
            prompt = next(v for v in workflow.values() if v["class_type"] == "CLIPTextEncode")["inputs"]["text"]
            OUTPUTS[name] = text_png(prompt)
            outputs[node] = {"images": [{"filename": name, "subfolder": "plasmo", "type": "output"}]}
        await send(client_id, "executing", {"node": node, "prompt_id": prompt_id})
    HISTORY[prompt_id] = {"prompt": [], "outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": []}}
    await send(client_id, "execution_success", {"prompt_id": prompt_id})
    await send(client_id, "executing", {"node": None, "prompt_id": prompt_id})


@app.post("/prompt")
async def prompt(request: Request):
    body = await request.json()
    workflow = body["prompt"]
    for spec in workflow.values():
        if spec["class_type"] in idol.PACK_NODES and not idol_pack_installed():
            return JSONResponse({"error": {"type": "invalid_prompt", "message": f"Cannot execute because node {spec['class_type']} does not exist."},
                                 "node_errors": {}}, status_code=400)
        if spec["class_type"] == "ImageOnlyCheckpointLoader" and spec["inputs"]["ckpt_name"] not in CHECKPOINTS:
            return JSONResponse({"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                                 "node_errors": {"1": {"errors": [{"message": "Value not in list"}]}}}, status_code=400)
    pid = str(uuid.uuid4())
    asyncio.create_task(execute(pid, body.get("client_id", ""), workflow))
    return {"prompt_id": pid, "number": len(HISTORY), "node_errors": {}}


@app.get("/history/{pid}")
def history(pid: str):
    return {pid: HISTORY[pid]} if pid in HISTORY else {}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output"):
    if filename not in OUTPUTS:
        return Response(status_code=404)
    media = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream" if filename.endswith(".ply") else "image/png"
    return Response(OUTPUTS[filename], media_type=media)


@app.post("/interrupt")
def interrupt():
    return {}


@app.post("/free")
def free():
    return {}


@app.websocket("/ws")
async def ws(websocket: WebSocket, clientId: str = ""):
    await websocket.accept()
    SOCKETS[clientId] = websocket
    await websocket.send_text(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": clientId}}))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        SOCKETS.pop(clientId, None)


def serve_in_thread(port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


BOOT_LOG = [
    "Checkpoint files will always be loaded safely.",
    "Total VRAM 12227 MB, total RAM 32693 MB",
    "pytorch version: 2.9.0+cu130",
    "Using pytorch attention",
    "Import times for custom nodes:",
    "Starting server",
]


if __name__ == "__main__":
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--lento", action="store_true", help="pasos más lentos para ver la barra")
    ap.add_argument("--paso", type=float, default=None, help="segundos por paso de muestreo")
    ap.add_argument("--arranque", type=float, default=0.0, help="segundos que tarda en arrancar (imita ComfyUI)")
    args = ap.parse_args()
    if args.lento:
        STEP_DELAY = 0.4
    if args.paso is not None:
        STEP_DELAY = args.paso
    for line in BOOT_LOG:  # mismas pistas que el ComfyUI real, para la barra de "encendiendo"
        print(line, flush=True)
        time.sleep(args.arranque / len(BOOT_LOG))
    print(f"To see the GUI go to: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
