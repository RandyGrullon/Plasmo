"""ComfyUI simulado para probar Plasmo sin GPU.

Imita las rutas que usa Plasmo (/system_stats, /models/checkpoints, /object_info,
/upload/image, /prompt, /history, /view, /ws, /interrupt, /free). En vez de
Hunyuan3D, "infla" la silueta de la imagen subida y la guarda como GLB Y-arriba
normalizado a [-1, 1], igual que la salida real.

    uv run python tools/mock_comfyui.py --port 8188
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import threading
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from PIL import Image
from scipy import ndimage

from plasmo.mesh.heightfield import heightfield_solid

app = FastAPI()
UPLOADS: dict[str, bytes] = {}
OUTPUTS: dict[str, bytes] = {}
HISTORY: dict[str, dict] = {}
SOCKETS: dict[str, WebSocket] = {}
CHECKPOINTS = ["hunyuan_3d_v2.1.safetensors", "hunyuan3d-dit-v2_fp16.safetensors", "sd_xl_base_1.0.safetensors"]
NODES = {"ImageOnlyCheckpointLoader", "Hunyuan3Dv2Conditioning", "EmptyLatentHunyuan3Dv2", "VAEDecodeHunyuan3D", "VoxelToMesh", "SaveGLB", "KSampler"}
STEP_DELAY = 0.02


@app.get("/system_stats")
def system_stats():
    return {"system": {"comfyui_version": "0.42.1-mock", "os": "nt"},
            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5070 (simulada)", "type": "cuda", "vram_total": 12 * 2**30, "vram_free": 11 * 2**30}]}


@app.get("/models/checkpoints")
def checkpoints():
    return CHECKPOINTS


@app.get("/object_info/{cls}")
def object_info(cls: str):
    return {cls: {"input": {}, "output": []}} if cls in NODES else {}


@app.post("/upload/image")
async def upload(image: UploadFile = File(...), overwrite: str = Form("false")):
    UPLOADS[image.filename] = await image.read()
    return {"name": image.filename, "subfolder": "", "type": "input"}


def inflate(png: bytes) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB").resize((160, 160))
    rgb = np.asarray(img, dtype=np.float32)
    mask = np.linalg.norm(rgb - 255, axis=-1) > 30
    mask = ndimage.binary_fill_holes(ndimage.binary_opening(mask, iterations=1))
    if mask.sum() < 50:
        mask = np.zeros((160, 160), bool)
        mask[40:120, 40:120] = True
    dist = ndimage.distance_transform_edt(mask)
    puff = np.sqrt(dist) * 2.0 + 0.5
    mesh = heightfield_solid(puff, mask, 1.0, bottom=-puff)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(2.0 / mesh.extents.max())
    return mesh.export(file_type="glb")


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
            await send(client_id, "executing", {"node": "8", "prompt_id": prompt_id})
            await send(client_id, "executing", {"node": "9", "prompt_id": prompt_id})
            name = f"plasmo_{len(OUTPUTS) + 1:05d}_.glb"
            OUTPUTS[name] = inflate(UPLOADS[load["inputs"]["image"]])
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
    media = "model/gltf-binary" if filename.endswith(".glb") else "image/png"
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
