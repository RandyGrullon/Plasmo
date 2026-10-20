"""Ollama simulado para probar Plasmo sin GPU.

    uv run python tools/mock_ollama.py --port 11434
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
MODELS = {"qwen3.5:9b": ["completion", "vision", "tools", "thinking"], "llama3.2:3b": ["completion", "tools"]}
CALLS: list[dict] = []
THINK_SECONDS = 0.0  # el modo demo lo sube para que se vea el "pensando"

ANALYSIS = {
    "descripcion": "Un muñeco caricaturesco de cabeza naranja con orejas puntiagudas y cuerpo azul.",
    "sujeto": "muñeco gato",
    "apta_para_figura": True,
    "problemas": [],
    "forma_recomendada": "figura",
    "por_que": "Es un solo personaje, entero y sobre fondo liso: ideal para figura 3D.",
    "tamano_mm": 90,
    "orientacion": "De pie, con la base plana sobre la cama.",
    "soportes": "minimos",
    "material": "PLA",
    "nombre_archivo": "Muñeco Gato",
    "consejos": ["Imprime a 0.12 mm de capa para las orejas.", "Usa soportes en árbol solo bajo la cabeza."],
}


@app.get("/api/version")
def version():
    return {"version": "0.40.6-mock"}


@app.get("/api/tags")
def tags():
    return {"models": [{"name": m, "model": m, "size": 6_600_000_000, "details": {"family": m.split(":")[0], "parameter_size": "9B", "quantization_level": "Q4_K_M"}} for m in MODELS]}


@app.post("/api/show")
async def show(request: Request):
    model = (await request.json()).get("model")
    if model not in MODELS:
        return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
    return {"capabilities": MODELS[model], "details": {}}


@app.get("/api/ps")
def ps():
    return {"models": []}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    CALLS.append(body)
    model = body.get("model")
    if model not in MODELS:
        return JSONResponse({"error": f"model '{model}' not found, try pulling it first"}, status_code=404)
    if not body.get("messages"):
        return {"model": model, "done": True, "done_reason": "unload" if body.get("keep_alive") == 0 else "load"}
    if THINK_SECONDS:
        await asyncio.sleep(THINK_SECONDS)
    if body.get("format"):
        props = (body["format"] or {}).get("properties", {})
        content = json.dumps(ANALYSIS if "forma_recomendada" in props else {"prompt_en": "a cute orange cat toy, full body, white background", "nombre": "gato naranja"}, ensure_ascii=False)
        return {"model": model, "message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop"}
    text = "Para PLA usa 210 °C en boquilla y 60 °C en cama; capa de 0.16 mm y 15 % de relleno."
    if not body.get("stream", True):
        return {"model": model, "message": {"role": "assistant", "content": text}, "done": True}

    async def gen():
        for word in text.split(" "):
            await asyncio.sleep(0.01)
            yield json.dumps({"model": model, "message": {"role": "assistant", "content": word + " "}, "done": False}) + "\n"
        yield json.dumps({"model": model, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/pull")
async def pull(request: Request):
    model = (await request.json()).get("model")

    async def gen():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        total = 1000
        for done in range(0, total + 1, 250):
            await asyncio.sleep(0.02)
            yield json.dumps({"status": "pulling abc123", "digest": "sha256:abc123", "total": total, "completed": done}) + "\n"
        MODELS.setdefault(model, ["completion", "vision"])
        yield json.dumps({"status": "success"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def serve_in_thread(port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    uvicorn.run(app, host="127.0.0.1", port=ap.parse_args().port, log_level="warning")
