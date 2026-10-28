"""Servidor local de Plasmo: API JSON + interfaz web (se abre en ventana propia)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

from plasmo import __version__, catalog, generators, pipeline
from plasmo.config import ConfigStore
from plasmo.engines import ENGINES, NAMES, EngineManager, start_many
from plasmo.imaging import downloads
from plasmo.imaging import processing as proc
from plasmo.jobs import JobManager
from plasmo.ollama_client import CHAT_SYSTEM, RECOMMENDED, OllamaClient, OllamaError, slug
from plasmo.projects import ProjectStore

WEB_DIR = Path(__file__).parent / "web"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def create_app(config: ConfigStore | None = None, jobs: JobManager | None = None, engines: EngineManager | None = None) -> FastAPI:
    app = FastAPI(title="Plasmo", version=__version__, docs_url="/api/docs", redoc_url=None)
    cfg = config or ConfigStore()
    manager = jobs or JobManager()
    motores = engines or EngineManager()
    status_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="estado")

    def settings():
        return cfg.settings

    def store() -> ProjectStore:
        return ProjectStore(Path(settings().output_dir))

    clients: dict[str, OllamaClient] = {}

    def ollama() -> OllamaClient:
        # uno por URL, para reutilizar la caché de capacidades de cada modelo
        url = settings().ollama_url
        if url not in clients:
            clients[url] = OllamaClient(url)
        return clients[url]

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Solo atiende a esta PC y a la propia interfaz: el Host evita DNS rebinding
        # y la cabecera propia obliga a un preflight CORS que otras webs no pasan.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS and host != "testserver":
            return JSONResponse({"detail": "Host no permitido"}, status_code=403)
        if request.method in ("POST", "PUT", "DELETE") and request.headers.get("x-plasmo") != "1":
            return JSONResponse({"detail": "Falta la cabecera X-Plasmo"}, status_code=403)
        return await call_next(request)

    # --------------------------------------------------------------- estado
    @app.get("/api/ping")
    def ping():
        return {"app": "plasmo", "version": __version__}

    @app.get("/api/estado")
    def estado():
        s = settings()
        # el generador se consulta en paralelo: un motor apagado cuesta ~2 s en Windows
        gen_status = status_pool.submit(lambda: generators.get(s).status())
        client = ollama()
        version = client.version()
        info = {"version": __version__, "ollama": {"ok": version is not None, "version": version, "url": s.ollama_url}}
        if version:
            try:
                models = client.models()
                vision = []
                for m in models:
                    try:
                        caps = client.capabilities(m["name"])
                    except Exception:
                        caps = []
                    m["capacidades"] = caps
                    if "vision" in caps:
                        vision.append(m["name"])
                info["ollama"].update(modelos=models, vision=vision, modelo_activo=s.vision_model,
                                      modelo_listo=s.vision_model in [m["name"] for m in models])
            except Exception as exc:
                info["ollama"]["error"] = str(exc)
        info["ollama"]["recomendados"] = RECOMMENDED
        info["generador"] = gen_status.result()
        info["onnx"] = [
            {"clave": m.key, "nombre": m.label, "tipo": m.kind, "mb": m.size_mb, "descargado": downloads.is_downloaded(m.key)}
            for m in downloads.CATALOG.values()
        ]
        return info

    @app.get("/api/config")
    def get_config():
        return settings().to_dict()

    @app.put("/api/config")
    async def put_config(request: Request):
        return cfg.update(await request.json()).to_dict()

    @app.get("/api/formas")
    def formas():
        return {"formas": catalog.FORMS}

    # ------------------------------------------------------------ proyectos
    @app.get("/api/proyectos")
    def proyectos():
        return {"proyectos": store().list()}

    def new_project(data: bytes, filename: str, recorte: str = "", **meta) -> dict:
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(413, "La imagen pesa más de 40 MB.")
        try:
            img = proc.load_image(data)
        except Exception:
            raise HTTPException(400, "No pude abrir la imagen (usa PNG, JPG o WebP).")
        pid, folder = store().create(slug(Path(filename or "imagen").stem))
        img.save(folder / "original.png")
        if meta:
            store().update_meta(pid, **meta)
        try:
            pipeline.segment_project(store(), pid, recorte or settings().bg_model)
        except Exception as exc:
            pipeline.segment_project(store(), pid, "simple")
            store().update_meta(pid, aviso_recorte=f"Recorte simple (sin modelo): {exc}")
        return store().meta(pid)

    @app.post("/api/proyectos")
    async def crear(imagen: UploadFile = File(...), recorte: str = Form("")):
        # el recorte de máxima calidad tarda: en un hilo aparte para no congelar el servidor
        return await run_in_threadpool(new_project, await imagen.read(), imagen.filename or "imagen", recorte)

    @app.post("/api/ejemplo")
    def ejemplo():
        """Proyecto con la imagen de muestra, para seguir el tutorial sin foto propia."""
        return new_project((WEB_DIR / "ejemplo.png").read_bytes(), "ejemplo-gatito", ejemplo=True)

    @app.get("/api/proyectos/{pid}")
    def proyecto(pid: str):
        try:
            return store().meta(pid) | {"id": pid}
        except KeyError:
            raise HTTPException(404, "Proyecto no encontrado")

    @app.post("/api/proyectos/{pid}/recorte")
    async def recortar(pid: str, request: Request):
        body = await request.json()
        try:
            objects = "principal" if body.get("objetos") == "principal" else "todos"
            return await run_in_threadpool(pipeline.segment_project, store(), pid, body.get("modelo") or settings().bg_model, objects)
        except KeyError:
            raise HTTPException(404, "Proyecto no encontrado")

    @app.get("/api/proyectos/{pid}/archivo/{name:path}")
    def archivo(pid: str, name: str, descargar: bool = False):
        try:
            path = store().file(pid, name)
        except KeyError:
            raise HTTPException(404, "Archivo no encontrado")
        return FileResponse(path, filename=path.name if descargar else None, headers={"Cache-Control": "no-store"})

    @app.post("/api/proyectos/{pid}/abrir")
    async def abrir(pid: str, request: Request):
        body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
        try:
            if body.get("archivo"):
                target = store().file(pid, body["archivo"])
            elif body.get("version"):
                target = store().file(pid, f"{body['version']}/info.json").parent
            else:
                target = store().path(pid)
        except KeyError:
            raise HTTPException(404, "No encontrado")
        open_path(target)
        return {"ok": True}

    @app.post("/api/proyectos/{pid}/analizar")
    def analizar(pid: str):
        try:
            folder = store().path(pid)
        except KeyError:
            raise HTTPException(404, "Proyecto no encontrado")
        s = settings()

        def work(job):
            job.start("analizar_ia", f"{s.vision_model} está mirando la foto…")
            original = Image.open(folder / "original.png")
            cut = Image.open(folder / "recorte.png") if (folder / "recorte.png").exists() else None
            # ve la foto entera (todo lo que hay) y el recorte, para avisar si le falta algo
            result = ollama().analyze_image(s.vision_model, original, cut)
            job.done("analizar_ia")
            store().update_meta(pid, analisis=result)
            return {"analisis": result}

        job = manager.submit("analisis", [("analizar_ia", "Analizar con Ollama", 1)], work)
        return job.to_dict()

    @app.post("/api/proyectos/{pid}/generar")
    async def generar(pid: str, request: Request):
        body = await request.json()
        form = body.get("forma", "relieve")
        try:
            store().path(pid)
        except KeyError:
            raise HTTPException(404, "Proyecto no encontrado")
        if form == "figura":
            st = generators.get(settings()).status()
            if not st.get("ok"):
                raise HTTPException(409, st.get("mensaje") or "El generador 3D no está disponible.")
        s = settings()
        job = manager.submit(
            f"forma:{form}",
            pipeline.phases_for(form),
            lambda job: pipeline.run_generation(job, store(), s, pid, form, body.get("parametros") or {}, body.get("impresion") or {}),
        )
        return job.to_dict()

    @app.post("/api/proyectos/{pid}/revisar/{version}")
    def revisar(pid: str, version: str):
        try:
            return pipeline.run_reanalysis(store(), settings(), pid, version)
        except (KeyError, OSError):
            raise HTTPException(404, "Versión no encontrada")

    @app.post("/api/texto-a-imagen")
    async def texto_a_imagen(request: Request):
        body = await request.json()
        idea = (body.get("idea") or "").strip()
        if not idea:
            raise HTTPException(400, "Escribe una idea.")
        s = settings()
        gen = generators.get(s)
        if not gen.supports_text_to_image():
            raise HTTPException(409, "Configura un checkpoint de texto→imagen para ComfyUI en Ajustes.")

        def work(job):
            job.start("prompt", "Ollama está escribiendo el prompt…")
            try:
                p = ollama().image_prompt(s.vision_model, idea)
            except Exception:
                p = {"prompt": idea, "nombre": slug(idea)}
            job.done("prompt")
            ollama().unload(s.vision_model)
            job.start("imagen", "Dibujando la imagen…")
            png = gen.text_to_image(p["prompt"], job, "imagen")
            job.done("imagen")
            pid, folder = store().create(p["nombre"])
            (folder / "original.png").write_bytes(png)
            store().update_meta(pid, prompt=p["prompt"], idea=idea)
            job.start("preparar", "Recortando…")
            try:
                pipeline.segment_project(store(), pid, s.bg_model)
            except Exception:
                pipeline.segment_project(store(), pid, "simple")
            job.done("preparar")
            return {"proyecto": pid}

        job = manager.submit("texto", [("prompt", "Escribir prompt", 10), ("imagen", "Generar imagen", 80), ("preparar", "Recortar", 10)], work)
        return job.to_dict()

    # -------------------------------------------------------------- trabajos
    @app.get("/api/trabajos/{jid}")
    def trabajo(jid: str):
        job = manager.get(jid)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return job.to_dict()

    @app.post("/api/trabajos/{jid}/cancelar")
    def cancelar(jid: str):
        return {"ok": manager.cancel(jid)}

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        s = settings()
        model = s.chat_model or s.vision_model
        system = CHAT_SYSTEM
        pid = body.get("proyecto")
        if pid:
            try:
                meta = store().meta(pid)
                ctx = {"analisis": meta.get("analisis"), "ultima_version": (meta.get("versiones") or [None])[-1]}
                system += "\nContexto del proyecto actual:\n" + json.dumps(ctx, ensure_ascii=False)[:6000]
            except KeyError:
                pass
        messages = [{"role": "system", "content": system}] + [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in (body.get("mensajes") or [])[-16:]
        ]

        def stream():
            try:
                for piece in ollama().chat_stream(model, messages):
                    yield piece
            except OllamaError as exc:
                yield f"\n\n⚠️ {exc}"
            except Exception:
                yield "\n\n⚠️ No pude hablar con Ollama. ¿Está abierto?"

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

    # -------------------------------------------------------------- motores
    def pick_engines(body: dict) -> list[str]:
        names = body.get("motores") or list(ENGINES)
        bad = [n for n in names if n not in ENGINES]
        if bad:
            raise HTTPException(400, f"Motor desconocido: {', '.join(bad)}")
        return names

    @app.get("/api/motores")
    def motores_estado():
        return motores.status(settings())

    @app.post("/api/motores/encender")
    async def motores_encender(request: Request):
        names = pick_engines(await request.json() if request.headers.get("content-length") not in (None, "0") else {})
        s = settings()
        phases = [(n, f"Encender {NAMES[n]}", 1 if n == "ollama" else 4) for n in names]
        return manager.submit("motores", phases, lambda job: start_many(motores, names, s, job)).to_dict()

    @app.post("/api/motores/apagar")
    async def motores_apagar(request: Request):
        names = pick_engines(await request.json() if request.headers.get("content-length") not in (None, "0") else {})
        return {n: motores.stop(n) for n in names}

    # ------------------------------------------------------------ descargas
    @app.post("/api/ollama/descargar")
    async def ollama_descargar(request: Request):
        model = (await request.json()).get("modelo", "").strip()
        if not model:
            raise HTTPException(400, "Falta el modelo")

        def work(job):
            job.start("descargar", f"Descargando {model}…")
            for prog in ollama().pull(model):
                job.update("descargar", prog["percent"] / 100, f"{prog['status']} {prog['percent']:.0f} %")
            job.done("descargar")
            return {"modelo": model}

        return manager.submit("descarga", [("descargar", f"Descargar {model}", 1)], work).to_dict()

    @app.post("/api/onnx/descargar")
    async def onnx_descargar(request: Request):
        key = (await request.json()).get("clave", "")
        if key not in downloads.CATALOG:
            raise HTTPException(400, "Modelo desconocido")

        def work(job):
            job.start("descargar", f"Descargando {downloads.CATALOG[key].label}…")
            downloads.download(key, lambda done, total: job.update("descargar", done / total if total else 0, f"{done / 1e6:.0f} / {total / 1e6:.0f} MB"))
            job.done("descargar")
            return {"clave": key}

        return manager.submit("descarga", [("descargar", "Descargar modelo", 1)], work).to_dict()

    # ------------------------------------------------------------- interfaz
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
    return app


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 — abre la carpeta o el archivo con su app
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
