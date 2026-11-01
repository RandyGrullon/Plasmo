"""IDOL de speridlabs: una foto → objeto 3D completo (splats gaussianos), dentro de ComfyUI.

IDOL usa la arquitectura de TripoSplat, que ComfyUI trae de fábrica desde la
0.43, junto con el nodo «Extract Mesh from Splat» que convierte el splat en
una malla a color. Plasmo instala solo lo que falta:

- el paquete de nodos ComfyUI-Idol (sin pip: el encuadre de IDOL para fotos y
  la precisión del muestreo), fijado a un commit revisado;
- los cuatro modelos públicos (~4,7 GB, Apache-2.0 y MIT) en las carpetas de
  ComfyUI, con descarga que se puede retomar.

El recorte lo pone Plasmo (su máscara de máxima calidad), así que no hace falta
el BiRefNet de ComfyUI.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from plasmo.config import Settings
from plasmo.engines import CREATE_NO_WINDOW, WINDOWS, EngineError, EngineManager, comfy_root
from plasmo.jobs import Job

PACK_REPO = "https://github.com/PedroMarinhoDev/ComfyUI-Idol"
PACK_COMMIT = "59ed4b43f0d04a2c8bac689f99e51901dd7c2329"  # 2026-10-30, revisado: solo nodos, sin dependencias
PACK_DIR = "ComfyUI-Idol"
PACK_URL = f"{PACK_REPO}/archive/{PACK_COMMIT}.zip"
MIN_VERSION = (0, 43, 0)

# nodos nativos de ComfyUI ≥ 0.43 y los dos del paquete que usa el flujo de Plasmo
NATIVE_NODES = ["TripoSplatConditioning", "VAEDecodeTripoSplat", "SplatToMesh", "SplatToFile3D", "UNETLoader", "CLIPVisionLoader", "InvertMask"]
PACK_NODES = ["IdolPreprocessImage", "IdolPrecision"]


@dataclass(frozen=True)
class ModelFile:
    folder: str  # carpeta de ComfyUI/models
    name: str
    url: str
    size: int

    @property
    def label(self) -> str:
        return f"{self.folder}/{self.name}"


HF = "https://huggingface.co"
MODELS = [
    ModelFile("diffusion_models", "idol_fp32.safetensors", f"{HF}/PedroMarinhoDev/Idol-ComfyUI/resolve/main/diffusion_models/idol_fp32.safetensors", 1_482_159_948),
    ModelFile("vae", "idol_vae_decoder_fp32.safetensors", f"{HF}/PedroMarinhoDev/Idol-ComfyUI/resolve/main/vae/idol_vae_decoder_fp32.safetensors", 1_152_256_316),
    ModelFile("clip_vision", "dino_v3_vit_h.safetensors", f"{HF}/VAST-AI/TripoSplat/resolve/main/clip_vision/dino_v3_vit_h.safetensors", 1_681_247_696),
    ModelFile("vae", "flux2-vae.safetensors", f"{HF}/VAST-AI/TripoSplat/resolve/main/vae/flux2-vae.safetensors", 336_213_556),
]
TOTAL_BYTES = sum(m.size for m in MODELS)
SIZES = {m.name: m.size for m in MODELS}  # los tamaños reales, para mostrar (el demo cambia MODELS por unos de mentira)


def gb(n: int) -> str:
    return f"{n / 1e9:.1f}".replace(".", ",")


def size_text(n: int) -> str:
    return f"{gb(n)} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class IdolError(RuntimeError):
    pass


def parse_version(text: str | None) -> tuple[int, ...]:
    parts = []
    for chunk in (text or "").split("-")[0].split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


# ------------------------------------------------------------------- estado

def status(client: httpx.Client) -> dict:
    """Qué tiene ya ComfyUI de IDOL. `client` apunta a ComfyUI y ya respondió."""
    info: dict = {"responde": True, "listo": False, "descarga_gb": round(TOTAL_BYTES / 1e9, 1)}
    try:
        version = ((client.get("/system_stats").json().get("system") or {}).get("comfyui_version"))
        native = [n for n in NATIVE_NODES if not client.get(f"/object_info/{n}").json()]
        pack = [n for n in PACK_NODES if not client.get(f"/object_info/{n}").json()]
        listed = {folder: set(client.get(f"/models/{folder}").json() or []) for folder in {m.folder for m in MODELS}}
    except (httpx.HTTPError, ValueError):
        return {"responde": False, "listo": False, "mensaje": "ComfyUI no responde: enciéndelo para ver o instalar IDOL."}
    missing = [m.name for m in MODELS if m.name not in listed.get(m.folder, set())]
    info.update(
        version=version,
        version_ok=parse_version(version) >= MIN_VERSION and not native,
        nodos=not pack,
        faltan=missing,
        falta_gb=round(sum(SIZES.get(name, 0) for name in missing) / 1e9, 1),
    )
    if not info["version_ok"]:
        info["mensaje"] = f"IDOL necesita ComfyUI 0.43 o más (tienes {version or 'una versión anterior'}). El instalador lo actualiza si es el portable."
    elif missing or pack:
        if missing:
            what = "los nodos y " if pack else ""
            info["mensaje"] = f"Falta instalar IDOL: {what}{len(missing)} modelos ({gb(sum(SIZES.get(n, 0) for n in missing))} GB)."
        else:
            info["mensaje"] = "Faltan los nodos de IDOL en ComfyUI (o reiniciarlo para que los cargue)."
    else:
        info.update(listo=True, mensaje="IDOL listo en ComfyUI.")
    return info


def folder_paths(client: httpx.Client) -> dict[str, list[str]]:
    """Las carpetas reales de ComfyUI (incluye las de extra_model_paths.yaml y custom_nodes)."""
    try:
        data = client.get("/internal/folder_paths").json()
    except (httpx.HTTPError, ValueError):
        return {}
    out = {}
    for key, value in (data or {}).items():
        paths = value[0] if value and isinstance(value[0], list) else value
        out[key] = [p for p in paths if isinstance(p, str)]
    return out


def _target_dir(paths: dict[str, list[str]], root: Path | None, folder: str) -> Path:
    if paths.get(folder):
        return Path(paths[folder][0])
    if root is None:
        raise IdolError("No sé dónde está tu carpeta de ComfyUI. Indícala en Ajustes → Motores.")
    return root / ("custom_nodes" if folder == "custom_nodes" else f"models/{folder}")


# ------------------------------------------------------------- instalación

PHASES = [
    ("comprobar", "Revisar ComfyUI", 3),
    ("actualizar", "Actualizar ComfyUI", 12),
    ("nodos", "Instalar los nodos de IDOL", 3),
    ("modelos", f"Descargar los modelos (~{gb(TOTAL_BYTES)} GB)", 74),
    ("reiniciar", "Reiniciar ComfyUI", 6),
    ("verificar", "Comprobar que IDOL responde", 2),
]


def _download(url: str, target: Path, size: int, job: Job, on_bytes) -> None:
    """Descarga a `.part` y la retoma si se cortó; al final la renombra."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as resp:
        if resp.status_code == 416:  # ya estaba completa
            resp.close()
        else:
            resp.raise_for_status()
            if done and resp.status_code != 206:  # el servidor no retoma: se empieza de cero
                done = 0
            with part.open("ab" if done else "wb") as fh:
                on_bytes(done)
                for chunk in resp.iter_bytes(1 << 20):
                    job.check()
                    fh.write(chunk)
                    done += len(chunk)
                    on_bytes(done)
    if part.stat().st_size != size:
        raise IdolError(f"{target.name} quedó incompleto ({part.stat().st_size} de {size} bytes). Vuelve a instalar para retomarlo.")
    part.replace(target)


def _install_pack(custom_nodes: Path, url: str = PACK_URL, commit: str = PACK_COMMIT) -> bool:
    """Deja ComfyUI-Idol en custom_nodes; devuelve True si cambió algo."""
    dest = custom_nodes / PACK_DIR
    marker = dest / ".plasmo-commit"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == commit:
        return False
    data = httpx.get(url, follow_redirects=True, timeout=httpx.Timeout(30, read=120)).raise_for_status().content
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        prefix = names[0].split("/", 1)[0] + "/" if names and all("/" in n for n in names) else ""
        tmp = custom_nodes / f".{PACK_DIR}-plasmo"
        shutil.rmtree(tmp, ignore_errors=True)
        for name in names:
            rel = name[len(prefix):]
            if not rel or rel.startswith(("..", "/")) or ".." in Path(rel).parts:
                continue
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
    if not (tmp / "__init__.py").exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise IdolError("El paquete de nodos de IDOL que bajé no trae __init__.py.")
    (tmp / ".plasmo-commit").write_text(commit, encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return True


def _portable_updater(root: Path | None) -> Path | None:
    if root is None:
        return None
    bat = root.parent / "update" / "update_comfyui.bat"
    return bat if bat.is_file() else None


def _run_updater(bat: Path, log: Path, job: Job) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        kwargs = {"cwd": str(bat.parent), "stdin": subprocess.DEVNULL, "stdout": fh, "stderr": subprocess.STDOUT}
        if WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW
        proc = subprocess.Popen(["cmd", "/c", bat.name] if WINDOWS else [str(bat)], **kwargs)
        while proc.poll() is None:
            job.check()
            time.sleep(1)
    if proc.returncode != 0:
        raise IdolError(f"El actualizador de ComfyUI terminó con error ({proc.returncode}). Mira {log}.")


def _restart(engines: EngineManager, settings: Settings, client: httpx.Client, job: Job) -> bool:
    """Reinicia ComfyUI si lo encendió Plasmo, o con ComfyUI-Manager si está. Devuelve si pudo."""
    if engines.ours("comfyui"):
        engines.stop("comfyui")
        engines.start("comfyui", settings, job, "reiniciar")
        return True
    for path in ("/api/manager/reboot", "/manager/reboot"):
        try:
            if client.get(path, timeout=5).status_code < 400:
                break
        except httpx.HTTPError:
            break  # se cayó mientras respondía: está reiniciando
    else:
        return False
    started = time.time()
    while time.time() - started < 360:  # baja y vuelve a subir
        job.check()
        time.sleep(2)
        if time.time() - started > 6 and engines.running("comfyui", settings):
            return True
    return False


def install(settings: Settings, engines: EngineManager, job: Job, models: list[ModelFile] | None = None,
            pack_url: str | None = None) -> dict:
    models = models or MODELS
    pack_url = pack_url or PACK_URL
    notes: list[str] = []

    job.start("comprobar", "Buscando ComfyUI…")
    if not engines.running("comfyui", settings):
        try:
            engines.start("comfyui", settings, job, "comprobar")
        except EngineError as exc:
            raise IdolError(f"Para instalar IDOL, ComfyUI tiene que estar encendido. {exc}") from exc
    client = httpx.Client(base_url=settings.comfy_url.rstrip("/"), timeout=httpx.Timeout(20, connect=4))
    try:
        before = status(client)
        paths = folder_paths(client)
        root = comfy_root(settings)
        job.done("comprobar")

        restart = False
        job.start("actualizar")
        if before.get("version_ok"):
            job.skip("actualizar")
        else:
            bat = _portable_updater(root)
            if bat is None:
                notes.append(f"Tu ComfyUI ({before.get('version') or 'versión desconocida'}) es anterior a 0.43 y no encontré "
                             "su actualizador. Actualízalo (ComfyUI-Manager → Update ComfyUI, o git pull) y reinícialo.")
                job.skip("actualizar")
            else:
                job.update("actualizar", 0.1, "Actualizando ComfyUI a la última versión…")
                _run_updater(bat, engines.logs_dir / "actualizar-comfyui.log", job)
                notes.append("Actualicé ComfyUI con su actualizador portable.")
                restart = True
                job.done("actualizar")

        job.start("nodos", "Instalando ComfyUI-Idol…")
        if _install_pack(_target_dir(paths, root, "custom_nodes"), pack_url):
            restart = True
            notes.append(f"Instalé los nodos de IDOL ({PACK_DIR}, commit {PACK_COMMIT[:7]}).")
        job.done("nodos")

        job.start("modelos")
        pending = []
        for m in models:
            target = _target_dir(paths, root, m.folder) / m.name
            if target.is_file() and target.stat().st_size == m.size:
                continue
            pending.append((m, target))
        total = sum(m.size for m, _ in pending) or 1
        base = 0
        for m, target in pending:
            def on_bytes(done, m=m, base=base):
                job.update("modelos", (base + done) / total, f"{m.name}: {size_text(done)} de {size_text(m.size)}")
            _download(m.url, target, m.size, job, on_bytes)
            base += m.size
        if pending:
            notes.append(f"Descargué {len(pending)} modelos ({size_text(sum(m.size for m, _ in pending))}).")
        job.done("modelos")

        job.start("reiniciar")
        restarted = None
        if restart and not status(client).get("listo"):  # algunos ComfyUI ya ven los nodos sin reiniciar
            job.update("reiniciar", 0.1, "Reiniciando ComfyUI para que cargue IDOL…")
            restarted = _restart(engines, settings, client, job)
            job.done("reiniciar")
        else:
            job.skip("reiniciar")

        job.start("verificar")
        after = status(client)
        if restarted is False and not after.get("listo"):
            notes.append("Reinicia ComfyUI para que cargue los nodos de IDOL: Plasmo no lo encendió y no lo puede reiniciar solo.")
        job.done("verificar")
    finally:
        client.close()
    return {"idol": after, "notas": notes}
