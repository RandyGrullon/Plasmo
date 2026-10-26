"""Encender y apagar Ollama y ComfyUI desde Plasmo.

Busca las instalaciones habituales en Windows (o usa las rutas de Ajustes),
los arranca sin ventana de consola con su salida en un log, espera a que
respondan por HTTP y solo apaga lo que Plasmo mismo encendió.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from plasmo.config import CACHE_DIR, Settings
from plasmo.jobs import Cancelled, Job

WINDOWS = sys.platform.startswith("win")
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
ENGINES = ("ollama", "comfyui")
NAMES = {"ollama": "Ollama", "comfyui": "ComfyUI"}
TIMEOUT_S = {"ollama": 60, "comfyui": 360}  # ComfyUI carga PyTorch: la primera vez tarda


class EngineError(RuntimeError):
    pass


@dataclass
class Launch:
    argv: list[str]
    kind: str  # bandeja | serve | portable | git | escritorio | comando
    desc: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    console: bool = True  # programa de consola: se oculta y su salida va al log


# ---------------------------------------------------------------- utilidades

def split_cmd(cmd: str) -> list[str]:
    """Parte un comando respetando comillas, también con rutas de Windows."""
    if WINDOWS:
        return [p[1:-1] if len(p) > 1 and p[0] == p[-1] == '"' else p for p in shlex.split(cmd, posix=False)]
    return shlex.split(cmd)


def host_port(url: str, default_port: int) -> tuple[str, int]:
    u = urlparse(url if "://" in url else f"http://{url}")
    return u.hostname or "127.0.0.1", u.port or default_port


def local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")


def fixed_drives() -> list[str]:
    """Discos locales (sin unidades de red, que pueden tardar en responder)."""
    if not WINDOWS:
        return []
    try:
        import ctypes

        drives = os.listdrives() if hasattr(os, "listdrives") else [f"{c}:\\" for c in "CDEFGHIJ"]
        return [d for d in drives if ctypes.windll.kernel32.GetDriveTypeW(d) == 3]
    except Exception:
        return ["C:\\"]


def _python_in(folder: Path) -> Path | None:
    for rel in ("venv/Scripts/python.exe", ".venv/Scripts/python.exe", "venv/bin/python", ".venv/bin/python"):
        p = folder / rel
        if p.is_file():
            return p
    return None


# ------------------------------------------------------------------- Ollama

def find_ollama(settings: Settings) -> Launch | None:
    host, port = host_port(settings.ollama_url, 11434)
    env = dict(os.environ)
    if (host, port) != ("127.0.0.1", 11434) and (host, port) != ("localhost", 11434):
        env["OLLAMA_HOST"] = f"{host}:{port}"
    if settings.ollama_comando.strip():
        return Launch(split_cmd(settings.ollama_comando), "comando", "Comando personalizado", env=env)

    candidates: list[Path] = []
    if settings.ollama_exe.strip():
        p = Path(settings.ollama_exe.strip().strip('"'))
        candidates += [p / "ollama app.exe", p / "ollama.exe"] if p.is_dir() else [p]
    base = local_appdata() / "Programs" / "Ollama"
    candidates += [base / "ollama app.exe", base / "ollama.exe"]
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found))

    for exe in candidates:
        if not exe.is_file():
            continue
        if exe.name.lower() == "ollama app.exe":
            return Launch([str(exe)], "bandeja", f"App de Ollama ({exe})", env=env, console=False)
        return Launch([str(exe), "serve"], "serve", f"ollama serve ({exe})", env=env)
    return None


# ------------------------------------------------------------------ ComfyUI

def comfy_dir_candidates() -> list[Path]:
    home = Path.home()
    names = ["ComfyUI_windows_portable", "ComfyUI"]
    out = [home / n for n in names]
    out += [home / sub / n for sub in ("Downloads", "Desktop", "Documents") for n in names]
    for drive in fixed_drives():
        root = Path(drive)
        out += [root / n for n in names]
        out += [root / sub / n for sub in ("IA", "AI", "Apps") for n in names]
    return out


def _comfy_launch_in(folder: Path, host: str, port: int) -> Launch | None:
    flags = ["--disable-auto-launch", "--listen", host, "--port", str(port)]
    portable = folder
    if not (folder / "python_embeded").is_dir() and (folder.parent / "python_embeded").is_dir():
        portable = folder.parent  # eligieron la carpeta ComfyUI de adentro
    py = portable / "python_embeded" / "python.exe"
    if py.is_file() and (portable / "ComfyUI" / "main.py").is_file():
        argv = [str(py), "-s", str(Path("ComfyUI") / "main.py"), "--windows-standalone-build", *flags]
        return Launch(argv, "portable", f"ComfyUI portable ({portable})", cwd=str(portable))
    if (folder / "main.py").is_file():
        py = _python_in(folder)
        if py:
            return Launch([str(py), "main.py", *flags], "git", f"ComfyUI con su entorno virtual ({folder})", cwd=str(folder))
    return None


def find_comfy(settings: Settings) -> Launch | None:
    host, port = host_port(settings.comfy_url, 8188)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if settings.comfy_comando.strip():
        cwd = settings.comfy_dir.strip() or None
        return Launch(split_cmd(settings.comfy_comando), "comando", "Comando personalizado", cwd=cwd, env=env)
    folders = [Path(settings.comfy_dir.strip().strip('"'))] if settings.comfy_dir.strip() else comfy_dir_candidates()
    for folder in folders:
        if folder.is_dir():
            launch = _comfy_launch_in(folder, host, port)
            if launch:
                launch.env = env
                return launch
    desktop = [local_appdata() / "Programs" / d / "ComfyUI.exe" for d in ("ComfyUI", "@comfyorgcomfyui-electron")]
    for exe in desktop:
        if exe.is_file():
            return Launch([str(exe)], "escritorio", f"ComfyUI Desktop ({exe}); usa el puerto 8000", console=False)
    return None


# Pistas del log de ComfyUI para que la espera diga en qué va
COMFY_HINTS = [
    ("To see the GUI go to", "Ya casi…"),
    ("Starting server", "Iniciando el servidor…"),
    ("Import times for custom nodes", "Cargando nodos…"),
    ("Using pytorch attention", "Preparando la GPU…"),
    ("Total VRAM", "Detectando la GPU…"),
    ("pytorch version", "Cargando PyTorch…"),
    ("Checkpoint files will always be loaded safely", "Cargando PyTorch…"),
]


# ------------------------------------------------------------------ gestor

class EngineManager:
    def __init__(self, logs_dir: Path | None = None):
        self.logs_dir = logs_dir or CACHE_DIR / "motores"
        self.procs: dict[str, subprocess.Popen] = {}
        self.kinds: dict[str, str] = {}
        self._lock = threading.Lock()

    def log_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def find(self, name: str, settings: Settings) -> Launch | None:
        return find_ollama(settings) if name == "ollama" else find_comfy(settings)

    def running(self, name: str, settings: Settings) -> bool:
        url = settings.ollama_url.rstrip("/") + "/api/version" if name == "ollama" else settings.comfy_url.rstrip("/") + "/system_stats"
        try:
            return httpx.get(url, timeout=2.0).status_code < 500
        except httpx.HTTPError:
            return False

    def ours(self, name: str) -> bool:
        proc = self.procs.get(name)
        return bool(proc and proc.poll() is None)

    def status(self, settings: Settings) -> dict:
        # en Windows, conectar a un puerto cerrado de localhost tarda ~2 s: se consultan a la vez
        with ThreadPoolExecutor(len(ENGINES)) as pool:
            up = dict(zip(ENGINES, pool.map(lambda n: self.running(n, settings), ENGINES)))
        out = {}
        for name in ENGINES:
            launch = self.find(name, settings)
            out[name] = {
                "nombre": NAMES[name],
                "encendido": up[name],
                "lanzado_por_plasmo": self.ours(name),
                "encontrado": launch is not None,
                "tipo": launch.kind if launch else None,
                "descripcion": launch.desc if launch else None,
                "log": str(self.log_path(name)),
            }
        return out

    def _tail(self, name: str, lines: int = 40) -> list[str]:
        try:
            data = self.log_path(name).read_bytes()[-12000:].decode("utf-8", "replace")
        except OSError:
            return []
        return [l.strip() for l in data.splitlines() if l.strip()][-lines:]

    def _hint(self, name: str) -> str | None:
        if name != "comfyui":
            return None
        tail = self._tail(name)
        for needle, text in COMFY_HINTS:
            if any(needle in line for line in tail):
                return text
        return None

    def start(self, name: str, settings: Settings, job: Job | None = None, phase: str | None = None) -> dict:
        if self.running(name, settings):
            return {"estado": "ya_estaba", "mensaje": f"{NAMES[name]} ya estaba encendido."}
        launch = self.find(name, settings)
        if launch is None:
            where = "la ruta de Ollama" if name == "ollama" else "la carpeta de ComfyUI"
            raise EngineError(f"No encontré {NAMES[name]} en esta PC. Indica {where} en Ajustes → Motores.")
        with self._lock:
            if not self.ours(name):
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                log = self.log_path(name).open("ab")
                log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} · {' '.join(launch.argv)}\n".encode())
                log.flush()
                kwargs = {"cwd": launch.cwd, "env": launch.env, "stdin": subprocess.DEVNULL}
                if launch.console:
                    kwargs.update(stdout=log, stderr=subprocess.STDOUT)
                else:
                    kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if WINDOWS:
                    kwargs["creationflags"] = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if launch.console else 0
                else:
                    kwargs["start_new_session"] = True
                try:
                    self.procs[name] = subprocess.Popen(launch.argv, **kwargs)
                except OSError as exc:
                    raise EngineError(f"No pude abrir {NAMES[name]}: {exc}") from exc
                finally:
                    log.close()
                self.kinds[name] = launch.kind
        proc = self.procs[name]
        started = time.time()
        while time.time() - started < TIMEOUT_S[name]:
            if job is not None:
                job.check()
                hint = self._hint(name)
                if hint and phase:
                    job.note(phase, f"{NAMES[name]}: {hint}")
            if self.running(name, settings):
                return {"estado": "encendido", "segundos": round(time.time() - started, 1), "tipo": launch.kind}
            if proc.poll() is not None and launch.console:
                tail = "\n".join(self._tail(name, 6))
                self.procs.pop(name, None)
                raise EngineError(f"{NAMES[name]} se cerró al arrancar (código {proc.returncode}). Últimas líneas del log:\n{tail}")
            time.sleep(0.5)
        raise EngineError(f"{NAMES[name]} no respondió en {TIMEOUT_S[name]} s. Revisa el log: {self.log_path(name)}")

    def stop(self, name: str) -> bool:
        """Apaga el motor solo si lo encendió Plasmo (con todo su árbol de procesos)."""
        proc = self.procs.get(name)
        if not proc or proc.poll() is not None:
            self.procs.pop(name, None)
            return False
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.procs.pop(name, None)
        return True


def start_many(manager: EngineManager, names: list[str], settings: Settings, job: Job) -> dict:
    """Enciende varios motores en orden; uno que falle no frena al otro."""
    results: dict[str, dict] = {}
    for name in names:
        job.start(name, f"Encendiendo {NAMES[name]}…")
        try:
            results[name] = manager.start(name, settings, job, name)
        except Cancelled:
            raise
        except EngineError as exc:
            results[name] = {"estado": "error", "mensaje": str(exc)}
            job.fail(name)
            continue
        job.done(name)
    failed = [r["mensaje"] for r in results.values() if r["estado"] == "error"]
    if failed and len(failed) == len(results):
        raise EngineError("\n\n".join(failed))
    return {"resultados": results}
