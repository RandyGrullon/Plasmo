"""Arranque de Plasmo: levanta el servidor local y abre la ventana de la app."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import httpx
import uvicorn

from plasmo import __version__

DEFAULT_PORT = 8765


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _already_running(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/api/ping", timeout=1.5).json().get("app") == "plasmo"
    except Exception:
        return False


def _pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        if _port_free(port):
            return port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(port: int) -> uvicorn.Server:
    from plasmo.app import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="plasmo-server").start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server


def _window(url: str) -> bool:
    try:
        import webview
    except Exception:
        return False
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True
    except Exception:
        pass
    webview.create_window("Plasmo", url, width=1400, height=900, min_size=(1024, 700), background_color="#0f1115")
    webview.start()
    return True


def _log_without_console() -> None:
    """El .exe y pythonw no tienen consola: los errores van a %LOCALAPPDATA%\\Plasmo\\plasmo.log."""
    if sys.stderr is None:
        from plasmo.config import CACHE_DIR

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log = open(CACHE_DIR / "plasmo.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stderr = log


def main(argv: list[str] | None = None) -> int:
    _log_without_console()
    parser = argparse.ArgumentParser(prog="plasmo", description="Imagen → modelo 3D para imprimir (local).")
    parser.add_argument("--puerto", type=int, default=DEFAULT_PORT)
    parser.add_argument("--navegador", action="store_true", help="abrir en el navegador en vez de ventana propia")
    parser.add_argument("--sin-ventana", action="store_true", help="solo el servidor (para pruebas)")
    parser.add_argument("--version", action="version", version=f"Plasmo {__version__}")
    args = parser.parse_args(argv)

    if _already_running(args.puerto):
        url = f"http://127.0.0.1:{args.puerto}/"
        if args.navegador or not _window(url):
            webbrowser.open(url)
        return 0

    port = _pick_port(args.puerto)
    server = _serve(port)
    url = f"http://127.0.0.1:{port}/"
    print(f"Plasmo {__version__} en {url}", flush=True)

    try:
        if args.sin_ventana:
            while True:
                time.sleep(3600)
        if args.navegador or not _window(url):
            webbrowser.open(url)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
