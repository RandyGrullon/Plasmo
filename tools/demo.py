"""Modo demo: Plasmo con Ollama y ComfyUI simulados y una configuración temporal.

Sirve para ver la app completa en una PC sin GPU ni modelos:

    uv run python tools/demo.py                      # abre la ventana
    uv run python tools/demo.py --sin-ventana
    uv run python tools/demo.py --motores-apagados   # para probar el botón «Encender IA»
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / "plasmo-demo"
TOOLS = Path(__file__).resolve().parent
os.environ["PLASMO_CONFIG_DIR"] = str(ROOT / "config")
sys.path.insert(0, str(TOOLS.parent))

from tools import mock_comfyui, mock_ollama  # noqa: E402


def free_port(preferred: int) -> int:
    """El preferido si está libre (p. ej. quedó otro demo abierto); si no, uno cualquiera."""
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("sin puertos libres")


OLLAMA_PORT, COMFY_PORT = free_port(11500), free_port(8199)


def mock_cmd(script: str, *args: str) -> str:
    return f'"{sys.executable}" "{TOOLS / script}" ' + " ".join(args)


def main() -> int:
    args = sys.argv[1:]
    engines_off = "--motores-apagados" in args
    args = [a for a in args if a != "--motores-apagados"]

    (ROOT / "config").mkdir(parents=True, exist_ok=True)
    cfg_file = ROOT / "config" / "config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig")) if cfg_file.exists() else {}
    cfg.update({
        "ollama_url": f"http://127.0.0.1:{OLLAMA_PORT}",
        "comfy_url": f"http://127.0.0.1:{COMFY_PORT}",
        "generator": "comfyui",
        "comfy_t2i_checkpoint": "sd_xl_base_1.0.safetensors",
        "output_dir": str(ROOT / "proyectos"),
        # con los motores apagados, «Encender IA» levanta los simuladores como procesos de verdad
        "ollama_comando": mock_cmd("mock_ollama.py", "--port", str(OLLAMA_PORT), "--pensar", "2.5") if engines_off else "",
        "comfy_comando": mock_cmd("mock_comfyui.py", "--port", str(COMFY_PORT), "--paso", "0.18", "--arranque", "6") if engines_off else "",
    })
    cfg_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if engines_off:
        print(f"Demo con motores apagados (Ollama :{OLLAMA_PORT}, ComfyUI :{COMFY_PORT}) · datos en {ROOT}", flush=True)
    else:
        mock_comfyui.STEP_DELAY = 0.18
        mock_ollama.THINK_SECONDS = 2.5
        mock_ollama.serve_in_thread(OLLAMA_PORT)
        mock_comfyui.serve_in_thread(COMFY_PORT)
        print(f"Demo: Ollama simulado :{OLLAMA_PORT} · ComfyUI simulado :{COMFY_PORT} · datos en {ROOT}", flush=True)

    from plasmo.__main__ import main as plasmo_main

    return plasmo_main(args)


if __name__ == "__main__":
    sys.exit(main())
