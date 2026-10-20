"""Modo demo: Plasmo con Ollama y ComfyUI simulados y una configuración temporal.

Sirve para ver la app completa en una PC sin GPU ni modelos:

    uv run python tools/demo.py              # abre la ventana
    uv run python tools/demo.py --sin-ventana
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / "plasmo-demo"
os.environ["PLASMO_CONFIG_DIR"] = str(ROOT / "config")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mock_comfyui, mock_ollama  # noqa: E402

OLLAMA_PORT, COMFY_PORT = 11500, 8199


def main() -> int:
    (ROOT / "config").mkdir(parents=True, exist_ok=True)
    cfg_file = ROOT / "config" / "config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    cfg.update({
        "ollama_url": f"http://127.0.0.1:{OLLAMA_PORT}",
        "comfy_url": f"http://127.0.0.1:{COMFY_PORT}",
        "generator": "comfyui",
        "comfy_t2i_checkpoint": "sd_xl_base_1.0.safetensors",
        "output_dir": str(ROOT / "proyectos"),
    })
    cfg_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    mock_comfyui.STEP_DELAY = 0.15
    mock_ollama.serve_in_thread(OLLAMA_PORT)
    mock_comfyui.serve_in_thread(COMFY_PORT)
    print(f"Demo: Ollama simulado :{OLLAMA_PORT} · ComfyUI simulado :{COMFY_PORT} · datos en {ROOT}", flush=True)

    from plasmo.__main__ import main as plasmo_main

    return plasmo_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
