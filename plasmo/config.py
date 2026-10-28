"""Configuración persistente de Plasmo (%APPDATA%\\Plasmo\\config.json)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _env_dir(var: str, fallback: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else fallback


HOME = Path.home()
CONFIG_DIR = Path(os.environ.get("PLASMO_CONFIG_DIR") or _env_dir("APPDATA", HOME / ".config") / "Plasmo")
CACHE_DIR = Path(os.environ.get("PLASMO_CACHE_DIR") or _env_dir("LOCALAPPDATA", HOME / ".cache") / "Plasmo")
MODELS_DIR = CACHE_DIR / "modelos"
DEFAULT_OUTPUT = HOME / "Documents" / "Plasmo"


@dataclass
class Printer:
    name: str = "Mi impresora"
    bed_x: float = 256.0
    bed_y: float = 256.0
    bed_z: float = 256.0
    nozzle: float = 0.4
    kind: str = "fdm"  # fdm | sla
    material: str = "PLA"


@dataclass
class Settings:
    ollama_url: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen3.5:9b"
    chat_model: str = ""  # vacío = usa el de visión
    generator: str = "comfyui"  # comfyui | hunyuan_api
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_checkpoint: str = "hunyuan_3d_v2.1.safetensors"
    comfy_t2i_checkpoint: str = ""
    hunyuan_api_url: str = "http://127.0.0.1:8081"
    bg_model: str = "auto"  # máxima calidad: IS-Net + BiRefNet combinados
    depth_model: str = "depth-anything-v2-small"
    output_dir: str = str(DEFAULT_OUTPUT)
    tutorial_visto: bool = False
    # motores: vacío = Plasmo los busca solo; los comandos personalizados mandan si existen
    ollama_exe: str = ""
    comfy_dir: str = ""
    ollama_comando: str = ""
    comfy_comando: str = ""
    encender_al_abrir: bool = False
    printer: Printer = field(default_factory=Printer)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known and k != "printer"}
        printer_known = {f.name for f in fields(Printer)}
        printer = Printer(**{k: v for k, v in (data or {}).get("printer", {}).items() if k in printer_known})
        if not str(clean.get("output_dir") or "").strip():
            clean.pop("output_dir", None)  # vacío = carpeta por defecto, nunca el directorio actual
        return cls(**clean, printer=printer)


class ConfigStore:
    def __init__(self, path: Path | None = None):
        self.path = path or CONFIG_DIR / "config.json"
        self._lock = threading.Lock()
        self._settings = self._load()

    def _load(self) -> Settings:
        try:
            # utf-8-sig: acepta el BOM que agregan el Bloc de notas o PowerShell
            return Settings.from_dict(json.loads(self.path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError, TypeError):
            return Settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def update(self, data: dict) -> Settings:
        with self._lock:
            merged = self._settings.to_dict()
            for key, value in (data or {}).items():
                if key == "printer" and isinstance(value, dict):
                    merged["printer"].update(value)
                else:
                    merged[key] = value
            self._settings = Settings.from_dict(merged)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            return self._settings
