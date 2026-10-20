"""Proyectos en disco: una carpeta por imagen, con todo lo que se generó."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

_lock = threading.Lock()
SAFE_NAME = re.compile(r"^[\w\-. ]+$")


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, pid: str) -> Path:
        if not SAFE_NAME.match(pid) or ".." in pid:
            raise KeyError(pid)
        path = self.root / pid
        if not path.is_dir():
            raise KeyError(pid)
        return path

    def create(self, slug: str) -> tuple[str, Path]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pid = f"{stamp}-{slug}"[:64]
        path = self.root / pid
        n = 2
        while path.exists():
            path = self.root / f"{pid}-{n}"
            n += 1
        path.mkdir(parents=True)
        self.save_meta(path.name, {"id": path.name, "creado": time.time(), "nombre": slug, "versiones": []})
        return path.name, path

    def path(self, pid: str) -> Path:
        return self._dir(pid)

    def file(self, pid: str, name: str) -> Path:
        """Archivo del proyecto o de una versión ('v2-relieve/modelo.stl')."""
        bits = name.split("/")
        if len(bits) > 2 or any(not SAFE_NAME.match(b) or b in (".", "..") for b in bits):
            raise KeyError(name)
        p = self._dir(pid).joinpath(*bits)
        if not p.is_file():
            raise KeyError(name)
        return p

    def meta(self, pid: str) -> dict:
        try:
            return json.loads((self._dir(pid) / "proyecto.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {"id": pid}

    def save_meta(self, pid: str, data: dict) -> dict:
        with _lock:
            path = self.root / pid / "proyecto.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    def update_meta(self, pid: str, **changes) -> dict:
        data = self.meta(pid)
        data.update(changes)
        return self.save_meta(pid, data)

    def list(self, limit: int = 100) -> list[dict]:
        if not self.root.is_dir():
            return []
        items = []
        for p in sorted(self.root.iterdir(), reverse=True):
            if (p / "proyecto.json").is_file():
                m = self.meta(p.name)
                last = (m.get("versiones") or [{}])[-1]
                items.append({
                    "id": p.name,
                    "nombre": m.get("nombre"),
                    "creado": m.get("creado"),
                    "forma": last.get("forma"),
                    "miniatura": "recorte.png" if (p / "recorte.png").exists() else "original.png",
                    "tiene_modelo": bool(m.get("versiones")),
                })
            if len(items) >= limit:
                break
        return items
