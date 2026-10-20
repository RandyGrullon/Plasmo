"""Trabajos en segundo plano con barra de progreso total por fases ponderadas.

Cada fase tiene un peso. Las que reportan pasos (p. ej. el muestreo de ComfyUI)
avanzan con datos reales; las que no, se estiman con la duración media de
corridas anteriores (guardada en tiempos.json) y nunca pasan del 95 %.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from plasmo.config import CONFIG_DIR


class Cancelled(Exception):
    pass


@dataclass
class Phase:
    key: str
    label: str
    weight: float
    progress: float = 0.0
    state: str = "pendiente"  # pendiente | activa | lista | omitida | error
    started: float = 0.0
    measured: bool = False  # True cuando la fase reporta pasos reales
    detail: str = ""


class Timings:
    def __init__(self, path: Path | None = None):
        self.path = path or CONFIG_DIR / "tiempos.json"
        self._lock = threading.Lock()
        try:
            self.data: dict[str, float] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def expected(self, key: str, default: float = 10.0) -> float:
        return self.data.get(key, default)

    def record(self, key: str, seconds: float) -> None:
        with self._lock:
            old = self.data.get(key)
            self.data[key] = seconds if old is None else old * 0.6 + seconds * 0.4
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
            except OSError:
                pass


DEFAULT_SECONDS = {
    "preparar": 4, "analizar_ia": 25, "liberar_vram": 2, "generar": 90, "convertir": 4, "reparar": 12,
    "imprimir": 6, "profundidad": 4, "construir": 6, "simplificar": 6, "revisar": 6, "exportar": 5,
    "prompt": 15, "imagen": 40, "descargar": 60,
}


@dataclass
class Job:
    id: str
    kind: str
    phases: list[Phase]
    timings: Timings
    status: str = "en_cola"  # en_cola | corriendo | listo | error | cancelado
    message: str = ""
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def _phase(self, key: str) -> Phase:
        for p in self.phases:
            if p.key == key:
                return p
        raise KeyError(key)

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def start(self, key: str, message: str = "") -> None:
        self.check()
        p = self._phase(key)
        p.state, p.started, p.progress = "activa", time.time(), 0.0
        self.message = message or p.label

    def update(self, key: str, fraction: float, detail: str = "") -> None:
        self.check()
        p = self._phase(key)
        p.measured = True
        p.progress = max(0.0, min(1.0, fraction))
        if detail:
            p.detail = detail
            self.message = detail

    def done(self, key: str) -> None:
        p = self._phase(key)
        if p.state == "activa":
            self.timings.record(f"{self.kind}:{key}", time.time() - p.started)
        p.state, p.progress = "lista", 1.0

    def skip(self, key: str) -> None:
        p = self._phase(key)
        p.state, p.progress = "omitida", 1.0

    def percent(self) -> float:
        total = sum(p.weight for p in self.phases) or 1.0
        acc = 0.0
        now = time.time()
        for p in self.phases:
            prog = p.progress
            if p.state == "activa" and not p.measured:
                expected = self.timings.expected(f"{self.kind}:{p.key}", DEFAULT_SECONDS.get(p.key, 10))
                prog = max(prog, min(0.95, (now - p.started) / max(expected, 0.5)))
            acc += p.weight * prog
        return round(acc / total * 100, 1) if self.status != "listo" else 100.0

    def eta_seconds(self) -> float | None:
        pct = self.percent()
        if self.status != "corriendo" or pct <= 2:
            return None
        elapsed = time.time() - self.created
        return round(elapsed * (100 - pct) / pct, 0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tipo": self.kind,
            "estado": self.status,
            "porcentaje": self.percent(),
            "mensaje": self.message,
            "eta_s": self.eta_seconds(),
            "fases": [
                {"clave": p.key, "nombre": p.label, "estado": p.state, "progreso": round(p.progress, 3), "detalle": p.detail}
                for p in self.phases
            ],
            "resultado": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, workers: int = 2, timings: Timings | None = None):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="plasmo")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.timings = timings or Timings()

    def submit(self, kind: str, phases: list[tuple[str, str, float]], fn: Callable[[Job], dict | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, phases=[Phase(k, l, w) for k, l, w in phases], timings=self.timings)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()

        def run():
            job.status = "corriendo"
            try:
                job.result = fn(job) or {}
                for p in job.phases:
                    if p.state in ("pendiente", "activa"):
                        job.done(p.key) if p.state == "activa" else job.skip(p.key)
                job.status, job.message = "listo", "Listo"
            except Cancelled:
                job.status, job.message = "cancelado", "Cancelado"
            except Exception as exc:  # el mensaje llega tal cual a la interfaz
                job.status, job.error = "error", str(exc) or exc.__class__.__name__
                job.message = job.error
                for p in job.phases:
                    if p.state == "activa":
                        p.state = "error"
                traceback.print_exc()
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status not in ("en_cola", "corriendo"):
            return False
        job.cancel_event.set()
        return True

    def _prune(self, keep: int = 60) -> None:
        if len(self._jobs) <= keep:
            return
        old = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.finished)
        for j in old[: len(self._jobs) - keep]:
            self._jobs.pop(j.id, None)
