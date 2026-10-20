"""Cliente de Ollama (API nativa /api/*): visión, chat y descarga de modelos.

Ollama no genera mallas ni (desde la v0.32.6) imágenes: aquí solo "lee" la foto,
recomienda la forma de impresión y conversa sobre impresión 3D.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Iterator

import httpx
from PIL import Image

RECOMMENDED = [
    {"tag": "qwen3.5:9b", "size_gb": 6.6, "label": "Recomendado: ve imágenes, buen español, cabe en 12 GB"},
    {"tag": "qwen3.5:4b", "size_gb": 3.3, "label": "Liviano: misma familia, más rápido"},
    {"tag": "gemma4:12b", "size_gb": 8.0, "label": "Alternativa para comparar redacción"},
]

FORM_KEYS = ["figura", "relieve", "litofania", "llavero", "iman", "pixel", "cortador"]
PROBLEMS = ["varios_objetos", "fondo_recargado", "angulo_malo", "desenfocada", "poca_luz", "objeto_cortado", "transparente_o_reflectante"]

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "descripcion": {"type": "string"},
        "sujeto": {"type": "string"},
        "apta_para_figura": {"type": "boolean"},
        "problemas": {"type": "array", "items": {"type": "string", "enum": PROBLEMS}},
        "forma_recomendada": {"type": "string", "enum": FORM_KEYS},
        "por_que": {"type": "string"},
        "tamano_mm": {"type": "number"},
        "orientacion": {"type": "string"},
        "soportes": {"type": "string", "enum": ["ninguno", "minimos", "necesarios"]},
        "material": {"type": "string"},
        "nombre_archivo": {"type": "string"},
        "consejos": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "descripcion", "sujeto", "apta_para_figura", "problemas", "forma_recomendada", "por_que",
        "tamano_mm", "orientacion", "soportes", "material", "nombre_archivo", "consejos",
    ],
}

ANALYSIS_SYSTEM = """Eres el asistente de Plasmo, una app que convierte fotos en piezas para impresión 3D.
Responde SIEMPRE en español y SOLO con JSON que cumpla el esquema.
Formas disponibles:
- figura: modelo 3D completo generado por IA (necesita UN objeto o personaje, bien iluminado, entero, fondo simple).
- relieve: placa con el dibujo en relieve (sirve con casi cualquier imagen).
- litofania: lámina fina que muestra la foto al ponerle luz detrás (ideal para fotos de personas o paisajes).
- llavero: silueta o placa pequeña con argolla.
- iman: pieza plana con bolsillo para imán de nevera.
- pixel: pixel art en bloques de colores (logos, dibujos simples, personajes de videojuego).
- cortador: cortador de galletas con la silueta (formas simples y cerradas).
Da medidas en milímetros pensando en una impresora FDM con cama de 256 mm."""

ANALYSIS_PROMPT = "Analiza esta imagen para imprimirla en 3D. Esquema JSON a cumplir:\n" + json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False)

CHAT_SYSTEM = """Eres el asistente de Plasmo, experto en impresión 3D (FDM y resina), slicers (Bambu Studio, OrcaSlicer, Cura, PrusaSlicer),
materiales y modelado. Responde en español dominicano neutro, claro y breve, con pasos concretos y valores numéricos.
Si te paso el contexto del proyecto actual (análisis de la foto y reporte de imprimibilidad), úsalo."""

PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt_en": {"type": "string"}, "nombre": {"type": "string"}},
    "required": ["prompt_en", "nombre"],
}


class OllamaError(RuntimeError):
    pass


def image_b64(img: Image.Image, max_side: int = 1024) -> str:
    """Ollama espera base64 plano (sin 'data:'); 1024 px bastan para describir."""
    rgb = Image.new("RGB", img.size, (255, 255, 255))
    rgb.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._caps: dict[str, list[str]] = {}

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout or self.timeout, connect=5.0))

    # --------------------------------------------------------------- estado
    def version(self) -> str | None:
        try:
            with self._client(5) as c:
                return c.get("/api/version").json().get("version")
        except (httpx.HTTPError, ValueError):
            return None

    def models(self) -> list[dict]:
        with self._client(10) as c:
            data = c.get("/api/tags").json()
        out = []
        for m in data.get("models", []):
            d = m.get("details") or {}
            out.append({
                "name": m.get("name") or m.get("model"),
                "size_gb": round((m.get("size") or 0) / 1e9, 1),
                "family": d.get("family"),
                "params": d.get("parameter_size"),
                "quant": d.get("quantization_level"),
            })
        return out

    def capabilities(self, model: str) -> list[str]:
        if model not in self._caps:
            with self._client(15) as c:
                r = c.post("/api/show", json={"model": model})
                if r.status_code == 404:
                    raise OllamaError(f"El modelo {model} no está descargado en Ollama.")
                r.raise_for_status()
                self._caps[model] = list(r.json().get("capabilities") or [])
        return self._caps[model]

    def loaded(self) -> list[dict]:
        with self._client(5) as c:
            return c.get("/api/ps").json().get("models", [])

    # -------------------------------------------------------------- modelos
    def pull(self, model: str) -> Iterator[dict]:
        """Descarga un modelo; emite {status, completed, total, percent} sumando capas."""
        layers: dict[str, tuple[int, int]] = {}
        with self._client(None) as c, c.stream("POST", "/api/pull", json={"model": model, "stream": True}, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(msg["error"])
                if msg.get("digest") and msg.get("total"):
                    layers[msg["digest"]] = (int(msg.get("completed") or 0), int(msg["total"]))
                done = sum(a for a, _ in layers.values())
                total = sum(b for _, b in layers.values())
                yield {"status": msg.get("status", ""), "completed": done, "total": total, "percent": round(done / total * 100, 1) if total else 0.0}

    def unload(self, model: str) -> None:
        """Libera la VRAM (keep_alive: 0) antes de generar el 3D en la misma GPU."""
        try:
            with self._client(20) as c:
                c.post("/api/chat", json={"model": model, "messages": [], "keep_alive": 0})
        except httpx.HTTPError:
            pass

    # ----------------------------------------------------------------- chat
    def _payload(self, model: str, messages: list[dict], fmt=None, stream=False, temperature=0.2) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "10m",
            "options": {"num_ctx": 8192, "temperature": temperature},
        }
        # Ollama enciende el "pensar" por defecto en local; lo apagamos para JSON rápido
        try:
            if "thinking" in self.capabilities(model):
                payload["think"] = False
        except (httpx.HTTPError, OllamaError):
            pass
        if fmt is not None:
            payload["format"] = fmt
        return payload

    def chat(self, model: str, messages: list[dict], fmt=None, temperature: float = 0.2) -> str:
        with self._client() as c:
            r = c.post("/api/chat", json=self._payload(model, messages, fmt, False, temperature))
        if r.status_code >= 400:
            raise OllamaError(_error_text(r))
        data = r.json()
        if "error" in data:
            raise OllamaError(data["error"])
        return (data.get("message") or {}).get("content", "")

    def chat_stream(self, model: str, messages: list[dict], temperature: float = 0.6) -> Iterator[str]:
        payload = self._payload(model, messages, None, True, temperature)
        with self._client() as c, c.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code >= 400:
                r.read()
                raise OllamaError(_error_text(r))
            for line in r.iter_lines():
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(msg["error"])
                piece = (msg.get("message") or {}).get("content")
                if piece:
                    yield piece

    # --------------------------------------------------------- tareas Plasmo
    def analyze_image(self, model: str, img: Image.Image) -> dict:
        caps = self.capabilities(model)
        if "vision" not in caps:
            raise OllamaError(f"{model} no ve imágenes; usa uno con visión, por ejemplo qwen3.5:9b.")
        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": ANALYSIS_PROMPT, "images": [image_b64(img)]},
        ]
        raw = self.chat(model, messages, fmt=ANALYSIS_SCHEMA, temperature=0.2)
        return normalize_analysis(_loads(raw))

    def image_prompt(self, model: str, idea: str) -> dict:
        messages = [
            {"role": "system", "content": "Convierte la idea del usuario en un prompt en inglés para un generador de imágenes. "
             "La imagen se usará para crear un modelo 3D: un solo objeto o personaje, entero, centrado, vista 3/4, "
             "fondo blanco liso, luz de estudio suave, sin sombras duras, sin texto. Responde solo JSON."},
            {"role": "user", "content": idea},
        ]
        data = _loads(self.chat(model, messages, fmt=PROMPT_SCHEMA, temperature=0.4))
        prompt = str(data.get("prompt_en") or idea).strip()
        return {"prompt": prompt, "nombre": slug(str(data.get("nombre") or idea))}


def _error_text(r: httpx.Response) -> str:
    try:
        return r.json().get("error") or r.text
    except ValueError:
        return r.text or f"HTTP {r.status_code}"


def _loads(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise OllamaError("El modelo no devolvió JSON válido.")


def slug(text: str, limit: int = 40) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return (t[:limit].strip("-")) or "modelo"


def normalize_analysis(data: dict) -> dict:
    """Valida lo que devuelve el modelo y rellena lo que falte con valores seguros."""
    form = data.get("forma_recomendada")
    if form not in FORM_KEYS:
        form = "figura" if data.get("apta_para_figura") else "relieve"
    try:
        size = float(data.get("tamano_mm") or 80)
    except (TypeError, ValueError):
        size = 80.0
    return {
        "descripcion": str(data.get("descripcion") or "").strip(),
        "sujeto": str(data.get("sujeto") or "").strip(),
        "apta_para_figura": bool(data.get("apta_para_figura")),
        "problemas": [p for p in (data.get("problemas") or []) if p in PROBLEMS],
        "forma_recomendada": form,
        "por_que": str(data.get("por_que") or "").strip(),
        "tamano_mm": max(10.0, min(size, 250.0)),
        "orientacion": str(data.get("orientacion") or "").strip(),
        "soportes": data.get("soportes") if data.get("soportes") in ("ninguno", "minimos", "necesarios") else "minimos",
        "material": str(data.get("material") or "PLA").strip(),
        "nombre_archivo": slug(str(data.get("nombre_archivo") or data.get("sujeto") or "modelo")),
        "consejos": [str(c) for c in (data.get("consejos") or [])][:6],
    }
