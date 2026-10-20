# Plasmo

**Sube una imagen y recibe una pieza lista para imprimir en 3D.** Todo corre en tu PC: Ollama mira la foto y te aconseja, ComfyUI + Hunyuan3D genera la figura, y Plasmo la repara, la escala y la exporta en STL/3MF para tu slicer.

Es una alternativa personal y gratuita a [Meshy](https://www.meshy.ai/), enfocada en impresión 3D.

![Plasmo](assets/plasmo.png)

## Qué hace

| Forma | Qué sale | Necesita |
|---|---|---|
| **Figura 3D (IA)** | Modelo completo del objeto o personaje de la foto, con base y aplanado opcionales | ComfyUI + Hunyuan3D 2.1 (GPU) |
| **Relieve** | Placa con la imagen en relieve: silueta, rectángulo, círculo, hexágono, escudo, estrella o corazón | Nada (CPU) |
| **Litofanía** | Lámina que revela la foto con luz detrás: plana, curva o cilindro para lámpara | Nada |
| **Llavero** | Silueta o medalla con argolla | Nada |
| **Imán de nevera** | Pieza con bolsillo trasero para un imán de neodimio | Nada |
| **Pixel art multicolor** | Bloques de colores; cada color es una pieza del 3MF para AMS/MMU | Nada |
| **Cortador de galletas** | Filo con la silueta y pestaña para empujar | Nada |

Además, en cada pieza:

- **Quita el fondo** con IS-Net, BiRefNet o U2-Net (ONNX en CPU) y calcula la **profundidad** con Depth Anything V2 para los relieves.
- **Analiza la foto con Ollama** (visión): qué es, qué forma conviene, tamaño, soportes, material y consejos. Y trae un **chat** que conoce tu proyecto.
- **Revisa la imprimibilidad** como el "Analyze Printability" de Meshy: malla cerrada, aristas no manifold, agujeros, caras degeneradas, paredes finas, voladizos sin apoyo, estabilidad, si cabe en la cama y peso estimado.
- **Repara y prepara**: elimina fragmentos sueltos, tapa agujeros, rehace por vóxeles si hace falta, simplifica, escala en mm, aplana la base, agrega pedestal y divide en partes si no cabe.
- **Exporta** STL, 3MF (multipieza con colores), OBJ y ZIP de piezas, y lo abre en tu slicer.
- **Barra de progreso con % total**, con los pasos reales de ComfyUI por WebSocket. Las fases sin pasos se estiman por la duración de las corridas anteriores.

## Cómo funciona

```
 foto ─► Plasmo (FastAPI + ventana WebView2)
          ├─ recorte y profundidad ── ONNX en CPU (IS-Net, Depth Anything V2)
          ├─ "¿qué ves?" ──────────── Ollama · qwen3.5:9b      (http://127.0.0.1:11434)
          ├─ figura 3D ────────────── ComfyUI · Hunyuan3D 2.1  (http://127.0.0.1:8188)
          └─ formas, reparación, análisis y exportación ── trimesh + manifold3d
```

Plasmo no carga PyTorch: habla con Ollama y ComfyUI por HTTP, así que la app pesa poco y cada motor se actualiza por su lado. Antes de generar la figura, Plasmo descarga el modelo de Ollama de la VRAM, y al terminar le pide a ComfyUI que libere la suya. Así los 12 GB de la RTX 5070 alcanzan para todo.

## Instalación (PC de casa)

1. **La IA (una sola vez):** sigue [docs/INSTALAR-IA.md](docs/INSTALAR-IA.md): driver NVIDIA, Ollama con `qwen3.5:9b`, y ComfyUI portable con `hunyuan_3d_v2.1.safetensors`.
2. **Plasmo:**
   ```powershell
   git clone https://github.com/RandyGrullon/Plasmo.git
   cd Plasmo
   powershell -ExecutionPolicy Bypass -File scripts\instalar.ps1
   ```
   El instalador pone [uv](https://docs.astral.sh/uv/) si falta, instala las dependencias y crea el acceso directo **Plasmo** en el escritorio.
3. **Abrir:** doble clic en el acceso directo o en `Plasmo.bat`. En **Ajustes** confirma las direcciones de Ollama y ComfyUI, el modelo de visión y las medidas de tu cama.

### Ejecutable (.exe)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\construir-exe.ps1
```

Deja `dist\Plasmo\Plasmo.exe` (carpeta portable de unos 400 MB) y su acceso directo en el escritorio. Si algo falla al abrir, mira `%LOCALAPPDATA%\Plasmo\plasmo.log`.

## Probar sin GPU (modo demo)

```powershell
uv run python tools/demo.py
```

Levanta un Ollama y un ComfyUI **simulados** (el 3D simulado "infla" la silueta) y abre Plasmo con datos en `%TEMP%\plasmo-demo`. Sirve para ver la app completa en cualquier PC.

## Desarrollo

```powershell
uv sync --group dev
uv run pytest            # 33 pruebas: geometría, 3MF, API completa contra los simuladores
uv run python -m plasmo --navegador   # abre en el navegador en vez de ventana propia
```

| Carpeta | Contenido |
|---|---|
| `plasmo/mesh/` | Sólidos desde mapas de altura, formas, imprimibilidad y exportación |
| `plasmo/imaging/` | Recorte, profundidad y descarga de modelos ONNX |
| `plasmo/generators.py` | ComfyUI (Hunyuan3D 2.0/2.1, texto→imagen) y servidor Hunyuan3D-2 |
| `plasmo/ollama_client.py` | Visión con salida JSON, chat por streaming y descarga de modelos |
| `plasmo/web/` | Interfaz (HTML/CSS/JS sin compilación + three.js r170) |
| `tools/` | Simuladores de Ollama y ComfyUI y modo demo |

## Investigación

Por qué estos modelos y no otros, qué hace Meshy y cómo se analizó con [REA](https://github.com/morluto/rea): [docs/INVESTIGACION.md](docs/INVESTIGACION.md).

## Licencias

El código de Plasmo es MIT. Los modelos tienen sus propias licencias:

- **Hunyuan3D 2.0/2.1** (Tencent Hunyuan Community License): permite uso comercial, pero **no vale en la UE, el Reino Unido ni Corea del Sur**.
- **Qwen3.5 / Gemma 4**: revisa la licencia de cada uno en su página de ollama.com.
- **IS-Net / U2-Net**: Apache 2.0. **BiRefNet**: MIT.
- **Depth Anything V2 Small**: Apache 2.0. Los tamaños Base y Large no permiten uso comercial; Plasmo usa el Small.
- **three.js**: MIT.
