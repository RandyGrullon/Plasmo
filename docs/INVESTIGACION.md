# Investigación (20 de octubre de 2026)

Qué hace Meshy, qué se pudo ver con REA y por qué Plasmo usa los modelos que usa en una **RTX 5070 de 12 GB**.

## 1. Meshy analizado con REA

[REA (Reverse Engineer Anything)](https://github.com/morluto/rea) es un servidor MCP y CLI para hacer ingeniería inversa de apps, webs y binarios. Pide Node ≥ 22.19; se corrió con un Node 22 temporal (`npx -p node@22.20.0 -p rea-agents@latest -- rea …`), sin instalar nada en el sistema.

Se ejecutó `rea analyze-web-bundle` contra la **página pública** de meshy.ai, en un Edge headless en modo InPrivate, sin cuenta y sin tocar su API:

- 66 scripts capturados y analizados (análisis completo, sin fallos de parseo).
- Lo que se ve: un sitio **Next.js** con analítica **PostHog** servida desde su propio dominio (`/ingest/…`, encuestas, *autocapture* de clics). No aparecen rutas ni endpoints de generación.
- Conclusión: **la generación 3D ocurre toda en sus servidores**, así que el front no revela cómo la hacen. La ficha técnica se sacó de su [API pública](https://docs.meshy.ai/), que sí documenta cada opción.

## 2. Qué ofrece Meshy y qué hace Plasmo con eso

| Meshy | Plasmo |
|---|---|
| Image to 3D (Meshy 6/7, `should_texture`, `target_formats` incl. STL/3MF) | **Figura 3D** con Hunyuan3D 2.1, solo geometría (para imprimir no hace falta textura) |
| Text to 3D / Text to Image | **Texto → imagen → 3D**: Ollama escribe el prompt y un checkpoint SDXL en ComfyUI dibuja (opcional) |
| Remesh / `decimation_mode` / `target_polycount` | Simplificación cuádrica que no abre la malla (`fast-simplification`) |
| **Analyze Printability**: watertight, volumen, aristas no manifold, caras degeneradas, agujeros, paredes finas (>5 % = error; FDM 1.0 mm, SLA 0.8 mm) | Mismas métricas y umbrales, más **voladizos sin apoyo** (rayos), **estabilidad** (centro de masa sobre el apoyo), **cabe en la cama** y **peso** |
| Repair Printability | Reparar: quitar fragmentos, tapar agujeros y, si hace falta, rehacer la malla por vóxeles |
| Balance / `auto_size` / `origin_at: bottom` | Escalar en mm, aplanar la base, pedestal de disco o rectangular |
| Auto Split | Dividir por planos para que cada parte quepa en la cama |
| Multi-Color Print | **Pixel art multicolor**: 3MF con una pieza por color, ensambladas para el AMS/MMU |
| Creative Lab: **Keychain** (relieve de 2.2 mm, base, `badge_shape`, curva lineal/gamma/S, suavizado, umbral de profundidad) | **Llavero** y **Relieve** con los mismos controles, más argolla con agujero |
| Creative Lab: Fridge Magnet / Lamp / Fidget Pixel | **Imán** (bolsillo para el imán), **Litofanía** plana/curva/**cilindro para lámpara**, **Pixel art** |
| — | **Cortador de galletas** (extra) |
| Texturizado PBR, rigging, animación | Fuera de alcance: no aportan nada a una pieza impresa |

## 3. Generador 3D para 12 GB

| # | Modelo | Licencia | VRAM (solo forma) | ComfyUI | Veredicto |
|---|---|---|---|---|---|
| 1 | **Hunyuan3D 2.1** (jun. 2025) | Tencent Hunyuan Community: uso comercial OK; no vale en UE/UK/Corea del Sur | ~10 GB | **Nativo** (plantilla oficial) | **Elegido**: grafo de 10 nodos sin extensiones que compilar y mallas cerradas, buenas para imprimir |
| 2 | Hunyuan3D 2.0 | Igual que 2.1 | ~6 GB | Nativo | **Respaldo** si falta VRAM |
| 3 | TRELLIS.2 / Pixal3D (dic. 2025 / may. 2026) | MIT (el codificador DINOv3 tiene licencia de Meta) | Pide ≥ 24 GB; en 12 GB no está probado | Nativo desde ago. 2026 | Futuro: más detalle, pero sus mallas salen con un hueco interior y no son cerradas (issue #16147) |
| 4 | Hunyuan3D 2mini | Igual que 2.0 | ~5 GB | Lo detecta, sin plantilla | Sin probar |
| — | TripoSG, Step1X-3D, Direct3D-S2 | MIT / Apache | 8–27 GB | Solo nodos externos | Necesitan compilar extensiones CUDA; frágiles en Windows con Blackwell |
| — | TripoSR, SF3D, SPAR3D | MIT / Stability | ~6–10 GB | — | Más viejos y de menor calidad |

**No hay** un Hunyuan3D más nuevo con pesos abiertos: el 2.5, el 3.0 y el 3.1 son solo por API.

Para Windows con RTX 50 se usa **ComfyUI portable v0.42.1** (PyTorch cu130, driver R580+). El `api_server.py` de Hunyuan3D-2.1 no sirve en 12 GB: ignora los parámetros y siempre texturiza (~21 GB). El de Hunyuan3D-2 sí sirve (usa 2mini-turbo), y Plasmo lo soporta como segundo motor.

## 4. Ollama: el modelo que "lee" la foto

Ollama no genera mallas, y desde la v0.32.6 tampoco imágenes. En Plasmo **ve la foto** (descripción, forma recomendada, tamaño, soportes, consejos, en JSON con esquema) y **conversa**.

| Modelo | Descarga | 12 GB a 8k | Nota |
|---|---|---|---|
| **`qwen3.5:9b`** | 6.6 GB | Sí (~8 GB) | **Recomendado.** Multimodal nativo (feb. 2026): MMMU-Pro 70.1, IFEval 91.5, 201 idiomas |
| `qwen3.5:4b` | 3.3 GB | Sí | Liviano; MMMU-Pro 66.3 |
| `gemma4:12b` | 8.0 GB | Justo | Para comparar redacción |
| `qwen3-vl:8b-instruct` | 6.2 GB | Sí | Ojo: `qwen3-vl:8b` a secas es la variante que "piensa" |

Detalles que Plasmo ya aplica:

- Ollama enciende el "pensar" por defecto en local: se manda `think: false`.
- El contexto por defecto es de 4k en GPUs de menos de 24 GB: se pide `num_ctx: 8192`.
- Las imágenes van en base64 plano, sin `data:`.
- Antes de usar la GPU para el 3D, el modelo se descarga de la VRAM con `keep_alive: 0`.

## 5. Lo que no se pudo verificar desde la oficina

La laptop de la oficina no tiene GPU NVIDIA. Todo el flujo se probó con **simuladores** de Ollama y ComfyUI (`tools/`), con las mismas rutas, mensajes de WebSocket y formatos que los reales. Quedan por medir en la PC de casa:

- Tiempos y VRAM reales de Hunyuan3D 2.1 y de `qwen3.5:9b` en la RTX 5070.
- La calidad del análisis en español de `qwen3.5:9b` con fotos reales.
- Ningún flujo de ComfyUI se corrió en una GPU: el grafo se armó a partir del código fuente y la plantilla oficial.

## Fuentes

- REA: <https://github.com/morluto/rea> (docs/cli.md)
- Meshy: <https://www.meshy.ai/>, <https://docs.meshy.ai/en/api/image-to-3d>, <https://docs.meshy.ai/en/api/analyze-printability>, <https://docs.meshy.ai/en/api/creative-lab-keychain>
- ComfyUI: <https://github.com/Comfy-Org/ComfyUI> (`comfy_extras/nodes_hunyuan3d.py`, `nodes_save_3d.py`, `server.py`), <https://github.com/Comfy-Org/workflow_templates>, <https://docs.comfy.org/tutorials/3d/hunyuan3D-2>, <https://blog.comfy.org/p/trellis2-and-pixal3d-are-now-native>
- Pesos: <https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged>, <https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged>, <https://huggingface.co/tencent/Hunyuan3D-2mini>
- Hunyuan3D: <https://github.com/Tencent-Hunyuan/Hunyuan3D-2>, <https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1>
- Ollama: <https://github.com/ollama/ollama/releases> (v0.32.6, v0.40.6), <https://docs.ollama.com/api/chat>, <https://docs.ollama.com/capabilities/structured-outputs>, <https://docs.ollama.com/context-length>, <https://ollama.com/library/qwen3.5>, <https://ollama.com/library/gemma4>
- Modelos ONNX: <https://github.com/danielgatis/rembg>, <https://huggingface.co/onnx-community/depth-anything-v2-small>
