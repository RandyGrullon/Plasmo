# Instalar la IA en la PC de casa

Guía para la PC con **RTX 5070 (12 GB), Ryzen 7 5800X y 32 GB de RAM**, con Windows 11. Se hace una sola vez. Guarda los modelos en el **NVMe** (por ejemplo `D:\IA`), no en el HDD.

> Precios: todo es gratis. Lo que se descarga son unos 15 GB.

## 0. Driver NVIDIA

ComfyUI portable trae PyTorch **CUDA 13.0**, que necesita un driver de la rama **R580 o más nueva**. La RTX 5070 (Blackwell, sm_120) no funciona con builds viejas de CUDA 12.4/12.6.

```powershell
nvidia-smi
```

Si la versión del driver empieza por menos de 580, actualízalo desde la app NVIDIA o nvidia.com.

## 1. Ollama: el que "ve" la foto y conversa

1. Instala Ollama desde <https://ollama.com/download> (Windows). Queda en la bandeja del sistema y escucha en `http://127.0.0.1:11434`.
2. *(Opcional)* Para guardar los modelos en el NVMe, crea la variable de usuario `OLLAMA_MODELS = D:\IA\ollama` y reinicia Ollama.
3. Descarga el modelo recomendado. También se puede desde Plasmo → Ajustes → Descargar.

```powershell
ollama pull qwen3.5:9b
```

| Modelo | Descarga | Para qué |
|---|---|---|
| `qwen3.5:9b` | 6.6 GB | **Recomendado.** Ve imágenes, buen español, responde JSON fiable y cabe en 12 GB con 8k de contexto |
| `qwen3.5:4b` | 3.3 GB | Más liviano y rápido, misma familia |
| `gemma4:12b` | 8.0 GB | Alternativa para comparar redacción |

No hace falta un modelo aparte para el chat: el mismo `qwen3.5:9b` sirve. Plasmo apaga el "modo pensar" (`think: false`) y pide `num_ctx: 8192` en cada llamada.

> Ollama **ya no genera imágenes**: la función experimental se quitó en la v0.32.6. Por eso el texto→imagen va por ComfyUI (paso 3).

## 2. ComfyUI + Hunyuan3D 2.1: la figura 3D

1. Descarga **ComfyUI portable** (v0.42.1 o más nueva): [`ComfyUI_windows_portable_nvidia.7z`](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (unos 2 GB). Extráelo con 7-Zip en `D:\IA\ComfyUI_windows_portable`.
   - No uses la build `_cu126`: es para tarjetas serie 10 o más viejas.
2. Descarga el checkpoint y ponlo en `ComfyUI\models\checkpoints\`:

   | Archivo | Tamaño | VRAM (solo forma) | Enlace |
   |---|---|---|---|
   | `hunyuan_3d_v2.1.safetensors` | 7.4 GB | ~10 GB | [Hugging Face](https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors) — **recomendado** |
   | `hunyuan3d-dit-v2_fp16.safetensors` | 4.9 GB | ~6 GB | [Hugging Face](https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors) — respaldo si te falta VRAM |

   Va en `checkpoints`, aunque el README del repositorio diga `diffusion_models`: el nodo `ImageOnlyCheckpointLoader` solo busca en `checkpoints`.
3. Arranca ComfyUI con `run_nvidia_gpu.bat`. Debe responder en `http://127.0.0.1:8188`.
4. *(Opcional, para comprobar)* En ComfyUI abre **Plantillas → 3D → Hunyuan3D 2.1** y genera algo de prueba.

Plasmo arma solo este grafo y lo envía por la API:

```
ImageOnlyCheckpointLoader → CLIPVisionEncode (center) → Hunyuan3Dv2Conditioning
ModelSamplingAuraFlow (shift 1.0) + EmptyLatentHunyuan3Dv2 (4096) → KSampler (30 pasos, cfg 5, euler/normal)
→ VAEDecodeHunyuan3D (octree 256) → VoxelToMesh (surface net, 0.6) → SaveGLB
```

Con el checkpoint 2.0, Plasmo usa los valores de su plantilla: 3072, 20 pasos y cfg 8.

## 3. *(Opcional)* Texto → imagen → 3D

Para escribir una idea en vez de subir una foto, pon un checkpoint de imagen de un solo archivo en `ComfyUI\models\checkpoints\` y elígelo en **Ajustes → Checkpoint texto→imagen**:

- `sd_xl_base_1.0.safetensors` (6.9 GB, licencia OpenRAIL++, permite uso comercial): [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors).

Ollama escribe el prompt en inglés, pensado para 3D: un solo objeto, fondo blanco, vista 3/4. ComfyUI dibuja la imagen y Plasmo la recorta y la deja lista.

## 4. Plasmo

```powershell
git clone https://github.com/RandyGrullon/Plasmo.git
cd Plasmo
powershell -ExecutionPolicy Bypass -File scripts\instalar.ps1
```

Abre Plasmo. Las dos píldoras de arriba deben estar en verde: **Ollama · qwen3.5:9b** y **ComfyUI · RTX 5070**.

## Cómo se reparte la VRAM (12 GB)

| Momento | Quién la usa |
|---|---|
| Analizar la foto o chatear | Ollama, ~8 GB |
| Generar la figura | Plasmo descarga Ollama (`keep_alive: 0`) y entra Hunyuan3D 2.1, ~10 GB |
| Al terminar | Plasmo llama a `POST /free` en ComfyUI para devolver la VRAM |

El recorte del fondo y la profundidad corren en la CPU (ONNX), así que no compiten por la GPU.

## Problemas comunes

| Síntoma | Solución |
|---|---|
| "ComfyUI no responde" | Abre `run_nvidia_gpu.bat`; revisa el puerto en Ajustes (ComfyUI Desktop usa el **8000**, no el 8188) |
| "Falta hunyuan_3d_v2.1.safetensors" | El archivo no está en `models\checkpoints` o tiene otro nombre: elígelo en Ajustes |
| "no tiene los nodos de Hunyuan3D" | Actualiza ComfyUI (`update\update_comfyui.bat`) |
| Se queda sin memoria | Usa el checkpoint 2.0 o baja "Detalle de la malla" a 192 |
| "Superficie suave" tarda minutos | `VoxelToMesh` corre en CPU; con octree 384–512 elige "Rápida (escalonada)" |
| Ollama: `model not found` | `ollama pull qwen3.5:9b` o Ajustes → Descargar |
