import io
import time
import zipfile

import pytest
import trimesh
from fastapi.testclient import TestClient

from plasmo.app import create_app
from plasmo.config import ConfigStore
from plasmo.generators import ComfyUIGenerator, hunyuan_workflow
from plasmo.jobs import JobManager, Timings
from plasmo.ollama_client import OllamaClient
from tests.sample import sample_image

H = {"X-Plasmo": "1"}


@pytest.fixture()
def client(tmp_path, ollama_url, comfy_url):
    cfg = ConfigStore(tmp_path / "config.json")
    cfg.update({
        "ollama_url": ollama_url, "comfy_url": comfy_url, "output_dir": str(tmp_path / "proyectos"),
        "bg_model": "simple", "depth_model": "luminancia", "comfy_t2i_checkpoint": "sd_xl_base_1.0.safetensors",
    })
    app = create_app(cfg, JobManager(timings=Timings(tmp_path / "tiempos.json")))
    with TestClient(app) as c:
        yield c


def png() -> bytes:
    buf = io.BytesIO()
    sample_image().save(buf, "PNG")
    return buf.getvalue()


def wait_job(client, job, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/trabajos/{job['id']}").json()
        if j["estado"] in ("listo", "error", "cancelado"):
            assert j["estado"] == "listo", j.get("error")
            assert j["porcentaje"] == 100
            return j["resultado"]
        time.sleep(0.1)
    raise AssertionError("el trabajo no terminó")


def new_project(client) -> str:
    r = client.post("/api/proyectos", files={"imagen": ("muñeco.png", png(), "image/png")}, headers=H)
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["recorte"]["cobertura"] > 0.1
    return meta["id"]


def test_guard_blocks_foreign_requests(client):
    assert client.post("/api/proyectos/x/analizar").status_code == 403  # sin cabecera
    assert client.get("/api/ping", headers={"host": "evil.example"}).status_code == 403
    assert client.get("/api/ping").json()["app"] == "plasmo"


def test_status_sees_mocks(client):
    st = client.get("/api/estado").json()
    assert st["ollama"]["ok"] and "qwen3.5:9b" in st["ollama"]["vision"]
    assert st["ollama"]["modelo_listo"]
    assert st["generador"]["ok"], st["generador"]
    assert "RTX 5070" in st["generador"]["gpu"]


def test_full_flow_relief_and_analysis(client):
    pid = new_project(client)
    res = wait_job(client, client.post(f"/api/proyectos/{pid}/analizar", headers=H).json())
    assert res["analisis"]["forma_recomendada"] == "figura"
    assert res["analisis"]["nombre_archivo"] == "muneco-gato"

    job = client.post(f"/api/proyectos/{pid}/generar", json={"forma": "relieve", "parametros": {"contorno": "circulo", "tamano_mm": 60}}, headers=H).json()
    out = wait_job(client, job)
    assert out["reporte"]["metricas"]["cerrada"]
    assert set(out["archivos"]) >= {"stl", "3mf", "obj", "glb"}
    stl = client.get(f"/api/proyectos/{pid}/archivo/{out['version']}/{out['archivos']['stl']}")
    mesh = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert mesh.is_watertight and max(mesh.extents[:2]) == pytest.approx(60, abs=1.5)
    meta = client.get(f"/api/proyectos/{pid}").json()
    assert meta["versiones"][-1]["version"] == "v1-relieve"
    assert client.get("/api/proyectos").json()["proyectos"][0]["id"] == pid


def test_figure_with_comfyui(client):
    pid = new_project(client)
    job = client.post(
        f"/api/proyectos/{pid}/generar",
        json={"forma": "figura", "parametros": {"pasos": 8}, "impresion": {"tamano_mm": 90, "base": "disco", "aplanar_mm": 1}},
        headers=H,
    ).json()
    out = wait_job(client, job)
    assert out["reporte"]["metricas"]["cerrada"]
    assert out["reporte"]["dimensiones_mm"][2] == pytest.approx(90 - 1 + 3, abs=2.5)  # aplanado + base
    assert out["reporte"]["estabilidad"]["estable"]
    assert "obj_color" in out["archivos"]  # la figura sale con los colores de la foto
    obj = client.get(f"/api/proyectos/{pid}/archivo/{out['version']}/{out['archivos']['obj_color']}").text
    assert next(line for line in obj.splitlines() if line.startswith("v ")).count(" ") == 6


def test_pixel_3mf_has_parts(client):
    pid = new_project(client)
    out = wait_job(client, client.post(f"/api/proyectos/{pid}/generar", json={"forma": "pixel", "parametros": {"colores": 3}}, headers=H).json())
    data = client.get(f"/api/proyectos/{pid}/archivo/{out['version']}/{out['archivos']['3mf']}").content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.read("3D/3dmodel.model").count(b"<object ") == len(out["piezas"]) + 1
    assert "zip" in out["archivos"]


def test_config_with_bom_is_read(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"vision_model": "gemma4:12b", "tutorial_visto": true}')  # como lo guarda el Bloc de notas
    s = ConfigStore(path).settings
    assert s.vision_model == "gemma4:12b" and s.tutorial_visto is True


def test_example_project_and_tutorial_flag(client):
    assert client.get("/api/config").json()["tutorial_visto"] is False
    meta = client.post("/api/ejemplo", headers=H).json()
    assert meta["ejemplo"] is True and meta["recorte"]["cobertura"] > 0.1
    assert client.get(f"/api/proyectos/{meta['id']}/archivo/recorte.png").status_code == 200
    assert client.put("/api/config", json={"tutorial_visto": True}, headers=H).json()["tutorial_visto"] is True


def test_chat_streams(client):
    r = client.post("/api/chat", json={"mensajes": [{"role": "user", "content": "¿Temperatura para PLA?"}]}, headers=H)
    assert "PLA" in r.text


def test_text_to_image_creates_project(client):
    res = wait_job(client, client.post("/api/texto-a-imagen", json={"idea": "un gato naranja de juguete"}, headers=H).json())
    meta = client.get(f"/api/proyectos/{res['proyecto']}").json()
    assert meta["prompt"].startswith("a cute orange cat")


def test_ollama_pull_progress(client):
    res = wait_job(client, client.post("/api/ollama/descargar", json={"modelo": "qwen3.5:4b"}, headers=H).json())
    assert res["modelo"] == "qwen3.5:4b"


def test_bad_file_and_path(client):
    assert client.post("/api/proyectos", files={"imagen": ("x.png", b"no soy imagen", "image/png")}, headers=H).status_code == 400
    pid = new_project(client)
    assert client.get(f"/api/proyectos/{pid}/archivo/../config.json").status_code == 404
    assert client.get(f"/api/proyectos/{pid}/archivo/..%2F..%2Fsecret").status_code == 404


def test_workflow_matches_official_template():
    fast = hunyuan_workflow("foto.png", "hunyuan_3d_v2.1.safetensors", {"calidad": "rapida"})
    assert fast["6"]["inputs"]["resolution"] == 4096  # igual que la plantilla oficial
    assert fast["7"]["inputs"]["steps"] == 30 and fast["7"]["inputs"]["cfg"] == 5.0
    assert fast["8"]["inputs"]["octree_resolution"] == 256
    best = hunyuan_workflow("foto.png", "hunyuan_3d_v2.1.safetensors", {})  # por defecto: calidad máxima
    assert best["7"]["inputs"]["steps"] == 50 and best["8"]["inputs"]["octree_resolution"] == 512
    wf20 = hunyuan_workflow("foto.png", "hunyuan3d-dit-v2_fp16.safetensors", {"octree": 384})
    assert wf20["6"]["inputs"]["resolution"] == 3072 and wf20["7"]["inputs"]["cfg"] == 8.0
    assert wf20["8"]["inputs"]["octree_resolution"] == 384
    assert best["10"]["class_type"] == "SaveGLB" and best["1"]["class_type"] == "ImageOnlyCheckpointLoader"


def test_comfy_rejects_missing_checkpoint(comfy_url):
    st = ComfyUIGenerator(comfy_url, "no-existe.safetensors").status()
    assert not st["ok"] and "no-existe" in st["mensaje"]


def test_ollama_client_disables_thinking(ollama_url):
    from tools import mock_ollama

    c = OllamaClient(ollama_url)
    c.analyze_image("qwen3.5:9b", sample_image())
    sent = mock_ollama.CALLS[-1]
    assert sent["think"] is False and sent["options"]["num_ctx"] == 8192
    assert not sent["messages"][1]["images"][0].startswith("data:")
