"""IDOL: flujo de ComfyUI, instalador (con el ComfyUI simulado), giro y colores propios."""

import io
import re

import numpy as np
import pytest
import trimesh

from plasmo import idol
from plasmo.config import Settings
from plasmo.engines import comfy_root
from plasmo.generators import idol_workflow
from plasmo.imaging import processing as proc
from plasmo.mesh import color as paint
from plasmo.mesh import printability as pr
from tests.sample import sample_image
from tests.test_api import H, client, new_project, wait_job  # noqa: F401  (fixture)
from tools import mock_comfyui


@pytest.fixture()
def comfy_dirs(tmp_path, monkeypatch, comfy_url):
    """ComfyUI simulado con sus carpetas vacías y descargas de mentira (nada de los 4,7 GB reales)."""
    monkeypatch.setattr(mock_comfyui, "ROOT", tmp_path / "comfy")
    monkeypatch.setattr(idol, "MODELS", mock_comfyui.fake_models(comfy_url))
    monkeypatch.setattr(idol, "PACK_URL", f"{comfy_url}/descargas/ComfyUI-Idol.zip")
    return tmp_path / "comfy"


def test_workflow_follows_idol_settings():
    wf = idol_workflow("recorte.png", {"calidad": "maxima", "semilla": 7, "modo_foto": "synthetic"})
    types = {v["class_type"] for v in wf.values()}
    assert {"IdolPreprocessImage", "IdolPrecision", "TripoSplatConditioning", "VAEDecodeTripoSplat", "SplatToMesh", "SplatToFile3D"} <= types
    assert wf["2"] == {"class_type": "InvertMask", "inputs": {"mask": ["1", 1]}}  # la máscara de LoadImage viene invertida
    k = wf["9"]["inputs"]
    assert (k["sampler_name"], k["scheduler"], k["steps"], k["cfg"], k["seed"]) == ("euler", "simple", 50, 4.0, 7)
    assert wf["3"]["inputs"]["mode"] == "synthetic" and wf["8"]["inputs"]["precision"] == "fp32, fp16 attention"
    assert wf["12"]["inputs"]["resolution"] == 512 and wf["11"]["inputs"]["num_gaussians"] == 262_144
    fast = idol_workflow("r.png", {"calidad": "rapida", "resolucion_malla": 1000, "gaussianas": 5, "precision_idol": "raro"})
    assert fast["9"]["inputs"]["steps"] == 30 and fast["8"]["inputs"]["precision"] == "default"
    assert fast["12"]["inputs"]["resolution"] == 768 and fast["11"]["inputs"]["num_gaussians"] == 32_768
    assert fast["3"]["inputs"]["mode"] == "real"


def test_version_parsing():
    assert idol.parse_version("0.43.0") >= idol.MIN_VERSION
    assert idol.parse_version("v0.42.1-mock") < idol.MIN_VERSION
    assert idol.parse_version(None) < idol.MIN_VERSION


def test_comfy_root_of_portable(tmp_path):
    portable = tmp_path / "ComfyUI_windows_portable"
    (portable / "python_embeded").mkdir(parents=True)
    (portable / "python_embeded" / "python.exe").write_text("")
    (portable / "ComfyUI").mkdir()
    (portable / "ComfyUI" / "main.py").write_text("")
    assert comfy_root(Settings(comfy_dir=str(portable))) == portable / "ComfyUI"


def test_install_downloads_resumes_and_is_ready(client, comfy_dirs):  # noqa: F811
    st = client.get("/api/estado").json()["generador"]["idol"]
    assert st["responde"] and not st["listo"] and len(st["faltan"]) == 4 and not st["nodos"]

    # una descarga cortada a la mitad se retoma desde donde quedó
    first = idol.MODELS[0]
    part = comfy_dirs / "models" / first.folder / (first.name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes((first.name.encode() * 99999)[: first.size // 2])

    out = wait_job(client, client.post("/api/idol/instalar", headers=H).json())
    assert out["idol"]["listo"], out
    assert (comfy_dirs / "custom_nodes" / "ComfyUI-Idol" / "__init__.py").is_file()
    assert (comfy_dirs / "custom_nodes" / "ComfyUI-Idol" / ".plasmo-commit").read_text() == idol.PACK_COMMIT
    for m in idol.MODELS:
        f = comfy_dirs / "models" / m.folder / m.name
        assert f.stat().st_size == m.size
    assert f.read_bytes()[:10] == m.name.encode()[:10]
    assert not part.exists()
    assert client.get("/api/estado").json()["generador"]["idol"]["listo"]

    again = wait_job(client, client.post("/api/idol/instalar", headers=H).json())  # nada que bajar
    assert again["idol"]["listo"] and not any("Descargué" in n for n in again["notas"])


def test_figure_needs_idol_installed(client, comfy_dirs):  # noqa: F811
    pid = new_project(client)
    r = client.post(f"/api/proyectos/{pid}/generar", json={"forma": "figura", "parametros": {"modelo_3d": "idol"}}, headers=H)
    assert r.status_code == 409 and "IDOL" in r.json()["detail"]


def test_idol_figure_turns_to_face_the_photo_and_keeps_its_colors(client, comfy_dirs):  # noqa: F811
    wait_job(client, client.post("/api/idol/instalar", headers=H).json())
    pid = new_project(client)
    job = client.post(
        f"/api/proyectos/{pid}/generar",
        json={"forma": "figura", "parametros": {"modelo_3d": "idol", "pasos": 4, "semilla": 99},
              "impresion": {"tamano_mm": 70, "voxeles": 256, "pulido": "suave"}},
        headers=H,
    ).json()
    out = wait_job(client, job)
    notes = " ".join(out["notas"])
    yaw = float(re.search(r"la giré (\d+)°", notes).group(1))
    assert abs(yaw - (360 - mock_comfyui.IDOL_YAW)) <= 6  # deshace el giro con el que la dejó «IDOL»
    assert "colores de IDOL" in notes
    assert out["reporte"]["metricas"]["cerrada"]
    assert "splat" in out["archivos"] and "obj_color" in out["archivos"]
    assert out["parametros"]["modo_foto"] == "real"  # foto subida: encuadre de foto real
    ply = client.get(f"/api/proyectos/{pid}/archivo/{out['version']}/{out['archivos']['splat']}").content
    assert ply.startswith(b"ply")


def test_orient_finds_front_even_when_mirrored():
    img = sample_image().convert("RGBA")
    cut = proc.cutout(img, proc.segment(img, "simple"))
    buf = io.BytesIO()
    cut.save(buf, "PNG")
    for turn in (40.0, 200.0):
        glb = mock_comfyui.inflate(buf.getvalue(), colored=True, yaw=turn)
        mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
        colors = paint.linear_to_srgb(np.asarray(mesh.visual.vertex_colors))
        pr.y_up_to_z_up(mesh)
        yaw, iou = paint.orient_to_photo(mesh, colors, cut)
        assert abs(((yaw + turn) % 360 + 180) % 360 - 180) <= 6, (turn, yaw)
        assert iou > 0.8


def test_linear_to_srgb_roundtrip():
    srgb = np.array([[0, 64, 128], [200, 255, 10]], dtype=np.float32) / 255
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    back = paint.linear_to_srgb(np.round(lin * 255 * 256) / 256)
    assert np.abs(back.astype(int) - (srgb * 255).round().astype(int)).max() <= 2


def test_transfer_colors_follows_the_scaled_figure():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation([0, 0, 1])
    colors = np.where(sphere.vertices[:, [0]] > 0, [[230, 40, 40]], [[40, 40, 230]]).astype(np.uint8)
    final = sphere.copy()
    final.apply_scale(60.0)
    final.apply_translation([5, -3, 0])
    out = paint.transfer_colors(sphere.vertices, colors, final)
    right = final.vertices[:, 0] > 5 + 6
    left = final.vertices[:, 0] < 5 - 6
    assert (out[right, 0] > 200).all() and (out[left, 2] > 200).all()
    with_base = paint.transfer_colors(sphere.vertices, colors, final, base_top=final.bounds[0][2] + 5)
    assert (with_base[final.vertices[:, 2] < final.bounds[0][2] + 5] == paint.BASE_COLOR).all()
