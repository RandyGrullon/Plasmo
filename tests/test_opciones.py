"""Opciones de calidad: catálogo, perfiles y lo que cada opción cambia de verdad."""

import numpy as np
import pytest
import trimesh

from plasmo import catalog
from plasmo.generators import hunyuan_workflow
from plasmo.imaging import processing as proc
from plasmo.mesh import forms, printability as pr
from tests.sample import sample_image
from tests.test_api import H, client, new_project, wait_job  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def src():
    img = sample_image().convert("RGBA")
    return forms.Source(img, proc.segment(img, "simple"), depth_model="luminancia")


# ------------------------------------------------------------------ catálogo

@pytest.mark.parametrize("form", catalog.FORMS, ids=lambda f: f["clave"])
def test_every_option_has_a_group_and_maxima_is_the_default(form):
    fields = {f["clave"]: f for f in [*form["parametros"], *form["impresion"]]}
    assert all(f["grupo"] in catalog.GRUPOS for f in fields.values())
    assert [g["clave"] for g in form["grupos"]] == list(dict.fromkeys(f["grupo"] for f in fields.values()))
    profiles = {p["clave"]: p for p in form["perfiles"]}
    assert "maxima" in profiles and "rapida" in profiles
    for p in profiles.values():
        for section in ("parametros", "impresion"):
            keys = {f["clave"] for f in form[section]}
            assert set(p[section]) <= keys, (p["clave"], set(p[section]) - keys)
    best = profiles["maxima"]
    for section in ("parametros", "impresion"):
        for key, value in best[section].items():
            assert fields[key]["defecto"] == value, key
    for f in fields.values():
        if f["tipo"] == "opcion":
            assert any(str(v) == str(f["defecto"]) for v, _ in f["opciones"]), f["clave"]
        if f["tipo"] == "numero":
            assert f["min"] <= f["defecto"] <= f["max"], f["clave"]


def test_workflow_takes_sampling_options():
    wf = hunyuan_workflow("foto.png", "hunyuan_3d_v2.1.safetensors", {
        "muestreador": "heun", "programador": "beta", "umbral": 0.45, "latentes": 6144, "shift": 2.0, "semilla": 7,
    })
    k = wf["7"]["inputs"]
    assert (k["sampler_name"], k["scheduler"], k["seed"]) == ("heun", "beta", 7)
    assert wf["9"]["inputs"]["threshold"] == 0.45
    assert wf["6"]["inputs"]["resolution"] == 6144
    assert wf["5"]["inputs"]["shift"] == 2.0
    # valores raros vuelven a los de la plantilla oficial o al rango de los nodos
    odd = hunyuan_workflow("foto.png", "hunyuan_3d_v2.1.safetensors", {"muestreador": "x; rm", "programador": "?", "octree": 4096})
    assert (odd["7"]["inputs"]["sampler_name"], odd["7"]["inputs"]["scheduler"]) == ("euler", "normal")
    assert odd["8"]["inputs"]["octree_resolution"] == 512 and odd["6"]["inputs"]["resolution"] == 4096


# ---------------------------------------------------------------------- malla

def test_thicken_grows_thin_parts_evenly():
    thin = trimesh.creation.capsule(height=1.0, radius=0.05)
    plain = pr.voxel_remesh(thin, 128)
    thick = pr.voxel_remesh(thin, 128, grow=0.03)
    assert thick.is_watertight
    grown = thick.extents - plain.extents
    assert np.allclose(grown, 0.06, atol=0.02)
    assert np.allclose(thick.bounds.mean(axis=0), plain.bounds.mean(axis=0), atol=0.01)  # crece parejo, no se corre


def test_repair_can_force_voxels_and_keep_fragments():
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    speck = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    speck.apply_translation([3, 0, 0])
    both = trimesh.util.concatenate([body, speck])
    kept, _ = pr.repair(both, min_part_ratio=0.0)
    dropped, _ = pr.repair(both, min_part_ratio=0.02)
    assert len(kept.split()) == 2 and len(dropped.split()) == 1
    forced, log = pr.repair(body, voxel_res=96, force_voxel=True)
    assert forced.is_watertight and any("como pediste" in line for line in log)


def test_polish_levels():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    noisy = sphere.copy()
    noisy.vertices *= 1 + np.random.default_rng(1).normal(0, 0.01, (len(noisy.vertices), 1))
    rough = lambda m: float(np.abs(np.linalg.norm(m.vertices, axis=1) - 1).mean())  # noqa: E731
    medio, fuerte = pr.polish(noisy.copy(), "medio"), pr.polish(noisy.copy(), "fuerte")
    assert rough(fuerte) < rough(medio) < rough(noisy)


# --------------------------------------------------------------------- formas

def test_relief_detail_options_change_the_surface(src):
    base = {"contorno": "circulo", "tamano_mm": 50, "resolucion_mm": 0.3}
    plain = forms.relieve(src, base).parts[0].mesh
    fine = forms.relieve(src, {**base, "contraste_local": 0.8, "tamano_detalle": "fino", "detalle": 0.9}).parts[0].mesh
    assert plain.is_watertight and fine.is_watertight
    assert np.abs(plain.vertices[:, 2] - fine.vertices[:, 2]).max() > 0.2  # misma grilla, otra superficie


def test_detail_cap_raises_grid_resolution():
    w1, h1, r1 = forms.fit_grid(1.0, 200, 0.1)
    w2, h2, r2 = forms.fit_grid(1.0, 200, 0.1, forms.max_pixels({"detalle_max": 3.0}))
    assert w2 * h2 > 2 * w1 * h1 and r2 < r1


def test_lithophane_tone_options(src):
    base = {"ancho_mm": 60, "resolucion_mm": 0.3, "contraste_local": 0, "nitidez": 0}
    flat = forms.litofania(src, base).parts[0].mesh
    crisp = forms.litofania(src, {**base, "contraste_local": 0.6, "nitidez": 1.0, "gamma": 1.4}).parts[0].mesh
    assert flat.is_watertight and crisp.is_watertight
    assert not np.isclose(flat.volume, crisp.volume, rtol=1e-3)


def test_pixel_dominant_lab_and_cleanup(src):
    for opts in ({"muestreo": "dominante", "espacio_color": "lab"}, {"muestreo": "promedio", "espacio_color": "rgb", "limpiar": False}):
        parts = forms.pixel_art(src, {"pixeles": 24, "colores": 4, **opts}).parts
        assert len(parts) >= 3 and all(p.mesh.is_watertight for p in parts)
    grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert (forms._clean_isolated(grid) == 0).all()
    pair = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    assert (forms._clean_isolated(pair) == pair).all()  # dos juntos no son «sueltos»


def test_cutter_drops_details_thinner_than_minimum(src):
    loose = forms.cortador(src, {"tamano_mm": 60, "detalle_min_mm": 0.4, "resolucion_mm": 0.3}).parts[0].mesh
    strict = forms.cortador(src, {"tamano_mm": 60, "detalle_min_mm": 4.0, "resolucion_mm": 0.3}).parts[0].mesh
    assert loose.is_watertight and strict.is_watertight
    assert not np.isclose(loose.area, strict.area, rtol=1e-3)


def test_depth_falls_back_to_small_then_light(monkeypatch):
    img = sample_image().convert("RGBA")
    source = forms.Source(img, proc.segment(img, "simple"), depth_model="depth-anything-v2-large")

    def broken(*args, **kwargs):
        raise RuntimeError("sin red")

    monkeypatch.setattr(proc, "depth", broken)
    rel = source.relief_map("profundidad")
    assert rel.shape == (source.cut.height, source.cut.width)
    assert source.depth_model == "luminancia" and len(source.notes) == 2


# ----------------------------------------------------------------------- IA

def test_figure_best_of_several_tries(client):  # noqa: F811
    pid = new_project(client)
    job = client.post(
        f"/api/proyectos/{pid}/generar",
        json={"forma": "figura", "parametros": {"pasos": 4, "intentos": 2, "semilla": 1234, "realzar": True, "encuadre": 80},
              "impresion": {"tamano_mm": 60, "voxeles": 256, "caras_max": 0, "engrosar_mm": 0.4, "pulido": "medio"}},
        headers=H,
    ).json()
    out = wait_job(client, job)
    notes = " ".join(out["notas"])
    assert "Probé 2 semillas" in notes and "Engrosé" in notes and "Pulido medio" in notes
    assert "Semilla 1234" in notes or "Semilla 1235" in notes
    assert out["reporte"]["metricas"]["cerrada"]
    files = client.get(f"/api/proyectos/{pid}").json()
    assert files["versiones"][-1]["parametros"]["intentos"] == 2
