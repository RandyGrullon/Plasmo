import io
import zipfile

import numpy as np
import pytest
import trimesh

from plasmo import pipeline
from plasmo.imaging import processing as proc
from plasmo.mesh import export, forms, printability as pr
from plasmo.mesh.heightfield import bend_cylinder, clean_mask, heightfield_solid
from tests.sample import sample_image


@pytest.fixture(scope="module")
def src():
    img = sample_image().convert("RGBA")
    return forms.Source(img, proc.segment(img, "simple"), depth_model="luminancia")


def test_heightfield_watertight_with_holes_and_pinches():
    yy, xx = np.mgrid[0:80, 0:100]
    mask = (xx - 50) ** 2 + (yy - 40) ** 2 < 35**2
    mask[35:45, 45:55] = False
    mask[5, 5] = mask[6, 6] = True  # contacto en diagonal
    mesh = heightfield_solid(2 + np.sin(xx / 5), mask, 0.25)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0


def test_clean_mask_removes_diagonals():
    m = np.zeros((4, 4), bool)
    m[1, 1] = m[2, 2] = True
    out = clean_mask(m)
    assert out[1, 2] and out[2, 1]


def test_cylinder_closes_seam():
    flat = heightfield_solid(np.full((30, 60), 2.0), None, 0.5, wrap_x=True)
    cyl = bend_cylinder(flat, 30.0)
    assert cyl.is_watertight and cyl.is_volume


@pytest.mark.parametrize(
    "name,params",
    [
        ("relieve", {}), ("relieve", {"contorno": "corazon"}), ("relieve", {"contorno": "escudo"}),
        ("litofania", {}), ("litofania", {"forma": "curva"}), ("litofania", {"forma": "cilindro", "diametro_mm": 50}),
        ("llavero", {}), ("llavero", {"contorno": "estrella"}), ("iman", {}), ("iman", {"contorno": "hexagono"}),
        ("pixel", {"colores": 3}), ("cortador", {}),
    ],
)
def test_forms_are_printable(src, name, params):
    result = forms.FORMS[name](src, params)
    for part in result.parts:
        assert part.mesh.is_watertight, part.name
        assert part.mesh.is_volume, part.name
    mesh = export.combined(result.parts)
    assert mesh.bounds[0][2] >= -1e-6  # apoyada en la cama
    report = pr.analyze(mesh, thin=pipeline.thin_policy(name))
    assert report["metricas"]["cerrada"]
    assert report["estado"] in ("sana", "aviso")


def test_pixel_parts_are_not_overhangs(src):
    result = forms.pixel_art(src, {"colores": 4})
    assert len(result.parts) >= 3  # placa + colores
    assert pr.analyze(export.combined(result.parts))["voladizos"] < 0.03


def test_keychain_has_hole(src):
    mesh = forms.llavero(src, {}).parts[0].mesh
    assert mesh.euler_number <= 0  # al menos un agujero pasante (género ≥ 1)


def test_3mf_is_valid_zip_with_objects(src):
    parts = forms.pixel_art(src, {"colores": 3}).parts
    data = export.to_3mf(parts, "prueba")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= set(z.namelist())
        model = z.read("3D/3dmodel.model").decode()
    assert model.count("<object ") == len(parts) + 1
    assert "<components>" in model and 'unit="millimeter"' in model
    loaded = trimesh.load(io.BytesIO(data), file_type="3mf")
    assert len(loaded.geometry) == len(parts)


def test_repair_closes_open_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=20)
    mesh.update_faces(np.arange(len(mesh.faces)) > 5)  # le abro un hueco
    mesh.remove_unreferenced_vertices()
    floater = trimesh.creation.box(extents=[1, 1, 1])
    floater.apply_translation([60, 0, 0])
    broken = trimesh.util.concatenate([mesh, floater])
    fixed, log = pr.repair(broken)
    assert fixed.is_watertight and fixed.is_volume
    assert fixed.body_count == 1
    assert log


def test_prepare_figure_base_flatten_and_split():
    mesh = trimesh.creation.capsule(height=200, radius=30)
    pr.place_on_bed(mesh)
    flat = pr.flatten_bottom(mesh, 2.0)
    assert flat.is_volume and pr.stability(flat)["apoyo_mm2"] > 100
    based = pr.add_base(flat.copy(), "disco", 3.0)
    assert based.is_volume and based.extents[0] > flat.extents[0]
    pieces = pr.split_for_bed(trimesh.creation.box(extents=[300, 40, 40]), (256, 256, 256))
    assert len(pieces) == 2 and all(p.is_volume and p.extents[0] < 256 for p in pieces)


def test_simplify_keeps_closed(src):
    big = forms.litofania(src, {"ancho_mm": 120}).parts[0].mesh
    small = pr.simplify(big, 150_000)
    assert len(small.faces) < len(big.faces)
    assert small.is_watertight and small.is_volume
