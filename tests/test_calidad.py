"""Calidad de detección y pulido: recorte, profundidad, colores de la foto y bordes."""

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from plasmo.imaging import downloads
from plasmo.imaging import processing as proc
from plasmo.mesh import color as paint
from plasmo.mesh import export, forms
from tests.sample import sample_image


def two_objects() -> Image.Image:
    img = Image.new("RGB", (400, 300), (245, 243, 240))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 60, 200, 240], fill=(220, 110, 60))  # objeto grande
    d.ellipse([290, 180, 350, 240], fill=(40, 90, 200))  # objeto chico separado
    return img.convert("RGBA")


def test_keep_components_keeps_every_object_but_drops_specks():
    m = np.zeros((300, 400), bool)
    m[60:240, 40:200] = True
    m[180:240, 290:350] = True  # objeto chico (3 % de la foto)
    m[10:12, 10:12] = True  # mota
    labels_kept = proc.keep_components(m)
    assert labels_kept[200, 320] and labels_kept[100, 100] and not labels_kept[10, 10]


def test_simple_segmentation_detects_both_objects():
    mask = proc.segment(two_objects(), "simple") > 0.5
    assert mask[150, 120] and mask[210, 320]  # los dos objetos
    assert not mask[20, 380]  # el fondo no


def test_refine_keeps_real_holes_and_fills_pinholes():
    img = Image.new("RGB", (200, 200), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 180, 180], fill=(200, 60, 60))
    d.ellipse([70, 70, 130, 130], fill=(240, 240, 240))  # agujero real (dona)
    prob = np.zeros((200, 200), np.float32)
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(yy - 100, xx - 100)
    prob[(r < 80) & (r > 30)] = 1.0
    prob[50, 50] = 0.0  # agujerito falso de un píxel
    prob[5:7, 5:7] = 1.0  # mota en el fondo
    out = proc.refine_mask(prob, img.convert("RGBA")) > 0.5
    assert out[50, 50] and not out[100, 100] and not out[6, 6]


def test_calibrate_makes_decided_subject_opaque():
    # BiRefNet ve el cuerpo (1) e IS-Net no (0): 70/30 da 0.7 → tiene que quedar opaco, no al 70 %
    fused = 0.7 * np.array([1.0, 1.0, 0.0]) + 0.3 * np.array([1.0, 0.0, 1.0])
    out = proc.calibrate(fused)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(1.0) and out[2] == pytest.approx(0.0)


def test_keep_main_drops_side_objects_only():
    prob = np.zeros((100, 200), np.float32)
    prob[20:80, 20:90] = 1.0  # protagonista
    prob[30:60, 140:180] = 1.0  # objeto de la mesa
    salient = np.zeros_like(prob)
    salient[25:75, 25:85] = 1.0  # IS-Net marca al protagonista
    out = proc.keep_main(prob, salient) > 0.5
    assert out[50, 50] and not out[45, 160]


def test_mask_quality_flags_cut_off_subjects():
    m = np.zeros((200, 200), np.float32)
    m[50:200, 60:140] = 1  # toca el borde de abajo
    q = proc.mask_quality(m)
    assert any("borde" in a for a in q["avisos"]) and q["nota"] < 100
    clean = np.zeros((200, 200), np.float32)
    clean[40:160, 60:140] = 1
    assert proc.mask_quality(clean)["calidad"] == "excelente"


def test_decontaminate_removes_background_halo():
    img = Image.new("RGB", (60, 60), (0, 0, 255))  # fondo azul
    ImageDraw.Draw(img).rectangle([20, 20, 40, 40], fill=(255, 0, 0))
    mask = np.zeros((60, 60), np.float32)
    mask[20:41, 20:41] = 1.0
    mask[19, 20:41] = 0.5  # borde semitransparente que tenía color de fondo
    out = np.asarray(proc.decontaminate(img, mask))
    assert tuple(out[19, 30]) == (255, 0, 0)


def test_guided_filter_follows_image_edges():
    guide = np.zeros((80, 80, 3), np.float32)
    guide[:, 40:] = 1.0  # borde nítido en x=40
    p = np.zeros((80, 80), np.float32)
    p[:, 36:] = 1.0  # la máscara está corrida 4 px
    q = proc.guided_filter(guide, p, 6, 1e-4)
    assert q[40, 45] > 0.9 and q[40, 30] < 0.2
    edge = np.argmax(q[40] > 0.5)
    assert edge > 36  # el borde de la máscara se corrió hacia el borde real de la foto (x=40)


@pytest.mark.skipif(not downloads.is_downloaded("depth-anything-v2-small"), reason="modelo de profundidad no descargado")
def test_depth_keeps_aspect_and_range():
    img = sample_image(320).resize((480, 240))
    d = proc.depth(img)
    assert d.shape == (240, 480) and 0 <= d.min() and d.max() <= 1


def test_connect_islands_makes_one_piece():
    m = np.zeros((100, 160), bool)
    m[20:80, 10:60] = True
    m[40:60, 120:150] = True
    out = forms.connect_islands(m, 6)
    from scipy import ndimage

    assert ndimage.label(out)[1] == 1


def test_bevel_softens_edges_only():
    subj = np.zeros((60, 60), bool)
    subj[10:50, 10:50] = True
    rel = np.where(subj, 1.0, 0.0)
    out = forms.bevel(rel, subj, 5)
    assert out[30, 30] == pytest.approx(1.0)  # el centro no cambia
    assert out[30, 10] < 0.15 < out[30, 11] < out[30, 12] < 1.0  # el canto sube suave


def test_relief_keeps_two_objects_in_one_piece():
    img = two_objects()
    src = forms.Source(img, proc.segment(img, "simple"), depth_model="luminancia")
    mesh = forms.relieve(src, {"tamano_mm": 80}).parts[0].mesh
    assert mesh.is_watertight and mesh.body_count == 1


def test_project_colors_front_sharp_back_soft():
    cut = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(cut)
    d.ellipse([10, 10, 90, 90], fill=(240, 120, 60, 255))
    d.ellipse([30, 35, 45, 50], fill=(10, 10, 10, 255))  # un ojo
    alpha = np.asarray(cut.getchannel("A"), np.float32) / 255
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=40)
    sphere.apply_translation([0, 0, 50])  # Z arriba; el frente mira a -Y
    frame = paint.fit_frame(sphere.vertices, alpha)
    colors = paint.project_colors(sphere, cut, frame).astype(int)
    v = sphere.vertices
    # el ojo (píxel 37,42 del recorte) cae en x≈-12.5, z≈57.5 de la esfera
    eye = np.argmin(np.linalg.norm(v - [-12.5, -37.3, 57.5], axis=1))  # de frente
    back = np.argmin(np.linalg.norm(v - [-12.5, 37.3, 57.5], axis=1))  # el mismo punto por detrás
    assert colors[eye].sum() < 200  # el ojo se ve oscuro de frente
    assert colors[back].sum() > 300  # atrás no se pinta el ojo


def test_colored_obj_has_vertex_colors():
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    part = forms.Part("figura", mesh, vertex_colors=np.tile([200, 50, 20], (len(mesh.vertices), 1)).astype(np.uint8))
    text = export.colored_obj([part])
    line = next(line for line in text.splitlines() if line.startswith("v "))
    assert len(line.split()) == 7  # x y z r g b
