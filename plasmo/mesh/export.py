"""Exportación: STL, OBJ, GLB (vista previa con colores) y 3MF multiobjeto."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from plasmo.mesh.forms import Part

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _mesh_xml(mesh: trimesh.Trimesh) -> str:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    verts = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
    return f"<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>"


def to_3mf(parts: list[Part], title: str = "Plasmo") -> bytes:
    """Un objeto con una pieza por color: el slicer las mantiene alineadas
    y deja asignar un filamento distinto a cada una."""
    materials = "".join(
        f'<base name="{_xml_escape(p.name)}" displaycolor="#{p.color[0]:02X}{p.color[1]:02X}{p.color[2]:02X}FF"/>' for p in parts
    )
    objects = []
    for i, p in enumerate(parts):
        objects.append(f'<object id="{i + 2}" type="model" name="{_xml_escape(p.name)}" pid="1" pindex="{i}">{_mesh_xml(p.mesh)}</object>')
    assembly_id = len(parts) + 2
    if len(parts) > 1:
        comps = "".join(f'<component objectid="{i + 2}"/>' for i in range(len(parts)))
        objects.append(f'<object id="{assembly_id}" type="model" name="{_xml_escape(title)}"><components>{comps}</components></object>')
        build = f'<item objectid="{assembly_id}"/>'
    else:
        build = '<item objectid="2"/>'
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="es-DO" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<metadata name="Title">{_xml_escape(title)}</metadata><metadata name="Application">Plasmo</metadata>'
        f'<resources><basematerials id="1">{materials}</basematerials>{"".join(objects)}</resources>'
        f"<build>{build}</build></model>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def combined(parts: list[Part]) -> trimesh.Trimesh:
    return trimesh.util.concatenate([p.mesh for p in parts]) if len(parts) > 1 else parts[0].mesh


def _painted(p: Part) -> trimesh.Trimesh:
    """Copia de la pieza con sus colores: los de la foto si los tiene, si no su color plano."""
    mesh = p.mesh.copy()
    if p.vertex_colors is not None and len(p.vertex_colors) == len(mesh.vertices):
        rgba = np.column_stack([p.vertex_colors, np.full(len(mesh.vertices), 255, np.uint8)])
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    else:
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.array([*p.color, 255], dtype=np.uint8))
    return mesh


def colored_obj(parts: list[Part]) -> str:
    """OBJ con color por vértice: Bambu Studio y OrcaSlicer lo convierten a filamentos."""
    meshes = []
    for p in parts:
        mesh = _painted(p)
        if p.vertex_colors is None:
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile([*p.color, 255], (len(mesh.vertices), 1)).astype(np.uint8))
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    return trimesh.exchange.obj.export_obj(mesh, include_color=True, include_normals=False, include_texture=False)


def preview_glb(parts: list[Part]) -> bytes:
    scene = trimesh.Scene()
    for p in parts:
        scene.add_geometry(_painted(p), node_name=p.name, geom_name=p.name)
    # el visor usa Y arriba (glTF); las piezas están en Z arriba
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return scene.export(file_type="glb")


def write_all(parts: list[Part], folder: Path, stem: str, title: str) -> dict[str, str]:
    """Escribe STL, 3MF, OBJ y la vista previa. Devuelve {formato: nombre de archivo}."""
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    mesh = combined(parts)
    (folder / f"{stem}.stl").write_bytes(mesh.export(file_type="stl"))
    files["stl"] = f"{stem}.stl"
    (folder / f"{stem}.3mf").write_bytes(to_3mf(parts, title))
    files["3mf"] = f"{stem}.3mf"
    (folder / f"{stem}.obj").write_text(mesh.export(file_type="obj"), encoding="utf-8")
    files["obj"] = f"{stem}.obj"
    if any(p.vertex_colors is not None for p in parts):
        (folder / f"{stem}-color.obj").write_text(colored_obj(parts), encoding="utf-8")
        files["obj_color"] = f"{stem}-color.obj"
    if len(parts) > 1:
        with zipfile.ZipFile(folder / f"{stem}-piezas.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in parts:
                z.writestr(f"{stem}-{p.name}.stl", p.mesh.export(file_type="stl"))
        files["zip"] = f"{stem}-piezas.zip"
    (folder / "preview.glb").write_bytes(preview_glb(parts))
    files["glb"] = "preview.glb"
    return files
