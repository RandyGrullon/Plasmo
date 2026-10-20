# -*- mode: python ; coding: utf-8 -*-
# Ejecutable de Windows: uv run pyinstaller --noconfirm --clean plasmo.spec
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = [("plasmo/web", "plasmo/web")]
binaries = []
hiddenimports = collect_submodules("uvicorn") + collect_submodules("plasmo")
for pkg in ("manifold3d", "rtree", "fast_simplification", "webview", "onnxruntime", "clr_loader", "pythonnet", "skimage"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files("trimesh")

a = Analysis(
    ["scripts/plasmo_exe.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Plasmo",
    console=False,
    icon="assets/plasmo.ico",
)
coll = COLLECT(exe, a.binaries, a.datas, name="Plasmo")
