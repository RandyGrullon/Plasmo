import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plasmo import engines
from plasmo.app import create_app
from plasmo.config import ConfigStore, Settings
from plasmo.jobs import JobManager, Timings
from tests.conftest import free_port

TOOLS = Path(__file__).resolve().parents[1] / "tools"
H = {"X-Plasmo": "1"}


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def mock_cmd(script: str, port: int, *extra: str) -> str:
    return " ".join(f'"{p}"' for p in (sys.executable, TOOLS / script)) + f" --port {port} " + " ".join(extra)


def test_detects_comfy_portable_and_inner_folder(tmp_path):
    root = tmp_path / "ComfyUI_windows_portable"
    py = touch(root / "python_embeded" / "python.exe")
    touch(root / "ComfyUI" / "main.py")
    for folder in (root, root / "ComfyUI"):  # también si eligen la carpeta de adentro
        launch = engines.find_comfy(Settings(comfy_dir=str(folder), comfy_url="http://127.0.0.1:8190"))
        assert launch.kind == "portable" and launch.argv[0] == str(py)
        assert launch.cwd == str(root)
        assert launch.argv[-4:] == ["--listen", "127.0.0.1", "--port", "8190"]
        assert "--windows-standalone-build" in launch.argv


def test_detects_comfy_git_install(tmp_path):
    root = tmp_path / "ComfyUI"
    touch(root / "main.py")
    py = touch(root / "venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python"))
    launch = engines.find_comfy(Settings(comfy_dir=str(root)))
    assert launch.kind == "git" and launch.argv[:2] == [str(py), "main.py"]


def test_missing_comfy_folder_is_none(tmp_path):
    assert engines.find_comfy(Settings(comfy_dir=str(tmp_path / "nada"))) is None


def test_ollama_serve_sets_host_for_custom_port(tmp_path):
    exe = touch(tmp_path / "ollama.exe")
    launch = engines.find_ollama(Settings(ollama_exe=str(exe), ollama_url="http://127.0.0.1:11999"))
    assert launch.kind == "serve" and launch.argv == [str(exe), "serve"]
    assert launch.env["OLLAMA_HOST"] == "127.0.0.1:11999"
    tray = touch(tmp_path / "ollama app.exe")
    launch = engines.find_ollama(Settings(ollama_exe=str(tmp_path)))
    assert launch.kind == "bandeja" and launch.argv == [str(tray)] and not launch.console


def test_split_cmd_keeps_quoted_paths():
    parts = engines.split_cmd('"C:\\Program Files\\Py\\python.exe" "D:\\IA\\main.py" --port 8188')
    assert parts[0].endswith("python.exe") and " " in parts[0] and parts[-2:] == ["--port", "8188"]


def test_start_stop_real_processes(tmp_path):
    ollama_port, comfy_port = free_port(), free_port()
    s = Settings(
        ollama_url=f"http://127.0.0.1:{ollama_port}", comfy_url=f"http://127.0.0.1:{comfy_port}",
        ollama_comando=mock_cmd("mock_ollama.py", ollama_port), comfy_comando=mock_cmd("mock_comfyui.py", comfy_port, "--arranque", "1"),
    )
    mgr = engines.EngineManager(tmp_path / "logs")
    try:
        for name in engines.ENGINES:
            assert not mgr.running(name, s)
            res = mgr.start(name, s)
            assert res["estado"] == "encendido", res
            assert mgr.running(name, s) and mgr.ours(name)
            assert mgr.start(name, s)["estado"] == "ya_estaba"
        assert "Starting server" in mgr.log_path("comfyui").read_text(encoding="utf-8", errors="replace")
        st = mgr.status(s)
        assert st["comfyui"]["encendido"] and st["comfyui"]["lanzado_por_plasmo"] and st["comfyui"]["tipo"] == "comando"
    finally:
        stopped = {n: mgr.stop(n) for n in engines.ENGINES}
    assert stopped == {"ollama": True, "comfyui": True}
    time.sleep(0.5)
    assert not mgr.running("ollama", s) and not mgr.running("comfyui", s)
    assert mgr.stop("ollama") is False  # ya no es nuestro


def test_crash_reports_log_tail(tmp_path):
    bad = f'"{sys.executable}" -c "print(\'arranque roto\'); raise SystemExit(3)"'
    s = Settings(comfy_url=f"http://127.0.0.1:{free_port()}", comfy_comando=bad)
    with pytest.raises(engines.EngineError, match="arranque roto"):
        engines.EngineManager(tmp_path / "logs").start("comfyui", s)


def test_api_turns_engines_on_and_off(tmp_path):
    ollama_port, comfy_port = free_port(), free_port()
    cfg = ConfigStore(tmp_path / "config.json")
    cfg.update({
        "ollama_url": f"http://127.0.0.1:{ollama_port}", "comfy_url": f"http://127.0.0.1:{comfy_port}",
        "ollama_comando": mock_cmd("mock_ollama.py", ollama_port), "comfy_comando": mock_cmd("mock_comfyui.py", comfy_port),
        "output_dir": str(tmp_path / "proyectos"),
    })
    app = create_app(cfg, JobManager(timings=Timings(tmp_path / "t.json")), engines.EngineManager(tmp_path / "logs"))
    with TestClient(app) as client:
        before = client.get("/api/motores").json()
        assert not before["ollama"]["encendido"] and before["comfyui"]["encontrado"]
        job = client.post("/api/motores/encender", json={}, headers=H).json()
        assert [f["clave"] for f in job["fases"]] == ["ollama", "comfyui"]
        for _ in range(240):
            j = client.get(f"/api/trabajos/{job['id']}").json()
            if j["estado"] in ("listo", "error"):
                break
            time.sleep(0.25)
        assert j["estado"] == "listo", j
        assert {r["estado"] for r in j["resultado"]["resultados"].values()} == {"encendido"}
        assert client.get("/api/estado").json()["generador"]["ok"]
        assert client.post("/api/motores/apagar", json={"motores": ["comfyui", "ollama"]}, headers=H).json() == {"comfyui": True, "ollama": True}
        assert client.post("/api/motores/encender", json={"motores": ["gpu"]}, headers=H).status_code == 400
