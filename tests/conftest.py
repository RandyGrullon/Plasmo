import socket
import time

import httpx
import pytest

from tools import mock_comfyui, mock_ollama


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise RuntimeError(f"no arrancó {url}")


@pytest.fixture(scope="session")
def ollama_url():
    port = free_port()
    server = mock_ollama.serve_in_thread(port)
    url = f"http://127.0.0.1:{port}"
    _wait(url + "/api/version")
    yield url
    server.should_exit = True


@pytest.fixture(scope="session")
def comfy_url():
    port = free_port()
    server = mock_comfyui.serve_in_thread(port)
    url = f"http://127.0.0.1:{port}"
    _wait(url + "/system_stats")
    yield url
    server.should_exit = True
