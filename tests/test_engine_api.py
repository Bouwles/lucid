import json
import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from lucid_engine.app import create_app  # noqa: E402
from lucid_engine.state import EngineState  # noqa: E402

from conftest import ROOT  # noqa: E402
from engine_fixtures import make_pack  # noqa: E402

TOKEN = "t" * 43


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    bundled = tmp_path_factory.mktemp("bundled")
    make_pack(bundled)
    state = EngineState(tmp_path_factory.mktemp("data"), bundled)
    state._hardware = {"os": "test", "summary": "test", "dream_fps": 1.0}  # skip the benchmark
    c = TestClient(create_app(state, TOKEN))
    c.headers["Authorization"] = f"Bearer {TOKEN}"
    yield c
    state.jobs.shutdown()


def test_token_is_required(client):
    raw = TestClient(client.app)
    assert raw.get("/health").status_code == 401
    assert raw.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert raw.get(f"/health?token={TOKEN}").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_catalogue_and_library(client):
    games = {g["id"]: g for g in client.get("/games").json()}
    assert games["test-track"]["dream"]["installed"] and games["test-track"]["dream"]["bundled"]
    assert (
        games["drift-kit"]["status"] == "coming_soon"
        and not games["drift-kit"]["dream"]["installed"]
    )
    lib = client.get("/library").json()
    assert lib["models"][0]["bundled"] and lib["free_bytes"] > 0
    assert client.delete("/library/models/test-track").status_code == 409
    assert client.get("/media/test-track.png").status_code == 200
    assert client.get("/media/..%2Fcatalog.json").status_code == 404


def test_settings_roundtrip(client):
    s = client.put("/settings", json={"seen_intro": True, "recording_consent": False}).json()
    assert s["seen_intro"] is True and client.get("/settings").json()["recording_consent"] is False


def test_jobs_endpoint(client):
    assert client.post("/jobs", json={"kind": "nope"}).status_code == 400
    j = client.post(
        "/jobs", json={"kind": "download_model", "params": {"game": "drift-kit"}}
    ).json()
    for _ in range(100):
        j = client.get(f"/jobs/{j['id']}").json()
        if j["state"] in ("failed", "done"):
            break
        time.sleep(0.02)
    assert j["state"] == "failed" and "no dream model" in j["error"]


def test_dream_websocket(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/dream") as ws:
            ws.receive_text()
    with client.websocket_connect(f"/dream?token={TOKEN}") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["t"] == "hello" and hello["games"][0]["name"] == "test-track"
        ws.send_text(json.dumps({"t": "stream", "fps": 60}))
        ws.send_text(json.dumps({"t": "reset", "game": "test-track"}))
        frames = 0
        while frames < 3:
            msg = ws.receive()
            if msg.get("bytes"):
                assert msg["bytes"][0] == 0 and msg["bytes"][9:13] == b"WEBP"
                frames += 1


def test_process_binds_localhost_and_prints_ready_line(tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'engine'}{os.pathsep}{ROOT}"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "lucid_engine", "--data-dir", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("LUCID_ENGINE_READY ")
        info = json.loads(line.split(" ", 1)[1])
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{info['port']}", "-sTCP:LISTEN"], capture_output=True, text=True
        ).stdout
        if out:  # lsof is not on every CI image
            assert "127.0.0.1:" in out and "*:" not in out
        proc.stdin.close()  # closing stdin must stop the engine
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
