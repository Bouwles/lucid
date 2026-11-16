import json

from fastapi.testclient import TestClient

from lucid.play.server import create_app

from test_engine import make_engine


def test_health_and_websocket_protocol():
    app = create_app(make_engine, max_players=1, idle_timeout=30, target_fps=50, web_dir=None)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["t"] == "hello" and hello["resolution"] == [16, 16]
        assert {g["name"] for g in hello["games"]} == {"a", "b"}
        ws.send_text(json.dumps({"t": "reset", "game": "a"}))
        ws.send_text(json.dumps({"t": "act", "buttons": 16, "look": [0, 0]}))
        ws.send_text(json.dumps({"t": "config", "sampler_steps": 1}))
        frames = 0
        while frames < 3:
            msg = ws.receive()
            if msg.get("bytes"):
                data = msg["bytes"]
                assert data[0] == 0 and data[1:5] == b"RIFF" and data[9:13] == b"WEBP"
                frames += 1
        # a second player is turned away while the first is connected
        with client.websocket_connect("/ws") as ws2:
            assert json.loads(ws2.receive_text())["t"] == "busy"
