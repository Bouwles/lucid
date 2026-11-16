import hashlib
import io
import time
import zipfile

import pytest
from lucid_engine.jobs import JobManager
from lucid_engine.state import download_and_install


def slow_job(steps=50, delay=0.01):
    def handler(ctx):
        start = int(ctx.job.get("progress", 0) * steps)
        for i in range(start, steps):
            time.sleep(delay)
            ctx.progress((i + 1) / steps, f"step {i + 1}")
        return {"steps": steps}

    return handler


def test_runs_to_completion(tmp_path):
    m = JobManager(tmp_path)
    m.register("slow", slow_job(10))
    j = m.wait(m.submit("slow")["id"])
    assert j["state"] == "done" and j["progress"] == 1.0 and j["result"] == {"steps": 10}
    m.shutdown()


def test_pause_resume_cancel(tmp_path):
    m = JobManager(tmp_path)
    m.register("slow", slow_job(200, 0.005))
    jid = m.submit("slow")["id"]
    m.wait(jid, states=("running",))
    time.sleep(0.1)
    m.pause(jid)
    j = m.wait(jid)
    assert j["state"] == "paused" and 0 < j["progress"] < 1
    m.resume(jid)
    m.wait(jid, states=("running",))
    m.cancel(jid)
    assert m.wait(jid)["state"] == "cancelled"
    m.shutdown()


def test_failure_is_recorded_not_raised(tmp_path):
    m = JobManager(tmp_path)

    def boom(ctx):
        raise RuntimeError("disk on fire")

    m.register("boom", boom)
    j = m.wait(m.submit("boom")["id"])
    assert j["state"] == "failed" and "disk on fire" in j["error"]
    m.shutdown()


def test_jobs_survive_a_restart(tmp_path):
    """A job file left as "running" (the app quit or crashed mid-job) comes back paused and resumes."""
    import json

    job = {
        "id": "j_crashed",
        "kind": "slow",
        "params": {},
        "state": "running",
        "progress": 0.5,
        "message": "step 100",
        "eta_s": 3.0,
        "created_utc": "2026-01-01T00:00:00Z",
        "updated_utc": "2026-01-01T00:00:00Z",
        "error": None,
        "result": None,
    }
    (tmp_path / "j_crashed.json").write_text(json.dumps(job))
    m = JobManager(tmp_path)
    m.register("slow", slow_job(200, 0.001))
    j = m.get("j_crashed")
    assert j["state"] == "paused" and j["progress"] == 0.5
    assert json.loads((tmp_path / "j_crashed.json").read_text())["state"] == "paused"
    m.resume("j_crashed")
    assert m.wait("j_crashed", timeout=30)["state"] == "done"
    m.shutdown()


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("dream.json", '{"format": 1, "id": "x"}')
        z.writestr("denoiser.onnx", b"\x00" * 1000)
        z.writestr("starts.npz", b"")
    return buf.getvalue()


def test_download_verifies_checksum_and_installs(tmp_path):
    data = make_zip()
    m = JobManager(tmp_path / "jobs")
    done = []

    def handler(ctx):
        return download_and_install(
            ctx,
            "https://example/x.zip",
            hashlib.sha256(data).hexdigest(),
            tmp_path / "models",
            "x",
            None,
            lambda: done.append(1),
            opener=lambda req, timeout: FakeResponse(data),
        )

    m.register("dl", handler)
    j = m.wait(m.submit("dl")["id"])
    assert j["state"] == "done" and (tmp_path / "models/x/dream.json").exists() and done


def test_download_with_wrong_checksum_is_rejected(tmp_path):
    data = make_zip()
    m = JobManager(tmp_path / "jobs")
    m.register(
        "dl",
        lambda ctx: download_and_install(
            ctx,
            "https://example/x.zip",
            "0" * 64,
            tmp_path / "models",
            "x",
            None,
            lambda: None,
            opener=lambda req, timeout: FakeResponse(data),
        ),
    )
    j = m.wait(m.submit("dl")["id"])
    assert j["state"] == "failed" and "checksum" in j["error"]
    assert not (tmp_path / "models/x").exists()


def test_unknown_kind(tmp_path):
    with pytest.raises(KeyError):
        JobManager(tmp_path).submit("nope")
