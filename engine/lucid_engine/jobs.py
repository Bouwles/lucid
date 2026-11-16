"""A small persistent job queue for long tasks (downloads, benchmarks; later imports and training).

Every job is a JSON file in <data dir>/jobs/. State changes are written immediately, so a job
survives the app quitting: anything that was running comes back as paused, and resuming picks
up where it stopped. One worker thread runs jobs in order.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("lucid.jobs")
ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Paused(Exception):
    pass


class Cancelled(Exception):
    pass


class JobContext:
    """What a job handler gets: its job record, a scratch folder, and progress/stop controls."""

    def __init__(self, manager: JobManager, job: dict):
        self.manager = manager
        self.job = job
        self.scratch = manager.root / job["id"]
        self._t0 = time.monotonic()
        self._p0 = job.get("progress", 0.0)

    @property
    def params(self) -> dict:
        return self.job["params"]

    def check(self) -> None:
        """Call often. Raises Paused or Cancelled when the user asked for that."""
        want = self.manager._requests.get(self.job["id"])
        if want == "cancel":
            raise Cancelled
        if want == "pause":
            raise Paused

    def progress(self, fraction: float, message: str = "") -> None:
        fraction = max(0.0, min(1.0, fraction))
        elapsed = time.monotonic() - self._t0
        done = fraction - self._p0
        eta = elapsed * (1 - fraction) / done if done > 1e-6 and elapsed > 1 else None
        self.manager._update(
            self.job["id"],
            progress=fraction,
            message=message,
            eta_s=None if eta is None else round(eta, 1),
        )
        self.check()


Handler = Callable[[JobContext], dict | None]


class JobManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.handlers: dict[str, Handler] = {}
        self._jobs: dict[str, dict] = {}
        self._requests: dict[str, str] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        for f in sorted(self.root.glob("*.json")):
            try:
                job = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("state") in ACTIVE:
                job["state"], job["eta_s"] = "paused", None
                job["message"] = "Paused when LUCID closed. Resume to continue."
            self._jobs[job["id"]] = job
            self._save(job)
        self._worker = threading.Thread(target=self._run, name="lucid-jobs", daemon=True)
        self._worker.start()

    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    # --------------------------------------------------------------- public API

    def submit(self, kind: str, params: dict | None = None) -> dict:
        if kind not in self.handlers:
            raise KeyError(f"unknown job kind {kind!r}")
        job = {
            "id": "j_" + uuid.uuid4().hex[:12],
            "kind": kind,
            "params": params or {},
            "state": "queued",
            "progress": 0.0,
            "message": "Waiting to start",
            "eta_s": None,
            "created_utc": _now(),
            "updated_utc": _now(),
            "error": None,
            "result": None,
        }
        with self._lock:
            self._jobs[job["id"]] = job
            self._save(job)
        self._queue.put(job["id"])
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            return dict(self._jobs[job_id])

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(
                (dict(j) for j in self._jobs.values()), key=lambda j: j["created_utc"], reverse=True
            )

    def pause(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] == "queued":
                self._update(job_id, state="paused", message="Paused")
            elif job["state"] == "running":
                self._requests[job_id] = "pause"
            return dict(job)

    def resume(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] in ("paused", "failed"):
                self._requests.pop(job_id, None)
                self._update(job_id, state="queued", message="Waiting to start", error=None)
                self._queue.put(job_id)
            return dict(job)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] == "running":
                self._requests[job_id] = "cancel"
            elif job["state"] in ("queued", "paused", "failed"):
                self._update(job_id, state="cancelled", message="Cancelled", eta_s=None)
            return dict(job)

    def wait(
        self, job_id: str, timeout: float = 30.0, states=("done", "failed", "cancelled", "paused")
    ) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            j = self.get(job_id)
            if j["state"] in states:
                return j
            time.sleep(0.02)
        raise TimeoutError(job_id)

    def shutdown(self) -> None:
        with self._lock:
            for jid, job in self._jobs.items():
                if job["state"] == "running":
                    self._requests[jid] = "pause"
        self._queue.put(None)
        self._worker.join(timeout=5)

    # --------------------------------------------------------------- internals

    def _save(self, job: dict) -> None:
        tmp = self.root / f"{job['id']}.json.tmp"
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(self.root / f"{job['id']}.json")

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields, updated_utc=_now())
            self._save(job)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["state"] != "queued":
                    continue
                self._update(job_id, state="running", message="Starting")
            ctx = JobContext(self, job)
            try:
                result = self.handlers[job["kind"]](ctx)
                self._update(
                    job_id, state="done", progress=1.0, message="Done", eta_s=None, result=result
                )
            except Paused:
                self._update(job_id, state="paused", message="Paused", eta_s=None)
            except Cancelled:
                self._update(job_id, state="cancelled", message="Cancelled", eta_s=None)
            except Exception as e:  # a failed job must never take the engine down
                log.exception("job %s failed", job_id)
                self._update(job_id, state="failed", message="Failed", eta_s=None, error=str(e))
            finally:
                self._requests.pop(job_id, None)
