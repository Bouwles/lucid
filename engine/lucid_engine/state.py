"""Everything the engine knows: data folders, catalogue, installed dream packs, settings, jobs."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

from lucid_engine import hardware
from lucid_engine.dream import DreamPack, benchmark, find_packs
from lucid_engine.jobs import JobContext, JobManager

ALLOWED_DOWNLOAD_PREFIX = "https://github.com/Bouwles/lucid/releases/download/"
DEFAULT_SETTINGS = {
    "seen_intro": False,
    "recording_consent": None,
    "sampler_steps": None,
    "last_dream": None,
}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class EngineState:
    def __init__(self, data_dir: Path, bundled_models: Path | None = None):
        self.data = Path(data_dir)
        for sub in ("models", "games", "projects", "jobs", "logs"):
            (self.data / sub).mkdir(parents=True, exist_ok=True)
        self.bundled = Path(bundled_models) if bundled_models else None
        self.catalog = json.loads(
            resources.files("lucid_engine").joinpath("catalog.json").read_text()
        )["games"]
        self._lock = threading.RLock()
        self._hardware: dict | None = None
        self.refresh_packs()
        self.jobs = JobManager(self.data / "jobs")
        self.jobs.register("download_model", self._job_download_model)
        self.jobs.register("benchmark", self._job_benchmark)

    # --------------------------------------------------------------- packs and catalogue

    def refresh_packs(self) -> None:
        with self._lock:
            self.packs: dict[str, DreamPack] = find_packs(
                (self.bundled, True), (self.data / "models", False)
            )

    def pack_for(self, game_id: str) -> DreamPack | None:
        return next((p for p in self.packs.values() if p.game == game_id), None)

    def game_entry(self, g: dict) -> dict:
        pack = self.pack_for(g["id"])
        game_dir = self.data / "games" / g["id"]
        media = {k: (f"/media/{v}" if v else None) for k, v in g["media"].items()}
        return {
            **{
                k: g[k]
                for k in (
                    "id",
                    "name",
                    "genre",
                    "tagline",
                    "description",
                    "accent",
                    "status",
                    "controls",
                )
            },
            "game": {
                "downloadable": bool(g["game"]["url"]),
                "installed": game_dir.exists(),
                "size_mb": g["game"]["size_mb"],
            },
            "dream": {
                "installed": pack is not None,
                "bundled": bool(pack and pack.bundled),
                "size_mb": round(pack.size_bytes() / 1e6, 1) if pack else g["dream"]["size_mb"],
                "quality": [pack.meta.get("quality", "fast")] if pack else [],
                "downloadable": bool(g["dream"]["url"]) and pack is None,
            },
            "media": media,
        }

    def games(self) -> list[dict]:
        return [self.game_entry(g) for g in self.catalog]

    def catalog_game(self, game_id: str) -> dict:
        for g in self.catalog:
            if g["id"] == game_id:
                return g
        raise KeyError(game_id)

    # --------------------------------------------------------------- settings

    @property
    def settings_path(self) -> Path:
        return self.data / "settings.json"

    def settings(self) -> dict:
        try:
            saved = json.loads(self.settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            saved = {}
        return {**DEFAULT_SETTINGS, **saved}

    def update_settings(self, changes: dict) -> dict:
        with self._lock:
            s = {**self.settings(), **changes}
            tmp = self.settings_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=1))
            tmp.replace(self.settings_path)
            return s

    # --------------------------------------------------------------- hardware

    def hardware(self) -> dict:
        with self._lock:
            if self._hardware is None:
                info = hardware.detect()
                info["training_pack_installed"] = (self.data / "training-pack").exists()
                pack = next(iter(self.packs.values()), None)
                fps = None
                if pack is not None:
                    try:
                        fps = round(benchmark(pack), 1)
                    except Exception:  # a broken pack must not break the hardware page
                        fps = None
                info["dream_fps"] = fps
                info["dream_fps_measured"] = fps is not None
                info["summary"] = hardware.summary(info, fps)
                self._hardware = info
            return dict(self._hardware)

    # --------------------------------------------------------------- library

    def library(self) -> dict:
        models = [
            {
                "id": p.id,
                "game": p.game,
                "size_bytes": p.size_bytes(),
                "bundled": p.bundled,
                "path": str(p.path),
            }
            for p in self.packs.values()
        ]
        games = [
            {"id": d.name, "size_bytes": dir_size(d), "path": str(d)}
            for d in sorted((self.data / "games").iterdir())
            if d.is_dir()
        ]
        projects = [
            {"id": d.name, "size_bytes": dir_size(d), "path": str(d)}
            for d in sorted((self.data / "projects").iterdir())
            if d.is_dir()
        ]
        usage = shutil.disk_usage(self.data)
        return {
            "models": models,
            "games": games,
            "projects": projects,
            "total_bytes": sum(
                x["size_bytes"] for x in models + games + projects if not x.get("bundled")
            ),
            "free_bytes": usage.free,
            "data_dir": str(self.data),
        }

    def delete(self, kind: str, item_id: str) -> None:
        if kind == "models":
            pack = self.packs.get(item_id)
            if pack is None:
                raise KeyError(item_id)
            if pack.bundled:
                raise PermissionError("bundled models come with the app and can't be deleted")
            shutil.rmtree(pack.path)
            self.refresh_packs()
            return
        if kind not in ("games", "projects"):
            raise KeyError(kind)
        target = (self.data / kind / item_id).resolve()
        if target.parent != (self.data / kind).resolve() or not target.is_dir():
            raise KeyError(item_id)
        shutil.rmtree(target)

    # --------------------------------------------------------------- jobs

    def _job_benchmark(self, ctx: JobContext) -> dict:
        with self._lock:
            self._hardware = None
        ctx.progress(0.1, "Measuring dream speed")
        return {"dream_fps": self.hardware()["dream_fps"]}

    def _job_download_model(self, ctx: JobContext) -> dict:
        """Download a dream pack zip from the catalogue URL, resuming if paused, and verify it."""
        g = self.catalog_game(ctx.params["game"])
        url, sha, size_mb = g["dream"]["url"], g["dream"]["sha256"], g["dream"]["size_mb"]
        if not url or not sha:
            raise ValueError(f"{g['name']} has no dream model to download yet")
        if not url.startswith(ALLOWED_DOWNLOAD_PREFIX):
            raise ValueError("downloads only come from the LUCID GitHub Releases")
        return download_and_install(
            ctx, url, sha, self.data / "models", g["id"], size_mb, self.refresh_packs
        )


def download_and_install(
    ctx: JobContext,
    url: str,
    sha256: str,
    models_dir: Path,
    game_id: str,
    size_mb: float | None,
    on_done,
    opener=urllib.request.urlopen,
) -> dict:
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    part = ctx.scratch / "download.part"
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with opener(req, timeout=30) as resp:
        if have and getattr(resp, "status", 200) != 206:  # server ignored Range: start again
            have = 0
        total = (
            have + int(resp.headers.get("Content-Length") or 0) or int((size_mb or 0) * 1e6) or None
        )
        with open(part, "ab" if have else "wb") as f:
            got = have
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    ctx.progress(
                        0.95 * got / total,
                        f"Downloading dream model ({got / 1e6:.0f} of {total / 1e6:.0f} MB)",
                    )
                else:
                    ctx.check()
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != sha256:
        part.unlink()
        raise ValueError(
            "the download didn't match its checksum, so it wasn't installed; try again"
        )
    ctx.progress(0.97, "Installing")
    dest = models_dir / game_id
    tmp = models_dir / f".{game_id}.installing"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(part) as z:
        for name in z.namelist():  # refuse paths that would escape the folder
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("unsafe path in download")
        z.extractall(tmp)
    if not (tmp / "dream.json").exists():
        raise ValueError("the download isn't a dream pack")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    shutil.rmtree(ctx.scratch, ignore_errors=True)
    on_done()
    return {"installed": str(dest)}
