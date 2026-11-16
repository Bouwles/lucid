import subprocess
from pathlib import Path


def git_commit(short: bool = False) -> str:
    """Current commit hash, with "-dirty" if there are uncommitted changes. "unknown" outside git."""
    root = Path(__file__).resolve().parents[2]
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        commit = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
