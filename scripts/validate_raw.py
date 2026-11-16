"""Validate raw recording sessions without building anything."""

import argparse
import sys
from pathlib import Path

from lucid.data.validate import validate_session, write_report
from lucid.utils.config import load_games


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sessions", nargs="+", help="session directories (data/raw/<session_id>)")
    p.add_argument("--games-file", default="configs/games.yaml")
    p.add_argument("--report", default=None, help="where to write the Markdown report")
    p.add_argument("--max-identical-seconds", type=float, default=3.0)
    args = p.parse_args()
    games = load_games(args.games_file)
    reports = []
    for s in args.sessions:
        reports += validate_session(s, games, args.max_identical_seconds)
    ok = sum(r.ok for r in reports)
    print(f"{ok} passed, {len(reports) - ok} failed")
    for r in reports:
        if not r.ok:
            print(f"  {r.session_id}/{r.episode_id}: {'; '.join(r.reasons)}")
    report = args.report or str(Path(args.sessions[0]) / "validation.md")
    write_report(reports, report)
    print(f"report: {report}")
    sys.exit(0 if ok == len(reports) else 1)


if __name__ == "__main__":
    main()
