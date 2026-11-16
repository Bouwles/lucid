import json
import shutil

from lucid.data.validate import validate_session
from lucid.utils.config import load_games

from conftest import ROOT

GAMES = load_games(ROOT / "configs/games_synthetic.yaml")


def _copy(session, tmp_path):
    dst = tmp_path / session.name
    shutil.copytree(session, dst)
    return dst


def _reasons(session):
    return {r.episode_id: r.reasons for r in validate_session(session, GAMES) if not r.ok}


def test_clean_session_passes(synthetic_session):
    assert _reasons(synthetic_session) == {}


def test_missing_frame_fails(synthetic_session, tmp_path):
    s = _copy(synthetic_session, tmp_path)
    (s / "episodes/ep_000001/frames/000003.png").unlink()
    assert "frames but" in " ".join(_reasons(s)["ep_000001"])


def test_gap_in_steps_and_bad_end_fail(synthetic_session, tmp_path):
    s = _copy(synthetic_session, tmp_path)
    p = s / "episodes/ep_000002/steps.jsonl"
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    lines[5]["step"] = 99
    lines[3]["episode_end"] = True
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    r = " ".join(_reasons(s)["ep_000002"])
    assert "contiguous" in r and "episode_end" in r


def test_unmapped_button_fails(synthetic_session, tmp_path):
    s = _copy(synthetic_session, tmp_path)
    p = s / "episodes/ep_000003/steps.jsonl"
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    lines[2]["buttons"] = 1 << 5  # action_a, not mapped for this game
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert "does not map" in " ".join(_reasons(s)["ep_000003"])


def test_corrupted_flag_fails(synthetic_session, tmp_path):
    s = _copy(synthetic_session, tmp_path)
    p = s / "episodes/ep_000004/meta.json"
    m = json.loads(p.read_text())
    m.update(corrupted=True, corruption_reason="disk full")
    p.write_text(json.dumps(m))
    assert "disk full" in " ".join(_reasons(s)["ep_000004"])


def test_frozen_game_fails(synthetic_session, tmp_path):
    s = _copy(synthetic_session, tmp_path)
    frames = s / "episodes/ep_000005/frames"
    first = (frames / "000000.png").read_bytes()
    for i in range(1, 60):
        (frames / f"{i:06d}.png").write_bytes(first)
    reasons = {
        r.episode_id: r.reasons
        for r in validate_session(s, GAMES, max_identical_seconds=2.0)
        if not r.ok
    }
    assert "identical frames" in " ".join(reasons["ep_000005"])
