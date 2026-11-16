import json
from pathlib import Path

import numpy as np

from lucid.data.format import (
    button_bit,
    cancel_opposites,
    pack_buttons,
    read_episode_meta,
    read_session,
    read_steps,
    unpack_buttons,
)
from lucid.data.validate import validate_session
from lucid.utils.config import load_games

FIXTURE = Path(__file__).parent / "fixtures" / "golden_session"
ROOT = Path(__file__).resolve().parents[1]


def test_button_packing_roundtrip():
    b = np.arange(256, dtype=np.uint8)
    assert (pack_buttons(unpack_buttons(b)) == b).all()
    assert unpack_buttons(button_bit("jump")).tolist() == [0, 0, 0, 0, 1, 0, 0, 0]


def test_opposites_cancel():
    lr = button_bit("left") | button_bit("right")
    assert cancel_opposites(lr) == 0
    assert cancel_opposites(lr | button_bit("jump")) == button_bit("jump")


def test_golden_episode_matches_the_contract():
    """A small checked-in episode (tests/fixtures) in exactly the docs/formats.md layout."""
    s = read_session(FIXTURE)
    assert s.format_version == 1 and s.game_id == "synthetic"
    ep = FIXTURE / "episodes" / "ep_000000"
    meta = read_episode_meta(ep)
    steps = read_steps(ep)
    assert meta.num_steps == len(steps) == len(list((ep / "frames").glob("*.png")))
    raw = json.loads((ep / "steps.jsonl").read_text().splitlines()[0])
    assert set(raw) == {"step", "time", "buttons", "look", "state", "episode_end"}
    assert set(raw["state"]) == {"lane", "x", "y", "vertical", "alive", "speed", "distance"}
    reports = validate_session(FIXTURE, load_games(ROOT / "configs/games_synthetic.yaml"))
    assert all(r.ok for r in reports), [r.reasons for r in reports]
