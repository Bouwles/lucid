"""Controllability (SPEC 17.4): does the dream obey the player?

From many real starting contexts, roll out short sequences that differ only in the held action,
read the result with the probes, and count how often the response matches the action.
"""

from __future__ import annotations

import numpy as np
import torch

from lucid.data.format import BUTTONS, VERTICAL_STATES
from lucid.eval.probes import predict
from lucid.eval.rollout import rollout

# (test name, button held, probe, check(start_value, values_over_rollout, num_classes) -> correct?)
LANE_TESTS = {
    "left": ("left", "lane", lambda s, v, n: v[-1] < s if s > 0 else v[-1] == 0),
    "right": ("right", "lane", lambda s, v, n: v[-1] > s if s < n - 1 else v[-1] == n - 1),
    "none": (None, "lane", lambda s, v, n: v[-1] == s),
}
GROUNDED = VERTICAL_STATES.index("grounded")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
VERTICAL_TESTS = {
    "jump": ("jump", "vertical", lambda s, v, n: (v == VERTICAL_STATES.index("jumping")).any()),
    "down": ("down", "vertical", lambda s, v, n: (v == VERTICAL_STATES.index("sliding")).any()),
}


@torch.no_grad()
def controllability(
    model,
    probes: dict,
    starts: list[dict],
    used_buttons: set[str],
    hold_steps: int = 8,
    history: bool = True,
    gen_kwargs: dict | None = None,
    batch_size: int = 32,
) -> dict:
    """starts: SequenceDataset items already prepared on the device (frames float, buttons, look, game_id),
    each with at least k real context frames. Returns per-test scores and confusion counts.
    """
    gen_kwargs = gen_kwargs or {}
    tests = {}
    if "lane" in probes:
        tests.update({k: v for k, v in LANE_TESTS.items() if v[0] is None or v[0] in used_buttons})
    if "vertical" in probes:
        tests.update({k: v for k, v in VERTICAL_TESTS.items() if v[0] in used_buttons})
    k = model.k
    results = {}
    for name, (button, probe_name, check) in tests.items():
        probe = probes[probe_name]
        n_classes = probe.net[-1].out_features
        correct, total = 0, 0
        confusion = np.zeros((n_classes, n_classes), np.int64)  # start state -> end state
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            frames = torch.stack([s["frames"][:k] for s in chunk])
            game = torch.stack([s["game_id"] for s in chunk])
            B = len(chunk)
            buttons = torch.zeros(B, k + hold_steps - 1, 8, device=frames.device)
            look = torch.zeros(B, k + hold_steps - 1, 2, device=frames.device)
            buttons[:, : k - 1] = torch.stack([s["buttons"][: k - 1] for s in chunk])  # real past
            look[:, : k - 1] = torch.stack([s["look"][: k - 1] for s in chunk])
            if button is not None:
                buttons[:, k - 1 :, BUTTONS.index(button)] = 1
            pred = rollout(model, frames, buttons, look, game, hold_steps, history, **gen_kwargs)
            start_v = predict(probe, frames[:, -1]).cpu().numpy()
            vals = predict(probe, pred.flatten(0, 1)).reshape(B, hold_steps).cpu().numpy()
            for b in range(B):
                # Jump/slide only do anything from the ground, in the dream as in the game.
                if probe_name == "vertical" and start_v[b] != GROUNDED:
                    continue
                ok = bool(check(start_v[b], vals[b], n_classes))
                correct += ok
                total += 1
                confusion[start_v[b], vals[b, -1]] += 1
        results[name] = {
            "score": correct / max(total, 1),
            "n": total,
            "confusion": confusion.tolist(),
        }
    return results


def real_controllability(episodes, hold_steps: int = 8) -> dict:
    """The same checks on real recorded data, using the true labels (should be near 100%).

    Real play rarely holds one button for exactly hold_steps from a clean start, so this finds
    every place a button goes down and keeps only the cases comparable to the dream test: the
    player stays alive, lane tests hold the button for the whole window without its opposite,
    and jump/slide tests start on the ground.
    """
    lane_eps = [e for e in episodes if "lane" in e.labels and (e.labels["lane"] >= 0).any()]
    n_lanes = max((int(e.labels["lane"].max()) + 1 for e in lane_eps), default=0)
    out = {}
    for name, (button, key, check) in {**LANE_TESTS, **VERTICAL_TESTS}.items():
        if button is None:
            continue
        bi = BUTTONS.index(button)
        opp = OPPOSITE.get(button)
        n = n_lanes if key == "lane" else len(VERTICAL_STATES)
        correct = total = 0
        for e in lane_eps if key == "lane" else episodes:
            if key not in e.labels:
                continue
            b = e.buttons[:, bi]
            for t in np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1:
                w = slice(t, t + hold_steps)
                if t + hold_steps >= len(e) or not e.labels["alive"][t : t + hold_steps + 1].all():
                    continue
                if key == "lane" and (
                    not e.buttons[w, bi].all()
                    or (opp is not None and e.buttons[w, BUTTONS.index(opp)].any())
                ):
                    continue
                if key == "vertical" and int(e.labels["vertical"][t]) != GROUNDED:
                    continue
                s_ = int(e.labels[key][t])
                v = e.labels[key][t + 1 : t + 1 + hold_steps].astype(int)
                total += 1
                correct += bool(check(s_, v, n))
        if total:
            out[name] = {"score": correct / total, "n": total}
    return out
