"""Play the dream in a local window with the keyboard.

Arrows/WASD move, space jumps, Z/X are action buttons, R resets, 1-9 switch game, Esc quits.
Needs pygame: pip install -e ".[play]".
"""

import argparse
import time

import numpy as np

from lucid.data.format import button_bit, cancel_opposites
from lucid.play.engine import DreamEngine
from lucid.utils.seed import seed_everything


def main() -> None:
    import pygame

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--game", default=None)
    args = p.parse_args()
    seed_everything(0)
    engine = DreamEngine(args.checkpoint, device=args.device, sampler_steps=args.steps)
    games = list(engine.games)
    frame = engine.reset(args.game or games[0])
    H, W = frame.shape[:2]
    pygame.init()
    screen = pygame.display.set_mode((W * args.scale, H * args.scale + 24))
    pygame.display.set_caption("LUCID")
    font = pygame.font.SysFont(None, 20)
    keymap = {
        pygame.K_LEFT: "left",
        pygame.K_a: "left",
        pygame.K_RIGHT: "right",
        pygame.K_d: "right",
        pygame.K_UP: "up",
        pygame.K_w: "up",
        pygame.K_DOWN: "down",
        pygame.K_s: "down",
        pygame.K_SPACE: "jump",
        pygame.K_z: "action_a",
        pygame.K_x: "action_b",
    }
    clock = pygame.time.Clock()
    gen_ms: list[float] = []
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                frame = engine.reset()
            elif ev.type == pygame.KEYDOWN and pygame.K_1 <= ev.key <= pygame.K_9:
                i = ev.key - pygame.K_1
                if i < len(games):
                    engine.set_game(games[i])
        keys = pygame.key.get_pressed()
        buttons = 0
        for k, name in keymap.items():
            if keys[k]:
                buttons |= button_bit(name)
        t = time.perf_counter()
        frame = engine.step(cancel_opposites(buttons))
        gen_ms.append(1000 * (time.perf_counter() - t))
        gen_ms = gen_ms[-30:]
        img = np.ascontiguousarray(
            frame.repeat(args.scale, 0).repeat(args.scale, 1).transpose(1, 0, 2)
        )
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(img), (0, 0))
        info = f"{engine.game}  gen {np.mean(gen_ms):.1f} ms  display {clock.get_fps():.1f} fps  steps {engine.sampler_steps}"
        screen.blit(font.render(info, True, (220, 220, 220)), (6, H * args.scale + 4))
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
