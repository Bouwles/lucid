// Keyboard and mouse to unified actions, for normal play in the app.

import { BUTTON, type Action } from "./types";

const KEYS: Record<string, number> = {
  ArrowLeft: BUTTON.left, KeyA: BUTTON.left,
  ArrowRight: BUTTON.right, KeyD: BUTTON.right,
  ArrowUp: BUTTON.up, KeyW: BUTTON.up,
  ArrowDown: BUTTON.down, KeyS: BUTTON.down,
  Space: BUTTON.jump,
  KeyZ: BUTTON.actionA, KeyJ: BUTTON.actionA,
  KeyX: BUTTON.actionB, KeyK: BUTTON.actionB,
};

export function isGameKey(code: string): boolean {
  return code in KEYS;
}

/** Left+right cancel out, as do up+down (the same rule as the recorders). */
export function cancelOpposites(b: number): number {
  for (const [x, y] of [[BUTTON.left, BUTTON.right], [BUTTON.up, BUTTON.down]]) {
    if ((b & x) && (b & y)) b &= ~(x | y);
  }
  return b;
}

export class InputState {
  private held = new Set<string>();
  private mouseFire = false;
  private lookX = 0;
  private lookY = 0;
  /** Mouse delta in pixels that maps to look = 1 in one tick. */
  lookScale = 12;

  keyDown(code: string): boolean {
    if (!(code in KEYS)) return false;
    this.held.add(code);
    return true;
  }

  keyUp(code: string): boolean {
    if (!(code in KEYS)) return false;
    this.held.delete(code);
    return true;
  }

  mouseMove(dx: number, dy: number): void {
    this.lookX += dx;
    this.lookY += dy;
  }

  /** Left mouse button acts as action_a (fire) while held. */
  mouseButton(down: boolean): void {
    this.mouseFire = down;
  }

  releaseAll(): void {
    this.held.clear();
    this.mouseFire = false;
    this.lookX = this.lookY = 0;
  }

  /** The action for one tick. Mouse movement since the last tick becomes the look vector. */
  take(): Action {
    let b = 0;
    for (const code of this.held) b |= KEYS[code];
    if (this.mouseFire) b |= BUTTON.actionA;
    const clamp = (v: number) => Math.max(-1, Math.min(1, v));
    // Screen y grows downwards; look[1] is positive when looking up.
    const look: [number, number] = [clamp(this.lookX / this.lookScale), clamp(-this.lookY / this.lookScale)];
    this.lookX = this.lookY = 0;
    return { buttons: cancelOpposites(b), look };
  }
}
