// Burrs: slow walkers that patrol a ledge, turning at walls and edges. Stomp them from above
// or dash through them; touching one any other way is a death.

import { isSolid, TILE, tileAt, type Level } from "./level";
import { DT, PHYS, type Player } from "./physics";

export const ENEMY = { w: 0.8, h: 0.85, speed: 1.6, squashTicks: 14 } as const;

export interface Enemy {
  x: number;
  y: number;
  dir: 1 | -1;
  alive: boolean;
  /** Ticks since it was defeated (counts up to squashTicks, then it is gone). */
  deadT: number;
  /** Defeated by a stomp (flattened) or by a dash (knocked away). */
  how: "stomp" | "smash" | null;
  /** Walking phase, for leg animation. */
  walk: number;
}

export function newEnemy(x: number, y: number, dir: 1 | -1): Enemy {
  return { x, y, dir, alive: true, deadT: 0, how: null, walk: 0 };
}

const supports = (t: number) => isSolid(t) || t === TILE.oneWay;

export function stepEnemy(e: Enemy, lv: Level): void {
  if (!e.alive) {
    if (e.deadT < ENEMY.squashTicks) e.deadT++;
    return;
  }
  const half = ENEMY.w / 2;
  const front = e.x + e.dir * (half + 0.04);
  const wall = isSolid(tileAt(lv, Math.floor(front), Math.floor(e.y + 0.3)));
  const ledge = !supports(tileAt(lv, Math.floor(front), Math.floor(e.y - 0.1)));
  if (wall || ledge) e.dir = e.dir === 1 ? -1 : 1;
  e.x += e.dir * ENEMY.speed * DT;
  e.walk += ENEMY.speed * DT;
}

export type Contact = "none" | "stomp" | "smash" | "hurt";

/** How the player touches this enemy this tick. prevY is the player's height before the tick. */
export function contact(e: Enemy, p: Player, prevY: number): Contact {
  if (!e.alive) return "none";
  const hw = (ENEMY.w * 0.9) / 2 + PHYS.w / 2;
  if (Math.abs(p.x - e.x) >= hw) return "none";
  if (p.y >= e.y + ENEMY.h || p.y + PHYS.h <= e.y + 0.05) return "none";
  if (p.vy <= 0 && prevY >= e.y + ENEMY.h * 0.45) return "stomp";
  if (p.dashT > 0) return "smash";
  return "hurt";
}
