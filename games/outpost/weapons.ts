// The blaster: hold fire for automatic shots, reload by hand (or automatically on an empty trigger).

import { type Box, rayBoxes } from "./arena";
import type { Enemy } from "./enemies";

export const MAG = 12;
export const MAX_RESERVE = 60;
export const START_RESERVE = 36;
/** Ticks between shots (6 shots a second). */
const COOLDOWN = 5;
/** Ticks a reload takes. */
export const RELOAD_TICKS = 33;
const RANGE = 60;
export const ENEMY_RADIUS = 0.62;

export interface Blaster {
  mag: number;
  reserve: number;
  cooldown: number;
  /** Ticks left in the current reload, 0 when not reloading. */
  reloading: number;
  /** Tick of the last shot, for the muzzle flash and recoil. */
  lastShot: number;
  /** Tick the last reload started. */
  reloadStart: number;
}

export interface Shot {
  /** Where the shot stopped. */
  x: number;
  y: number;
  z: number;
  /** Index of the enemy it hit, or -1. */
  enemy: number;
}

export function newBlaster(): Blaster {
  return { mag: MAG, reserve: START_RESERVE, cooldown: 0, reloading: 0, lastShot: -100, reloadStart: -100 };
}

export function startReload(b: Blaster, tick: number): boolean {
  if (b.reloading || b.mag >= MAG || b.reserve <= 0) return false;
  b.reloading = RELOAD_TICKS;
  b.reloadStart = tick;
  return true;
}

/** Advances timers; finishes a reload when its time is up. */
export function tickBlaster(b: Blaster): boolean {
  if (b.cooldown > 0) b.cooldown--;
  if (b.reloading > 0 && --b.reloading === 0) {
    const n = Math.min(MAG - b.mag, b.reserve);
    b.mag += n;
    b.reserve -= n;
    return true;
  }
  return false;
}

/** True if a shot can be fired this tick (and spends the round). */
export function tryFire(b: Blaster, tick: number): boolean {
  if (b.cooldown > 0 || b.reloading > 0) return false;
  if (b.mag <= 0) {
    startReload(b, tick);
    return false;
  }
  b.mag--;
  b.cooldown = COOLDOWN;
  b.lastShot = tick;
  return true;
}

/** Hitscan from the eye along a unit direction: the nearest of a live enemy or the arena. */
export function traceShot(
  boxes: Box[],
  enemies: Enemy[],
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
): Shot {
  let best = rayBoxes(boxes, ox, oy, oz, dx, dy, dz, RANGE);
  let hit = -1;
  for (let i = 0; i < enemies.length; i++) {
    const e = enemies[i];
    if (!e.alive) continue;
    const t = raySphere(ox, oy, oz, dx, dy, dz, e.x, e.y, e.z, ENEMY_RADIUS);
    if (t >= 0 && t < best) {
      best = t;
      hit = i;
    }
  }
  return { x: ox + dx * best, y: oy + dy * best, z: oz + dz * best, enemy: hit };
}

/** Distance along a unit ray to a sphere, or -1. */
export function raySphere(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, cx: number, cy: number, cz: number, r: number): number {
  const lx = cx - ox;
  const ly = cy - oy;
  const lz = cz - oz;
  const tc = lx * dx + ly * dy + lz * dz;
  if (tc < 0) return -1;
  const d2 = lx * lx + ly * ly + lz * lz - tc * tc;
  if (d2 > r * r) return -1;
  return tc - Math.sqrt(r * r - d2);
}
