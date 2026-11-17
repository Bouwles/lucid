// Player movement and tile collision. Fixed 30 Hz tick, units are tiles and seconds.
//
// Collision is axis by axis: move along x and push out of walls, then along y and land on (or bump
// into) whatever was crossed. No single tick moves more than about 0.7 tiles, so nothing tunnels.

import { isSolid, TILE, tileAt, type Level } from "./level";

export const DT = 1 / 30;

export const PHYS = {
  /** Hitbox, centred on x, standing on y. */
  w: 0.6,
  h: 0.9,
  runMax: 6.6,
  groundAccel: 70,
  groundDecel: 60,
  turnAccel: 110,
  airAccel: 42,
  airDecel: 10,
  /** Slows a dash's leftover speed back to running speed. */
  overspeedDecel: 24,
  gravity: 52,
  /** Heavier on the way down, so jumps feel snappy rather than floaty. */
  fallMult: 1.35,
  // With the 30 Hz integration this peaks at about 2.4 tiles: enough to climb a 2-tile wall.
  jumpV: 16.8,
  /** Releasing jump while rising multiplies the upward speed by this (variable jump height). */
  jumpCut: 0.42,
  maxFall: 20,
  /** Ticks after leaving a ledge during which a jump still works. */
  coyote: 5,
  /** Ticks a jump press is remembered before landing. */
  buffer: 5,
  dashSpeed: 15,
  dashTicks: 7,
  dashCooldown: 16,
  stompV: 11.5,
  stompVHeld: 15,
} as const;

export interface Player {
  x: number;
  y: number;
  vx: number;
  vy: number;
  facing: 1 | -1;
  grounded: boolean;
  /** The moving or falling platform stood on, or -1. */
  platform: number;
  coyote: number;
  buffer: number;
  jumpPrev: boolean;
  /** Rising from a jump that can still be cut short by releasing the button. */
  cuttable: boolean;
  dashT: number;
  dashCd: number;
  /** One dash per touch of the ground. */
  dashReady: boolean;
  /** Ticks since leaving the ground (0 while grounded). */
  air: number;
}

export function newPlayer(x: number, y: number): Player {
  return {
    x, y, vx: 0, vy: 0, facing: 1, grounded: true, platform: -1, coyote: 0, buffer: 0,
    jumpPrev: false, cuttable: false, dashT: 0, dashCd: 0, dashReady: true, air: 0,
  };
}

/** A one-way platform surface that moves (lifts, moving and falling platforms). */
export interface Surface {
  id: number;
  x: number;
  top: number;
  w: number;
}

export interface Input {
  dir: -1 | 0 | 1;
  jump: boolean;
  dash: boolean;
}

/** What happened during one tick of movement, for sound, effects and game rules. */
export interface MoveResult {
  jumped: boolean;
  dashed: boolean;
  landed: boolean;
  /** Downward speed at the moment of landing. */
  landSpeed: number;
  /** Tile bumped from below, or null. */
  bump: { x: number; y: number } | null;
}

const EPS = 1e-4;

function approach(v: number, target: number, step: number): number {
  return v < target ? Math.min(v + step, target) : Math.max(v - step, target);
}

export function movePlayer(p: Player, inp: Input, lv: Level, surfaces: readonly Surface[]): MoveResult {
  const res: MoveResult = { jumped: false, dashed: false, landed: false, landSpeed: 0, bump: null };
  const wasGrounded = p.grounded;

  if (p.grounded) p.dashReady = true;
  if (p.dashCd > 0) p.dashCd--;
  const pressed = inp.jump && !p.jumpPrev;
  p.buffer = pressed ? PHYS.buffer : Math.max(0, p.buffer - 1);
  p.jumpPrev = inp.jump;
  if (inp.dir !== 0 && p.dashT === 0) p.facing = inp.dir;

  // Dash: a short burst in the facing direction that ignores gravity. Holding the button dashes
  // again as soon as it's allowed, so the button always does something while held.
  if (inp.dash && p.dashT === 0 && p.dashCd === 0 && p.dashReady) {
    p.dashT = PHYS.dashTicks;
    p.dashCd = PHYS.dashCooldown;
    p.dashReady = p.grounded;
    res.dashed = true;
  }

  const canJump = p.grounded || p.coyote > 0;
  if ((inp.jump || p.buffer > 0) && canJump) {
    // Holding jump on the ground jumps again on landing: holding a key always has an effect.
    p.vy = PHYS.jumpV;
    p.grounded = false;
    p.coyote = 0;
    p.buffer = 0;
    p.cuttable = true;
    p.platform = -1;
    if (p.dashT > 0) p.dashT = 0; // jumping out of a dash keeps its speed: a long jump
    res.jumped = true;
  }

  if (p.dashT > 0) {
    p.dashT--;
    p.vx = p.facing * PHYS.dashSpeed;
    p.vy = p.grounded ? -1 : 0;
  } else {
    const target = inp.dir * PHYS.runMax;
    let accel: number;
    if (inp.dir === 0) accel = p.grounded ? PHYS.groundDecel : PHYS.airDecel;
    else if (p.vx !== 0 && Math.sign(p.vx) !== inp.dir) accel = PHYS.turnAccel;
    else if (Math.abs(p.vx) > PHYS.runMax) accel = PHYS.overspeedDecel;
    else accel = p.grounded ? PHYS.groundAccel : PHYS.airAccel;
    p.vx = approach(p.vx, target, accel * DT);

    if (p.cuttable && p.vy > 0 && !inp.jump) {
      p.vy *= PHYS.jumpCut;
      p.cuttable = false;
    }
    if (p.vy <= 0) p.cuttable = false;
    p.vy -= PHYS.gravity * (p.vy < 0 ? PHYS.fallMult : 1) * DT;
    if (p.vy < -PHYS.maxFall) p.vy = -PHYS.maxFall;
  }

  moveX(p, lv);
  moveY(p, lv, surfaces, res);

  // moveY lands the player every tick they stand on something; only report the first.
  res.landed = res.landed && !wasGrounded;
  p.coyote = p.grounded ? PHYS.coyote : Math.max(0, p.coyote - 1);
  p.air = p.grounded ? 0 : p.air + 1;
  return res;
}

function rows(p: Player): [number, number] {
  return [Math.floor(p.y + EPS), Math.floor(p.y + PHYS.h - EPS)];
}

function moveX(p: Player, lv: Level): void {
  if (p.vx === 0) return;
  p.x += p.vx * DT;
  const half = PHYS.w / 2;
  const [r0, r1] = rows(p);
  if (p.vx > 0) {
    const cx = Math.floor(p.x + half - EPS);
    for (let cy = r0; cy <= r1; cy++) {
      if (isSolid(tileAt(lv, cx, cy))) {
        p.x = cx - half;
        p.vx = 0;
        return;
      }
    }
  } else {
    const cx = Math.floor(p.x - half + EPS);
    for (let cy = r0; cy <= r1; cy++) {
      if (isSolid(tileAt(lv, cx, cy))) {
        p.x = cx + 1 + half;
        p.vx = 0;
        return;
      }
    }
  }
}

function moveY(p: Player, lv: Level, surfaces: readonly Surface[], res: MoveResult): void {
  const prevY = p.y;
  const vy = p.vy;
  p.y += vy * DT;
  const half = PHYS.w / 2;
  const c0 = Math.floor(p.x - half + EPS);
  const c1 = Math.floor(p.x + half - EPS);
  p.grounded = false;
  p.platform = -1;

  if (vy <= 0) {
    let best = -Infinity;
    let bestId = -1;
    // Tile tops: the only integer height that can have been crossed this tick.
    const top = Math.floor(prevY + EPS);
    if (top >= p.y - EPS && top <= prevY + EPS) {
      for (let cx = c0; cx <= c1; cx++) {
        const t = tileAt(lv, cx, top - 1);
        if (isSolid(t) || t === TILE.oneWay) {
          best = top;
          break;
        }
      }
    }
    for (const s of surfaces) {
      if (p.x + half <= s.x || p.x - half >= s.x + s.w) continue;
      if (prevY >= s.top - 0.06 && p.y <= s.top + EPS && s.top > best) {
        best = s.top;
        bestId = s.id;
      }
    }
    if (best > -Infinity) {
      p.y = best;
      res.landed = true;
      res.landSpeed = -vy;
      p.vy = 0;
      p.grounded = true;
      p.platform = bestId;
    }
  } else {
    const row = Math.floor(p.y + PHYS.h - EPS);
    const prevRow = Math.floor(prevY + PHYS.h - EPS);
    if (row !== prevRow) {
      // Bump the blocking tile nearest the player's centre.
      let hit = -1;
      let hitDist = Infinity;
      for (let cx = c0; cx <= c1; cx++) {
        if (isSolid(tileAt(lv, cx, row))) {
          const d = Math.abs(cx + 0.5 - p.x);
          if (d < hitDist) {
            hitDist = d;
            hit = cx;
          }
        }
      }
      if (hit >= 0) {
        // Corner correction: clipping a corner by a sliver nudges the player past it instead.
        const overlapL = p.x + half - hit;
        const overlapR = hit + 1 - (p.x - half);
        if (c0 !== c1 && overlapL < 0.22 && !isSolid(tileAt(lv, hit - 1, row))) {
          p.x = hit - half;
        } else if (c0 !== c1 && overlapR < 0.22 && !isSolid(tileAt(lv, hit + 1, row))) {
          p.x = hit + 1 + half;
        } else {
          p.y = row - PHYS.h;
          p.vy = 0;
          p.cuttable = false;
          res.bump = { x: hit, y: row };
        }
      }
    }
  }
}
