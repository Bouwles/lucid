// First-person player: walking, strafing, jumping and mouse look, with hand-written collision.

import { BUTTON } from "../shared/types";
import { type Box, ceilingAbove, collideXZ, groundHeight } from "./arena";

export const DT = 1 / 30;
export const EYE = 1.5;
export const RADIUS = 0.35;
export const HEIGHT = 1.75;
/** Ledges up to this high are walked onto without jumping (the kit's small platforms are 0.5). */
export const STEP = 0.55;
const FOOT = 0.22;
const SPEED = 5.4;
const GRAVITY = 20;
const JUMP = 7.2;
/** look = 1 turns about 6 degrees in one tick. */
export const LOOK_RAD = (6 * Math.PI) / 180;
const MAX_PITCH = (80 * Math.PI) / 180;
const COYOTE = 3;
/** Below this the player has fallen off the platform. */
export const FALL_Y = -7;

export interface Player {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  /** Radians. 0 faces north (-z); positive turns right (towards +x). */
  yaw: number;
  /** Radians. Positive looks up. */
  pitch: number;
  grounded: boolean;
  coyote: number;
  health: number;
  distance: number;
}

export interface PlayerEvents {
  jumped: boolean;
  /** Downward speed at landing, 0 if the player did not land this tick. */
  landed: number;
}

export function newPlayer(x: number, y: number, z: number, yaw: number): Player {
  return { x, y, z, vx: 0, vy: 0, vz: 0, yaw, pitch: 0, grounded: true, coyote: 0, health: 100, distance: 0 };
}

export function wrapAngle(a: number): number {
  return a - Math.PI * 2 * Math.round(a / (Math.PI * 2));
}

/** Unit view direction from yaw and pitch. */
export function viewDir(yaw: number, pitch: number): [number, number, number] {
  const c = Math.cos(pitch);
  return [Math.sin(yaw) * c, Math.sin(pitch), -Math.cos(yaw) * c];
}

/** One tick of player movement. Mutates p. */
export function stepPlayer(p: Player, buttons: number, look: [number, number], boxes: Box[], canMove: boolean): PlayerEvents {
  const ev: PlayerEvents = { jumped: false, landed: 0 };
  if (canMove) {
    p.yaw = wrapAngle(p.yaw + clamp1(look[0]) * LOOK_RAD);
    p.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, p.pitch + clamp1(look[1]) * LOOK_RAD));
  }

  // Wanted horizontal velocity in world space.
  let f = 0;
  let s = 0;
  if (canMove) {
    if (buttons & BUTTON.up) f += 1;
    if (buttons & BUTTON.down) f -= 1;
    if (buttons & BUTTON.right) s += 1;
    if (buttons & BUTTON.left) s -= 1;
  }
  const len = Math.hypot(f, s) || 1;
  const sy = Math.sin(p.yaw);
  const cy = Math.cos(p.yaw);
  // forward = (sin yaw, -cos yaw), right = (cos yaw, sin yaw)
  const wx = ((f * sy + s * cy) / len) * SPEED;
  const wz = ((-f * cy + s * sy) / len) * SPEED;
  const k = p.grounded ? 0.3 : 0.07;
  p.vx += (wx - p.vx) * k;
  p.vz += (wz - p.vz) * k;

  const ox = p.x;
  const oz = p.z;
  p.x += p.vx * DT;
  p.z += p.vz * DT;
  collideXZ(boxes, p, p.y, RADIUS, HEIGHT, p.grounded ? STEP : 0.05);
  // Velocity follows what actually happened, so sliding along walls doesn't build up speed.
  p.vx = (p.x - ox) / DT;
  p.vz = (p.z - oz) / DT;
  if (p.grounded) p.distance += Math.hypot(p.x - ox, p.z - oz);

  const ground = groundHeight(boxes, p.x, p.z, FOOT, p.y, p.grounded ? STEP : 0.02);
  if (p.grounded) {
    if (ground >= p.y - 0.3) {
      p.y = ground;
      p.coyote = COYOTE;
    } else {
      p.grounded = false;
      p.vy = 0;
    }
  } else if (p.coyote > 0) p.coyote--;

  if (canMove && buttons & BUTTON.jump && (p.grounded || p.coyote > 0) && p.vy <= 0.01) {
    p.vy = JUMP;
    p.grounded = false;
    p.coyote = 0;
    ev.jumped = true;
  }

  if (!p.grounded) {
    p.vy -= GRAVITY * DT;
    let ny = p.y + p.vy * DT;
    if (p.vy > 0) {
      const ceil = ceilingAbove(boxes, p.x, p.z, FOOT, p.y + HEIGHT - 0.01);
      if (ny + HEIGHT > ceil) {
        ny = ceil - HEIGHT;
        p.vy = 0;
      }
    } else {
      const g = groundHeight(boxes, p.x, p.z, FOOT, p.y, 0.02);
      if (ny <= g) {
        ev.landed = -p.vy;
        ny = g;
        p.vy = 0;
        p.grounded = true;
        p.coyote = COYOTE;
      }
    }
    p.y = ny;
  }
  return ev;
}

function clamp1(v: number): number {
  return Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
}

export function verticalState(p: Player): "grounded" | "jumping" | "falling" {
  if (p.grounded) return "grounded";
  return p.vy > 0 ? "jumping" : "falling";
}
