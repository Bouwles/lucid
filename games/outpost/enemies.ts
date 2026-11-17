// Flying robots that arrive in waves, circle the player and fire slow orbs that can be dodged.

import type { Rng } from "../shared/rng";
import { type Box, lineBlocked, pointInBoxes } from "./arena";
import { DT, HEIGHT, type Player } from "./player";

export const MAX_ENEMIES = 6;
const MAX_PROJECTILES = 24;
const ENEMY_SPEED = 3.6;
const ORB_SPEED = 8.5;
const ORB_LIFE = 150;
export const ORB_DAMAGE = 9;
const ORB_HIT_RADIUS = 0.42;
/** Ticks of warning glow before a robot fires. */
export const CHARGE_TICKS = 12;

export interface Enemy {
  alive: boolean;
  x: number;
  y: number;
  z: number;
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  hp: number;
  fireCd: number;
  orbit: number;
  orbitRate: number;
  orbitDist: number;
  alt: number;
  phase: number;
  /** Ticks left of the white hit flash. */
  flash: number;
  /** Tick it spawned, for the arrival animation. */
  born: number;
}

export interface Orb {
  alive: boolean;
  x: number;
  y: number;
  z: number;
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  life: number;
}

export interface Waves {
  wave: number;
  /** Robots of this wave not yet spawned. */
  toSpawn: number;
  spawnCd: number;
  /** Ticks of calm before the next wave. */
  rest: number;
  kills: number;
}

export interface EnemyEvents {
  fired: { x: number; y: number; z: number }[];
  hurtPlayer: number;
  /** Orbs that burst against the arena or the player: positions. */
  pops: { x: number; y: number; z: number }[];
  waveStarted: boolean;
}

function emptyEnemy(): Enemy {
  return {
    alive: false, x: 0, y: 0, z: 0, px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0, hp: 0, fireCd: 0,
    orbit: 0, orbitRate: 0, orbitDist: 0, alt: 0, phase: 0, flash: 0, born: 0,
  };
}

function emptyOrb(): Orb {
  return { alive: false, x: 0, y: 0, z: 0, px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0, life: 0 };
}

export class EnemySim {
  enemies: Enemy[] = Array.from({ length: MAX_ENEMIES }, emptyEnemy);
  orbs: Orb[] = Array.from({ length: MAX_PROJECTILES }, emptyOrb);
  waves: Waves = { wave: 0, toSpawn: 0, spawnCd: 0, rest: 45, kills: 0 };

  constructor(private rng: Rng) {}

  reset(rng: Rng): void {
    this.rng = rng;
    for (const e of this.enemies) e.alive = false;
    for (const o of this.orbs) o.alive = false;
    this.waves = { wave: 0, toSpawn: 0, spawnCd: 0, rest: 45, kills: 0 };
  }

  aliveCount(): number {
    let n = 0;
    for (const e of this.enemies) if (e.alive) n++;
    return n;
  }

  private waveSize(w: number): number {
    return Math.min(2 + w, 10);
  }

  private maxAlive(w: number): number {
    return Math.min(2 + Math.ceil(w / 2), MAX_ENEMIES);
  }

  private spawn(tick: number): void {
    const e = this.enemies.find((x) => !x.alive);
    if (!e) return;
    const r = this.rng;
    const a = r() * Math.PI * 2;
    const w = this.waves.wave;
    Object.assign(e, emptyEnemy(), {
      alive: true,
      x: Math.cos(a) * 24,
      y: 6 + r() * 4,
      z: Math.sin(a) * 24,
      hp: 3 + Math.floor((w - 1) / 3),
      fireCd: 50 + Math.floor(r() * 50),
      orbit: a,
      orbitRate: (r() < 0.5 ? -1 : 1) * (0.15 + r() * 0.25),
      orbitDist: 6 + r() * 6,
      alt: 2.4 + r() * 2.8,
      phase: r() * Math.PI * 2,
      born: tick,
    });
    e.px = e.x;
    e.py = e.y;
    e.pz = e.z;
  }

  /** Damage an enemy; returns true if it was destroyed. */
  damage(i: number, amount: number): boolean {
    const e = this.enemies[i];
    e.hp -= amount;
    e.flash = 3;
    if (e.hp <= 0) {
      e.alive = false;
      this.waves.kills++;
      return true;
    }
    return false;
  }

  step(p: Player, boxes: Box[], tick: number, canAttack: boolean): EnemyEvents {
    const ev: EnemyEvents = { fired: [], hurtPlayer: 0, pops: [], waveStarted: false };
    const r = this.rng;
    const W = this.waves;

    // Waves: rest, then spawn robots one at a time up to the live cap.
    if (W.toSpawn === 0 && this.aliveCount() === 0) {
      if (W.rest > 0) W.rest--;
      else {
        W.wave++;
        W.toSpawn = this.waveSize(W.wave);
        W.spawnCd = 0;
        ev.waveStarted = true;
      }
    }
    if (W.toSpawn > 0) {
      if (W.spawnCd > 0) W.spawnCd--;
      else if (this.aliveCount() < this.maxAlive(W.wave)) {
        this.spawn(tick);
        W.toSpawn--;
        W.spawnCd = 40;
        if (W.toSpawn === 0) W.rest = 75;
      }
    }

    const chestY = p.y + 1.1;
    for (const e of this.enemies) {
      e.px = e.x;
      e.py = e.y;
      e.pz = e.z;
      if (!e.alive) continue;
      if (e.flash > 0) e.flash--;
      e.orbit += e.orbitRate * DT;
      const t = tick * DT;
      const tx = p.x + Math.cos(e.orbit) * e.orbitDist;
      const tz = p.z + Math.sin(e.orbit) * e.orbitDist;
      const ty = Math.max(p.y, 0) + e.alt + Math.sin(t * 2.2 + e.phase) * 0.35;
      let ax = (tx - e.x) * 1.2 - e.vx * 1.4;
      let ay = (ty - e.y) * 1.6 - e.vy * 1.8;
      let az = (tz - e.z) * 1.2 - e.vz * 1.4;
      // Keep apart from each other.
      for (const o of this.enemies) {
        if (o === e || !o.alive) continue;
        const dx = e.x - o.x;
        const dy = e.y - o.y;
        const dz = e.z - o.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 4 && d2 > 1e-6) {
          const k = (4 - d2) * 2;
          ax += dx * k;
          ay += dy * k * 0.5;
          az += dz * k;
        }
      }
      e.vx += ax * DT;
      e.vy += ay * DT;
      e.vz += az * DT;
      const sp = Math.hypot(e.vx, e.vy, e.vz);
      if (sp > ENEMY_SPEED) {
        e.vx *= ENEMY_SPEED / sp;
        e.vy *= ENEMY_SPEED / sp;
        e.vz *= ENEMY_SPEED / sp;
      }
      e.x += e.vx * DT;
      e.y += e.vy * DT;
      e.z += e.vz * DT;
      // Stay above the deck and out of solid walls.
      if (e.y < 1.6) {
        e.y = 1.6;
        e.vy = Math.max(e.vy, 0);
      }
      if (pointInBoxes(boxes, e.x, e.y, e.z)) {
        e.y += 0.15;
        e.vy = Math.max(e.vy, 1);
      }

      // Shooting: count down, glow for CHARGE_TICKS, then fire if the player is in sight.
      if (e.fireCd > 0) e.fireCd--;
      if (e.fireCd === 0) {
        const dx = p.x - e.x;
        const dy = chestY - e.y;
        const dz = p.z - e.z;
        const d = Math.hypot(dx, dy, dz);
        const born = tick - e.born > 45;
        if (canAttack && born && d < 26 && d > 0.5 && !lineBlocked(boxes, e.x, e.y, e.z, p.x, chestY, p.z)) {
          const o = this.orbs.find((x) => !x.alive);
          if (o) {
            Object.assign(o, {
              alive: true, x: e.x + (dx / d) * 0.5, y: e.y + (dy / d) * 0.5 - 0.1, z: e.z + (dz / d) * 0.5,
              vx: (dx / d) * ORB_SPEED, vy: (dy / d) * ORB_SPEED, vz: (dz / d) * ORB_SPEED, life: ORB_LIFE,
            });
            o.px = o.x;
            o.py = o.y;
            o.pz = o.z;
            ev.fired.push({ x: e.x, y: e.y, z: e.z });
          }
          e.fireCd = Math.max(40, 70 + Math.floor(r() * 50) - W.wave * 4);
        } else {
          // Not in sight: try again soon, without the long cooldown.
          e.fireCd = 8 + Math.floor(r() * 10);
        }
      }
    }

    for (const o of this.orbs) {
      o.px = o.x;
      o.py = o.y;
      o.pz = o.z;
      if (!o.alive) continue;
      o.x += o.vx * DT;
      o.y += o.vy * DT;
      o.z += o.vz * DT;
      if (--o.life <= 0) {
        o.alive = false;
        continue;
      }
      // Distance from the orb to the player's body (a vertical segment).
      const cy = Math.max(p.y + 0.3, Math.min(p.y + HEIGHT - 0.2, o.y));
      if (canAttack && Math.hypot(o.x - p.x, o.y - cy, o.z - p.z) < ORB_HIT_RADIUS) {
        o.alive = false;
        ev.hurtPlayer += ORB_DAMAGE;
        ev.pops.push({ x: o.x, y: o.y, z: o.z });
        continue;
      }
      if (pointInBoxes(boxes, o.x, o.y, o.z)) {
        o.alive = false;
        ev.pops.push({ x: o.x, y: o.y, z: o.z });
      }
    }
    return ev;
  }
}

/** Ticks until this robot fires, for the warning glow (Infinity if it won't soon). */
export function chargeLevel(e: Enemy): number {
  return e.fireCd > 0 && e.fireCd <= CHARGE_TICKS ? 1 - e.fireCd / CHARGE_TICKS : 0;
}
