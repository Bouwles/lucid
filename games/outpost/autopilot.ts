// Outpost's autopilot: plays like a person, from a careful player to a distracted one.
//
// Each episode it picks a profile (competent, sloppy, chaotic or idler) and jitters its settings.
// Within an episode it mostly "plays" (aim, burst fire, strafe, move between cover and pickups,
// reload, jump), broken up by stretches of random held inputs and look swings, and by idling.
// Everything comes from its own seeded generator, so the same seed and game give the same inputs.

import { cancelOpposites } from "../shared/input";
import { mulberry32, type Rng } from "../shared/rng";
import { type Action, type Autopilot, BUTTON, type LucidGame } from "../shared/types";
import { COVER, groundHeight, PADS, SPAWNS } from "./arena";
import type { OutpostGame } from "./index";
import { EYE, LOOK_RAD, wrapAngle } from "./player";
import { MAG } from "./weapons";

type Kind = "competent" | "sloppy" | "chaotic" | "idler";

interface Profile {
  kind: Kind;
  /** Fraction of the remaining aim error corrected per tick. */
  aimGain: number;
  /** Size of the wandering aim error, degrees. */
  aimNoise: number;
  /** How quickly the look output follows what it wants (0..1). */
  smooth: number;
  /** Fires when the aim error is below this, degrees. */
  fireTol: number;
  reactTicks: number;
  burst: [number, number];
  pause: [number, number];
  /** Chance of strafing (vs standing) when fighting. */
  strafe: number;
  jumpRate: number;
  randomRate: number;
  idleRate: number;
  idleLen: [number, number];
  /** Chance per decision of going for a pickup when it's needed. */
  seek: number;
  /** Chance of stopping before walking off an edge. */
  edgeCare: number;
  reloadDelay: [number, number];
}

const BASE: Record<Kind, Profile> = {
  competent: {
    kind: "competent", aimGain: 0.42, aimNoise: 1.5, smooth: 0.55, fireTol: 4, reactTicks: 6, burst: [4, 12], pause: [3, 12],
    strafe: 0.85, jumpRate: 0.012, randomRate: 0.0025, idleRate: 0.001, idleLen: [10, 40], seek: 0.95, edgeCare: 0.995,
    reloadDelay: [1, 5],
  },
  sloppy: {
    kind: "sloppy", aimGain: 0.16, aimNoise: 7, smooth: 0.3, fireTol: 14, reactTicks: 16, burst: [3, 18], pause: [6, 30],
    strafe: 0.3, jumpRate: 0.008, randomRate: 0.005, idleRate: 0.004, idleLen: [20, 80], seek: 0.35, edgeCare: 0.93,
    reloadDelay: [6, 30],
  },
  chaotic: {
    kind: "chaotic", aimGain: 0.22, aimNoise: 5, smooth: 0.4, fireTol: 10, reactTicks: 10, burst: [3, 15], pause: [4, 20],
    strafe: 0.6, jumpRate: 0.02, randomRate: 0.03, idleRate: 0.002, idleLen: [10, 40], seek: 0.6, edgeCare: 0.9,
    reloadDelay: [3, 15],
  },
  idler: {
    kind: "idler", aimGain: 0.28, aimNoise: 3.5, smooth: 0.45, fireTol: 8, reactTicks: 12, burst: [4, 12], pause: [5, 20],
    strafe: 0.5, jumpRate: 0.008, randomRate: 0.004, idleRate: 0.02, idleLen: [30, 140], seek: 0.6, edgeCare: 0.97,
    reloadDelay: [4, 20],
  },
};

const MIX: Record<Kind, number> = { competent: 0.35, sloppy: 0.35, chaotic: 0.15, idler: 0.15 };

type Segment = "play" | "random" | "idle";

interface Goal {
  x: number;
  z: number;
  kind: "wander" | "cover" | "health" | "ammo";
  ticks: number;
}

export class OutpostAutopilot implements Autopilot {
  private rng: Rng;
  private profile: Profile = BASE.competent;
  private episode = -1;
  private seg: Segment = "play";
  private segLeft = 0;
  private segButtons = 0;
  private segLook: [number, number] = [0, 0];
  private lx = 0;
  private ly = 0;
  private aimOff: [number, number] = [0, 0];
  private target = -1;
  private react = 0;
  private burstLeft = 0;
  private pauseLeft = 0;
  private strafeDir = 0;
  private strafeLeft = 0;
  private standing = false;
  private distPref = 8;
  private goal: Goal | null = null;
  private jumpLeft = 0;
  private reloadWait = -1;
  private reloadHold = 0;
  private glance = 0;
  private glanceLeft = 0;
  private wanderPitch = 0;
  private stuck = 0;
  private lastX = 0;
  private lastZ = 0;
  private episodes = 0;
  private spray = 0;
  private reckless = false;

  constructor(private seed: number) {
    this.rng = mulberry32(seed);
  }

  describe(): Record<string, unknown> {
    const { kind, ...params } = this.profile;
    return { name: "outpost-autopilot", seed: this.seed, profile: kind, params, mix: MIX };
  }

  private range(lo: number, hi: number): number {
    return lo + this.rng() * (hi - lo);
  }

  private int([lo, hi]: [number, number]): number {
    return lo + Math.floor(this.rng() * (hi - lo + 1));
  }

  private newEpisode(): void {
    let r = this.rng();
    let kind: Kind = "competent";
    for (const k of Object.keys(MIX) as Kind[]) {
      if (r < MIX[k]) {
        kind = k;
        break;
      }
      r -= MIX[k];
    }
    const b = BASE[kind];
    const j = () => this.range(0.75, 1.3);
    this.profile = {
      ...b,
      aimGain: b.aimGain * j(),
      aimNoise: b.aimNoise * j(),
      fireTol: b.fireTol * j(),
      jumpRate: b.jumpRate * j(),
      randomRate: b.randomRate * j(),
      idleRate: b.idleRate * j(),
      strafe: Math.min(1, b.strafe * j()),
    };
    this.seg = "play";
    this.segLeft = 0;
    this.lx = this.ly = 0;
    this.aimOff = [0, 0];
    this.target = -1;
    this.goal = null;
    this.jumpLeft = 0;
    this.reloadWait = -1;
    this.reloadHold = 0;
    this.burstLeft = 0;
    this.pauseLeft = 0;
    this.spray = 0;
    this.distPref = this.range(6, 11);
    this.wanderPitch = this.range(-0.25, 0.1);
    this.episodes++;
  }

  act(game: LucidGame): Action {
    const g = game as OutpostGame;
    const s = g.sim;
    if (s.episode !== this.episode) {
      this.episode = s.episode;
      this.newEpisode();
    }
    if (!s.alive) return { buttons: 0, look: [0, 0] };

    // Switch into random-input or idle stretches now and then.
    if (this.seg !== "play") {
      if (--this.segLeft <= 0) this.seg = "play";
    } else {
      const r = this.rng();
      if (r < this.profile.randomRate) this.startRandom();
      else if (r < this.profile.randomRate + this.profile.idleRate) {
        this.seg = "idle";
        this.segLeft = this.int(this.profile.idleLen);
      }
    }
    if (this.seg === "random") {
      // Most random stretches stop short of the edge; a reckless one walks right off.
      if (!this.reckless && this.nearVoid(g)) this.seg = "play";
      else return { buttons: cancelOpposites(this.segButtons), look: [...this.segLook] };
    }
    if (this.seg === "idle") {
      // Idle hands still drift the mouse a little now and then.
      const drift = this.rng() < 0.08 ? (this.rng() - 0.5) * 0.2 : 0;
      return { buttons: 0, look: [drift, 0] };
    }
    return this.play(g);
  }

  /** True if the player's current motion carries them over the void within about half a second. */
  private nearVoid(g: OutpostGame): boolean {
    const { player: p, boxes } = g.sim;
    const ahead = (t: number) => groundHeight(boxes, p.x + p.vx * t, p.z + p.vz * t, 0.05, p.y + 0.6, 0.6);
    return !Number.isFinite(ahead(0.25)) || !Number.isFinite(ahead(0.5));
  }

  private startRandom(): void {
    this.seg = "random";
    this.segLeft = this.int([6, 45]);
    const r = () => this.rng();
    this.reckless = r() < (this.profile.kind === "chaotic" ? 0.3 : 0.12);
    let b = 0;
    if (r() < 0.32) b |= BUTTON.left;
    if (r() < 0.32) b |= BUTTON.right;
    if (r() < 0.45) b |= BUTTON.up;
    if (r() < 0.28) b |= BUTTON.down;
    if (r() < 0.18) b |= BUTTON.jump;
    if (r() < 0.35) b |= BUTTON.actionA;
    if (r() < 0.2) b |= BUTTON.actionB;
    this.segButtons = b;
    const swing = (scale: number) => (r() < 0.75 ? (r() * 2 - 1) * scale : 0);
    this.segLook = [swing(1), swing(0.6)];
  }

  private play(g: OutpostGame): Action {
    const s = g.sim;
    const p = s.player;
    const P = this.profile;
    const r = () => this.rng();
    const eyeY = p.y + EYE;
    let buttons = 0;

    // Pick a target: keep the current one while it's alive and visible, else the best visible one.
    const enemies = s.enemies.enemies;
    const visible = (i: number) => enemies[i].alive && g.canSee(enemies[i].x, enemies[i].y, enemies[i].z);
    if (this.target < 0 || !visible(this.target)) {
      let best = -1;
      let bestScore = Infinity;
      for (let i = 0; i < enemies.length; i++) {
        const e = enemies[i];
        if (!e.alive) continue;
        const d = Math.hypot(e.x - p.x, e.z - p.z);
        if (d > 30 || !visible(i)) continue;
        const yawTo = Math.atan2(e.x - p.x, -(e.z - p.z));
        const score = d + Math.abs(wrapAngle(yawTo - p.yaw)) * 6;
        if (score < bestScore) {
          bestScore = score;
          best = i;
        }
      }
      if (best !== this.target) this.react = best >= 0 ? P.reactTicks + Math.floor(r() * 6) : 0;
      this.target = best;
    }
    const tgt = this.target >= 0 ? enemies[this.target] : null;

    // Wandering aim error (a slow random walk pulled back to zero).
    const n = P.aimNoise;
    this.aimOff[0] += (-this.aimOff[0] * 0.05 + (r() - 0.5) * n * 0.3);
    this.aimOff[1] += (-this.aimOff[1] * 0.05 + (r() - 0.5) * n * 0.2);

    // Goals: pickups when needed, cover when fighting, otherwise wander.
    if (this.goal) {
      this.goal.ticks--;
      if (this.goal.ticks <= 0 || Math.hypot(this.goal.x - p.x, this.goal.z - p.z) < 0.8) this.goal = null;
    }
    if (!this.goal || (this.goal.kind === "wander" && tgt && r() < 0.02)) this.goal = this.pickGoal(g, !!tgt);

    // Aim and look.
    let wantYaw: number;
    let wantPitch: number;
    let aimErr = Infinity;
    if (tgt) {
      const dx = tgt.x - p.x;
      const dz = tgt.z - p.z;
      wantYaw = Math.atan2(dx, -dz) + (this.aimOff[0] * Math.PI) / 180;
      wantPitch = Math.atan2(tgt.y - eyeY, Math.hypot(dx, dz)) + (this.aimOff[1] * Math.PI) / 180;
      const trueYaw = Math.atan2(dx, -dz);
      const truePitch = Math.atan2(tgt.y - eyeY, Math.hypot(dx, dz));
      aimErr = (Math.hypot(wrapAngle(trueYaw - p.yaw), truePitch - p.pitch) * 180) / Math.PI;
    } else {
      const gx = (this.goal?.x ?? 0) - p.x;
      const gz = (this.goal?.z ?? 0) - p.z;
      if (--this.glanceLeft <= 0) {
        this.glance = r() < 0.5 ? 0 : (r() - 0.5) * 2.6;
        this.glanceLeft = this.int([15, 60]);
        if (r() < 0.4) this.wanderPitch = this.range(-0.35, 0.25);
      }
      wantYaw = Math.atan2(gx, -gz) + this.glance;
      wantPitch = this.wanderPitch;
    }
    let tx = 0;
    let ty = 0;
    if (this.react > 0) this.react--;
    else {
      const ey = wrapAngle(wantYaw - p.yaw) / LOOK_RAD;
      const ep = (wantPitch - p.pitch) / LOOK_RAD;
      const gain = tgt ? P.aimGain : 0.18;
      tx = clamp(ey * gain);
      ty = clamp(ep * gain);
    }
    this.lx += (tx - this.lx) * P.smooth;
    this.ly += (ty - this.ly) * P.smooth;
    const look: [number, number] = [Math.abs(this.lx) < 0.004 ? 0 : this.lx, Math.abs(this.ly) < 0.004 ? 0 : this.ly];

    // Fire in bursts while the aim is close enough.
    if (tgt && this.react === 0 && aimErr < P.fireTol) {
      if (this.burstLeft > 0) {
        buttons |= BUTTON.actionA;
        this.burstLeft--;
        if (this.burstLeft === 0) this.pauseLeft = this.int(P.pause);
      } else if (this.pauseLeft > 0) this.pauseLeft--;
      else this.burstLeft = this.int(P.burst);
    } else if (!tgt && this.spray === 0 && r() < (P.kind === "competent" ? 0.002 : 0.008)) {
      // Firing at nothing: testing the gun, or a nervous trigger finger.
      this.spray = this.int([3, 10]);
    }
    if (this.spray > 0) {
      buttons |= BUTTON.actionA;
      this.spray--;
    }

    // Reload when empty (after a reaction delay); top up when it's calm.
    const b = s.blaster;
    if (this.reloadHold > 0) {
      buttons |= BUTTON.actionB;
      this.reloadHold--;
    } else if (!b.reloading && b.reserve > 0) {
      if (b.mag === 0) {
        if (this.reloadWait < 0) this.reloadWait = this.int(P.reloadDelay);
        else if (this.reloadWait-- === 0) {
          this.reloadHold = this.int([3, 9]);
          this.reloadWait = -1;
        }
      } else if (!tgt && b.mag < MAG / 2 && r() < (P.kind === "competent" ? 0.04 : 0.01)) {
        this.reloadHold = this.int([3, 8]);
      }
    }

    // Movement in the player's own frame.
    let fwd = 0;
    let side = 0;
    if (tgt) {
      const d = Math.hypot(tgt.x - p.x, tgt.z - p.z);
      if (--this.strafeLeft <= 0) {
        this.standing = r() > P.strafe;
        this.strafeDir = this.standing ? 0 : r() < 0.5 ? -1 : 1;
        this.strafeLeft = this.int([10, 40]);
      }
      side = this.strafeDir;
      if (!this.standing) {
        if (d < this.distPref - 3) fwd = -1;
        else if (d > this.distPref + 5) fwd = 1;
      }
      // Heading for a pickup or cover overrides the dance.
      if (this.goal && this.goal.kind !== "wander" && !this.standing) {
        [fwd, side] = this.towards(p.x, p.z, p.yaw, this.goal.x, this.goal.z);
      }
    } else if (this.goal) {
      const yawTo = Math.atan2(this.goal.x - p.x, -(this.goal.z - p.z));
      // Walk forward once roughly facing the goal; otherwise move towards it sideways.
      if (Math.abs(wrapAngle(yawTo - p.yaw)) < 0.6 && Math.abs(this.glance) < 0.1) fwd = 1;
      else [fwd, side] = this.towards(p.x, p.z, p.yaw, this.goal.x, this.goal.z);
    }

    // Don't walk off the edge (most of the time).
    if (fwd || side) {
      const sy = Math.sin(p.yaw);
      const cy = Math.cos(p.yaw);
      const mx = fwd * sy + side * cy;
      const mz = -fwd * cy + side * sy;
      const ml = Math.hypot(mx, mz) || 1;
      const solid = (x: number, z: number) => Number.isFinite(groundHeight(s.boxes, x, z, 0.05, p.y + 0.6, 0.6));
      // Check where the buttons point and where the player is actually sliding (along walls).
      const safe =
        solid(p.x + (mx / ml) * 0.6, p.z + (mz / ml) * 0.6) &&
        solid(p.x + (mx / ml) * 1.3, p.z + (mz / ml) * 1.3) &&
        solid(p.x + p.vx * 0.4, p.z + p.vz * 0.4);
      if (!safe && r() < P.edgeCare) {
        fwd = side = 0;
        this.strafeDir = -this.strafeDir;
        // Back off towards solid ground instead of trying the same line again.
        const c = SPAWNS[Math.floor(r() * SPAWNS.length)];
        this.goal = { x: c.x, z: c.z, kind: "wander", ticks: 60 };
      }
    }
    if (fwd > 0) buttons |= BUTTON.up;
    if (fwd < 0) buttons |= BUTTON.down;
    if (side > 0) buttons |= BUTTON.right;
    if (side < 0) buttons |= BUTTON.left;

    // Jump: now and then, and when stuck against something.
    const moved = Math.hypot(p.x - this.lastX, p.z - this.lastZ);
    this.lastX = p.x;
    this.lastZ = p.z;
    if ((fwd || side) && moved < 0.03) this.stuck++;
    else this.stuck = 0;
    if (this.jumpLeft > 0) {
      buttons |= BUTTON.jump;
      this.jumpLeft--;
    } else if (this.stuck > 8 || r() < P.jumpRate) {
      this.jumpLeft = this.int([2, 6]);
      if (this.stuck > 40) this.goal = null;
    }

    return { buttons: cancelOpposites(buttons), look };
  }

  /** Buttons (forward, side) that move from (x, z) towards (gx, gz) given the facing yaw. */
  private towards(x: number, z: number, yaw: number, gx: number, gz: number): [number, number] {
    const dx = gx - x;
    const dz = gz - z;
    const l = Math.hypot(dx, dz) || 1;
    const f = (dx * Math.sin(yaw) - dz * Math.cos(yaw)) / l;
    const sd = (dx * Math.cos(yaw) + dz * Math.sin(yaw)) / l;
    return [f > 0.38 ? 1 : f < -0.38 ? -1 : 0, sd > 0.38 ? 1 : sd < -0.38 ? -1 : 0];
  }

  private pickGoal(g: OutpostGame, fighting: boolean): Goal {
    const s = g.sim;
    const p = s.player;
    const P = this.profile;
    const r = () => this.rng();
    const ticks = this.int([120, 300]);
    const nearest = (kind: "health" | "ammo") => {
      let best: Goal | null = null;
      let bd = Infinity;
      for (const it of s.pickups) {
        if (it.wait > 0 || it.pad.kind !== kind) continue;
        const d = Math.hypot(it.pad.x - p.x, it.pad.z - p.z);
        if (d < bd) {
          bd = d;
          best = { x: it.pad.x, z: it.pad.z, kind, ticks };
        }
      }
      return best;
    };
    const needHealth = p.health < (P.kind === "competent" ? 60 : 40);
    const needAmmo = s.blaster.mag + s.blaster.reserve < 14;
    let goal: Goal | null = null;
    if (needHealth && r() < P.seek) goal = nearest("health");
    if (!goal && needAmmo && r() < P.seek) goal = nearest("ammo");
    if (!goal && fighting && r() < 0.5) {
      const c = COVER[Math.floor(r() * COVER.length)];
      goal = { x: c.x, z: c.z, kind: "cover", ticks };
    }
    if (!goal) {
      const pool = r() < 0.5 ? SPAWNS : COVER;
      const c = pool[Math.floor(r() * pool.length)];
      goal = { x: c.x + (r() - 0.5) * 3, z: c.z + (r() - 0.5) * 3, kind: "wander", ticks };
      if (r() < 0.15) {
        const pad = PADS[Math.floor(r() * PADS.length)];
        goal = { x: pad.x, z: pad.z, kind: "wander", ticks };
      }
    }
    // The lone pad at the end of the yellow walkway is reached by walking down the walkway.
    if (goal.z > 13 && p.z < 13 && Math.hypot(p.x + 10.4, p.z - 11.6) > 1.2) goal = { ...goal, x: -10.4, z: 11.6, ticks: Math.min(goal.ticks, 150) };
    return goal;
  }
}

function clamp(v: number): number {
  return Math.max(-1, Math.min(1, v));
}
