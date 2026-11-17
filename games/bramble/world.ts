// The whole game state and its fixed-tick update. No Three.js here: the world is plain data, so the
// autopilot can clone it and try moves ahead of time, and the same seed and inputs always replay
// exactly.

import { BUTTON } from "../shared/types";
import { mulberry32 } from "../shared/rng";
import { contact, ENEMY, newEnemy, stepEnemy, type Enemy } from "./enemies";
import { moverPos, parseLevel, type Level } from "./level";
import { LEVELS } from "./levels";
import { DT, movePlayer, newPlayer, PHYS, type Player, type Surface } from "./physics";

export const LOGIC_HZ = 30;
export const EPISODE_TICKS = 90 * LOGIC_HZ;
export const OVER_TICKS = 30;
/** Celebration at a flag before the next level starts. */
export const CLEAR_TICKS = 24;
/** Below this height the player has fallen off the world. */
export const KILL_Y = -2.5;

const FALLER = { shakeTicks: 16, respawnTicks: 90, id0: 100 } as const;

export const PARSED: readonly Level[] = LEVELS.map((d, i) => parseLevel(d, i));

export type Phase = "play" | "clear" | "over";
export type OverKind = "hit" | "fall" | "time" | "win";

export type GameEvent =
  | { t: "jump" | "dash" | "die" | "fall" | "flag" | "level" | "win"; x: number; y: number }
  | { t: "land"; x: number; y: number; v: number }
  | { t: "coin" | "block" | "stomp" | "smash"; x: number; y: number };

export interface Faller {
  x: number;
  top: number;
  vy: number;
  /** 0 resting, 1 shaking, 2 falling, 3 gone. */
  state: 0 | 1 | 2 | 3;
  t: number;
}

export class World {
  level: Level = PARSED[0];
  levelTick = 0;
  /** Ticks since the episode started. */
  tick = 0;
  coins = 0;
  player: Player = newPlayer(0, 0);
  enemies: Enemy[] = [];
  coinTaken = new Uint8Array(0);
  blockUsed = new Uint8Array(0);
  fallers: Faller[] = [];
  phase: Phase = "play";
  phaseT = 0;
  overKind: OverKind = "hit";
  /** Progress from levels already cleared, plus the furthest x on this one. */
  distanceBase = 0;
  maxX = 0;
  /** Filled during step() when not null; the renderer drains it. Clones used for planning leave it null. */
  events: GameEvent[] | null = [];

  reset(seed: number): void {
    const rng = mulberry32(seed ^ 0x5eed);
    this.tick = 0;
    this.coins = 0;
    this.distanceBase = 0;
    this.load(Math.floor(rng() * PARSED.length));
    this.events?.push({ t: "level", x: this.player.x, y: this.player.y });
  }

  load(index: number): void {
    const lv = PARSED[index];
    this.level = lv;
    this.levelTick = 0;
    this.player = newPlayer(lv.spawn.x, lv.spawn.y);
    // Enemies start walking towards the player's side so they meet on screen.
    this.enemies = lv.enemies.map((e) => newEnemy(e.x, e.y, -1));
    this.coinTaken = new Uint8Array(lv.coins.length);
    this.blockUsed = new Uint8Array(lv.coinBlocks.length);
    this.fallers = lv.fallers.map((f) => ({ x: f.x, top: f.y, vy: 0, state: 0, t: 0 }));
    this.phase = "play";
    this.phaseT = 0;
    this.overKind = "hit";
    this.winFloor = 0;
    this.maxX = lv.spawn.x;
  }

  get alive(): boolean {
    return this.phase !== "over";
  }

  get over(): boolean {
    return this.phase === "over" && this.phaseT >= OVER_TICKS;
  }

  get distance(): number {
    return this.distanceBase + this.maxX;
  }

  clone(): World {
    const w = new World();
    w.level = this.level;
    w.levelTick = this.levelTick;
    w.tick = this.tick;
    w.coins = this.coins;
    w.player = { ...this.player };
    w.enemies = this.enemies.map((e) => ({ ...e }));
    w.coinTaken = this.coinTaken.slice();
    w.blockUsed = this.blockUsed.slice();
    w.fallers = this.fallers.map((f) => ({ ...f }));
    w.phase = this.phase;
    w.phaseT = this.phaseT;
    w.overKind = this.overKind;
    w.distanceBase = this.distanceBase;
    w.maxX = this.maxX;
    w.winFloor = this.winFloor;
    w.events = null;
    return w;
  }

  /** Moving surfaces at the current levelTick (lifts, moving platforms, falling platforms). */
  surfaces(): Surface[] {
    const out: Surface[] = [];
    const lv = this.level;
    for (let i = 0; i < lv.movers.length; i++) {
      const m = lv.movers[i];
      const p = moverPos(m, this.levelTick);
      out.push({ id: i, x: p.x, top: p.y, w: m.w });
    }
    for (let i = 0; i < this.fallers.length; i++) {
      const f = this.fallers[i];
      if (f.state !== 3) out.push({ id: FALLER.id0 + i, x: f.x, top: f.top, w: 2 });
    }
    return out;
  }

  private emit(e: GameEvent): void {
    this.events?.push(e);
  }

  private end(kind: OverKind): void {
    const p = this.player;
    this.phase = "over";
    this.phaseT = 0;
    this.overKind = kind;
    if (kind === "hit") {
      // A little hop before tumbling off the screen.
      p.vy = 12;
      p.vx = -p.facing * 1.2;
    } else if (kind === "win") {
      p.vy = 9;
      p.vx = 0;
    } else if (kind === "time") {
      p.vx = 0;
      p.vy = Math.min(p.vy, 0);
    }
    p.dashT = 0;
    this.emit({ t: kind === "win" ? "win" : kind === "fall" ? "fall" : "die", x: p.x, y: p.y });
  }

  step(buttons: number): void {
    const p = this.player;
    if (this.phase === "over") {
      if (this.phaseT < OVER_TICKS) this.phaseT++;
      // Ballistic tumble, no collision.
      if (this.overKind !== "time") {
        // Floatier than normal play, so the tumble stays on screen for most of the hold.
        const g = this.overKind === "hit" ? PHYS.gravity * 0.55 : PHYS.gravity;
        p.vy = Math.max(p.vy - g * DT, -PHYS.maxFall);
        if (this.overKind === "win" && p.y + p.vy * DT < this.winFloor) {
          p.y = this.winFloor;
          p.vy = 9; // victory hops on the spot
        }
        p.x += p.vx * DT;
        p.y += p.vy * DT;
      }
      this.levelTick++;
      for (const e of this.enemies) stepEnemy(e, this.level);
      return;
    }

    this.tick++;
    this.levelTick++;
    const lv = this.level;

    // Platforms move first and carry whoever stands on them.
    if (p.platform >= 0 && p.platform < FALLER.id0) {
      const m = lv.movers[p.platform];
      const a = moverPos(m, this.levelTick - 1);
      const b = moverPos(m, this.levelTick);
      p.x += b.x - a.x;
      p.y += b.y - a.y;
    }
    for (let i = 0; i < this.fallers.length; i++) {
      const f = this.fallers[i];
      if (f.state === 1 && ++f.t >= FALLER.shakeTicks) {
        f.state = 2;
        f.t = 0;
      } else if (f.state === 2) {
        f.vy = Math.max(f.vy - 30 * DT, -14);
        const dy = f.vy * DT;
        f.top += dy;
        if (p.platform === FALLER.id0 + i) p.y += dy;
        if (f.top < KILL_Y - 3) {
          f.state = 3;
          f.t = 0;
        }
      } else if (f.state === 3 && ++f.t >= FALLER.respawnTicks) {
        const src = lv.fallers[i];
        Object.assign(f, { top: src.y, vy: 0, state: 0, t: 0 });
      }
    }

    const clearing = this.phase === "clear";
    const b = clearing ? 0 : buttons;
    const left = (b & BUTTON.left) !== 0;
    const right = (b & BUTTON.right) !== 0;
    const dir = left === right ? 0 : left ? -1 : 1;
    const prevY = p.y;
    const wasGrounded = p.grounded;
    const r = movePlayer(p, { dir, jump: (b & BUTTON.jump) !== 0, dash: (b & BUTTON.actionA) !== 0 }, lv, this.surfaces());

    if (r.jumped) this.emit({ t: "jump", x: p.x, y: p.y });
    if (r.dashed) this.emit({ t: "dash", x: p.x, y: p.y });
    if (r.landed && !wasGrounded) this.emit({ t: "land", x: p.x, y: p.y, v: r.landSpeed });
    if (p.platform >= FALLER.id0) {
      const f = this.fallers[p.platform - FALLER.id0];
      if (f.state === 0) {
        f.state = 1;
        f.t = 0;
      }
    }
    if (r.bump) {
      const k = lv.coinBlocks.findIndex((c) => c.x === r.bump!.x && c.y === r.bump!.y);
      if (k >= 0 && !this.blockUsed[k]) {
        this.blockUsed[k] = 1;
        this.coins++;
        this.emit({ t: "block", x: r.bump.x + 0.5, y: r.bump.y + 1 });
      }
    }

    // Enemies.
    for (const e of this.enemies) {
      stepEnemy(e, lv);
      if (clearing) continue;
      const c = contact(e, p, prevY);
      if (c === "stomp") {
        e.alive = false;
        e.how = "stomp";
        p.y = Math.max(p.y, e.y + ENEMY.h * 0.6);
        p.vy = (b & BUTTON.jump) !== 0 ? PHYS.stompVHeld : PHYS.stompV;
        p.cuttable = false;
        p.dashReady = true;
        this.emit({ t: "stomp", x: e.x, y: e.y + ENEMY.h });
      } else if (c === "smash") {
        e.alive = false;
        e.how = "smash";
        this.emit({ t: "smash", x: e.x, y: e.y + ENEMY.h / 2 });
      } else if (c === "hurt") {
        this.end("hit");
        return;
      }
    }

    // Coins: a circle of radius 0.45 against the hitbox.
    const coins = lv.coins;
    for (let i = 0; i < coins.length; i++) {
      if (this.coinTaken[i]) continue;
      const c = coins[i];
      const nx = Math.max(p.x - PHYS.w / 2, Math.min(c.x, p.x + PHYS.w / 2));
      const ny = Math.max(p.y, Math.min(c.y, p.y + PHYS.h));
      if ((nx - c.x) ** 2 + (ny - c.y) ** 2 < 0.45 * 0.45) {
        this.coinTaken[i] = 1;
        this.coins++;
        this.emit({ t: "coin", x: c.x, y: c.y });
      }
    }

    if (p.x > this.maxX) this.maxX = p.x;

    if (clearing) {
      if (++this.phaseT >= CLEAR_TICKS) {
        this.distanceBase += lv.flag.x;
        this.load(lv.index + 1);
        this.emit({ t: "level", x: this.player.x, y: this.player.y });
      }
    } else if (p.x >= lv.flag.x - 0.2 && p.y < lv.flag.y + 3) {
      if (lv.index === PARSED.length - 1) {
        this.winFloor = p.y;
        this.end("win");
      } else {
        this.phase = "clear";
        this.phaseT = 0;
        this.emit({ t: "flag", x: lv.flag.x, y: lv.flag.y });
      }
      return;
    }

    if (p.y < KILL_Y) this.end("fall");
    else if (this.tick >= EPISODE_TICKS) this.end("time");
  }

  /** Ground height for the victory hops. */
  winFloor = 0;
}
