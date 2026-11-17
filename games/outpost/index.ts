// Outpost: a first-person arena shooter on a floating platform. Flying robots arrive in waves and
// fire slow orbs; the player fights back with a blaster and picks up health and ammo.
// It is a game people play in the LUCID app and the training data for its dream (docs/games.md).

import * as THREE from "three";
import { mulberry32, type Rng } from "../shared/rng";
import { type Action, type Autopilot, BUTTON, type GameOptions, type GameState, type Hud, type LucidGame } from "../shared/types";
import { type Box, groundHeight, lineBlocked, SPAWNS } from "./arena";
import { loadAssets } from "./assets";
import { OutpostAutopilot } from "./autopilot";
import { Overlay, Particles, reloadProgress, Sfx, Viewmodel } from "./effects";
import { EnemySim } from "./enemies";
import { newPickups, type PickupState, stepPickups } from "./pickups";
import { EYE, FALL_Y, newPlayer, type Player, stepPlayer, verticalState, viewDir, wrapAngle } from "./player";
import { type EyeView, World } from "./view";
import { type Blaster, newBlaster, startReload, tickBlaster, traceShot, tryFire } from "./weapons";

const EPISODE_TICKS = 120 * 30;
const GAME_OVER_TICKS = 30;
const HURT_TICKS = 10;

interface Snapshot {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
}

/** What the autopilot may read. */
export interface OutpostSim {
  player: Player;
  blaster: Blaster;
  enemies: EnemySim;
  pickups: PickupState[];
  boxes: Box[];
  alive: boolean;
  episode: number;
  tick: number;
}

export class OutpostGame implements LucidGame {
  readonly id = "outpost";
  readonly logicHz = 30;

  private mode: GameOptions["mode"] = "normal";
  private audio = false;
  private canvas!: HTMLCanvasElement;
  private renderer!: THREE.WebGLRenderer;
  private world!: World;
  private vm!: Viewmodel;
  private overlay!: Overlay;
  private particles!: Particles;
  private sfx = new Sfx();
  private boxes: Box[] = [];

  private rng: Rng = mulberry32(1);
  private fxRng: Rng = mulberry32(2);
  private enemySim = new EnemySim(mulberry32(3));
  private player: Player = newPlayer(0, 0, 0, 0);
  private prev: Snapshot = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0 };
  private blaster: Blaster = newBlaster();
  private pickups: PickupState[] = newPickups();
  private tick = 0;
  private episode = 0;
  private alive = true;
  private deadTicks = 0;
  private over = false;
  private hurt = 0;
  private trauma = 0;
  private shake = { x: 0, y: 0 };
  private lastLook: [number, number] = [0, 0];
  private flashRot = 0;
  private flashScale = 1;
  private mouseFire = false;
  private detach: (() => void) | null = null;

  async init(opts: GameOptions): Promise<void> {
    this.mode = opts.mode;
    this.audio = opts.audio && opts.mode === "normal";
    this.canvas = opts.canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas: opts.canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Fixed exposure in both modes: no tone mapping, no adaptation.
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;

    const assets = await loadAssets();
    this.world = new World(assets);
    this.boxes = this.world.arena.boxes;
    const vmMaterial = new THREE.MeshLambertMaterial({ map: assets.colormap });
    this.vm = new Viewmodel(assets, vmMaterial);
    this.overlay = new Overlay(assets);
    this.particles = new Particles(assets, this.mode === "recording");
    this.world.scene.add(this.particles.group);

    // Mouse firing comes through the shared input as action_a, so it is part of the recorded action.
    this.resize(opts.canvas.width, opts.canvas.height);
    this.reset(opts.seed);
    this.renderer.compile(this.world.scene, this.world.camera);
  }

  /** Mouse button fires in normal play (the shared input maps only keys). */
  private attachMouse(): void {
    const down = (e: MouseEvent) => {
      if (e.button === 0 && document.pointerLockElement === this.canvas) this.mouseFire = true;
    };
    const up = (e: MouseEvent) => {
      if (e.button === 0) this.mouseFire = false;
    };
    const lockChange = () => {
      if (document.pointerLockElement !== this.canvas) this.mouseFire = false;
    };
    this.canvas.addEventListener("mousedown", down);
    window.addEventListener("mouseup", up);
    document.addEventListener("pointerlockchange", lockChange);
    this.detach = () => {
      this.canvas.removeEventListener("mousedown", down);
      window.removeEventListener("mouseup", up);
      document.removeEventListener("pointerlockchange", lockChange);
    };
  }

  reset(seed: number): void {
    this.rng = mulberry32(seed);
    this.fxRng = mulberry32((seed ^ 0x5bd1e995) >>> 0);
    const r = this.rng;
    const sp = SPAWNS[Math.floor(r() * SPAWNS.length)];
    const yaw = wrapAngle(Math.atan2(-sp.x, sp.z) + (r() - 0.5) * 1.4);
    const y = groundHeight(this.boxes, sp.x, sp.z, 0.2, 20, 0);
    this.player = newPlayer(sp.x, Number.isFinite(y) ? y : 0, sp.z, yaw);
    this.enemySim.reset(mulberry32(Math.floor(r() * 4294967296)));
    this.blaster = newBlaster();
    this.pickups = newPickups();
    this.tick = 0;
    this.episode++;
    this.alive = true;
    this.deadTicks = 0;
    this.over = false;
    this.hurt = 0;
    this.trauma = 0;
    this.shake = { x: 0, y: 0 };
    this.lastLook = [0, 0];
    this.particles?.clear();
    this.snapshot();
  }

  private snapshot(): void {
    const p = this.player;
    this.prev = { x: p.x, y: p.y, z: p.z, yaw: p.yaw, pitch: p.pitch };
  }

  step(action: Action): void {
    if (this.over) return;
    if (this.audio) this.sfx.ensure();
    const normal = this.mode === "normal";
    const buttons = action.buttons | (normal && this.mouseFire ? BUTTON.actionA : 0);
    const look: [number, number] = [finite(action.look?.[0]), finite(action.look?.[1])];
    this.tick++;
    this.snapshot();
    this.lastLook = look;
    const p = this.player;
    const b = this.blaster;

    if (this.alive) {
      const pev = stepPlayer(p, buttons, look, this.boxes, true);
      if (pev.jumped) this.sound("jump", 0.5);
      if (pev.landed > 4) this.sound("land", Math.min(1, pev.landed / 9));

      if (tickBlaster(b)) this.sound("pickup", 0.35, 1.25);
      if (buttons & BUTTON.actionB && startReload(b, this.tick)) this.sound("pickup", 0.3, 0.8);
      if (buttons & BUTTON.actionA && tryFire(b, this.tick)) this.fire();

      for (const got of stepPickups(this.pickups, p, b, true)) {
        this.sound("pickup", 0.8, got.pad.kind === "health" ? 1.15 : 1);
        if (normal) this.particles.sparksAt(this.fxRng, got.pad.x, got.pad.y + 0.6, got.pad.z, 8, 2.5);
      }

      const eev = this.enemySim.step(p, this.boxes, this.tick, true);
      if (eev.fired.length) this.sound("enemyAttack", 0.35);
      for (const pop of eev.pops) this.particles.flash(pop.x, pop.y, pop.z, "pop", 0.9, 6);
      if (eev.hurtPlayer > 0) {
        p.health = Math.max(0, p.health - eev.hurtPlayer);
        this.hurt = HURT_TICKS;
        this.trauma = Math.min(1, this.trauma + 0.45);
        this.sound("enemyHurt", 0.7, 0.7);
      }

      if (p.health <= 0 || p.y < FALL_Y || this.tick >= EPISODE_TICKS) {
        this.alive = false;
        this.deadTicks = 0;
        if (p.health <= 0) this.sound("enemyDestroy", 0.6, 0.6);
      }
    } else {
      // Game-over hold: the world keeps moving, the player can't act.
      stepPlayer(p, 0, [0, 0], this.boxes, false);
      stepPickups(this.pickups, p, b, false);
      this.enemySim.step(p, this.boxes, this.tick, false);
      if (++this.deadTicks >= GAME_OVER_TICKS) this.over = true;
    }

    if (this.hurt > 0) this.hurt--;
    if (normal) {
      this.trauma = Math.max(0, this.trauma - 0.06);
      const k = this.trauma * this.trauma;
      this.shake.x = (this.fxRng() * 2 - 1) * k * 0.045;
      this.shake.y = (this.fxRng() * 2 - 1) * k * 0.045;
    }
    this.particles.step(this.boxes);
  }

  private fire(): void {
    const p = this.player;
    const normal = this.mode === "normal";
    const [dx, dy, dz] = viewDir(p.yaw, p.pitch);
    const ex = p.x;
    const ey = p.y + EYE;
    const ez = p.z;
    const shot = traceShot(this.boxes, this.enemySim.enemies, ex, ey, ez, dx, dy, dz);
    this.flashRot = normal ? this.fxRng() * Math.PI * 2 : 0;
    this.flashScale = normal ? 0.85 + this.fxRng() * 0.35 : 1;
    this.sound("blaster", 0.45, normal ? 0.95 + this.fxRng() * 0.1 : 1);
    if (normal) {
      // Tracer from roughly the muzzle (right of and below the eye) to the hit point.
      const rx = Math.cos(p.yaw);
      const rz = Math.sin(p.yaw);
      const from = new THREE.Vector3(ex + rx * 0.22 + dx * 0.5, ey - 0.16 + dy * 0.5, ez + rz * 0.22 + dz * 0.5);
      this.particles.tracerShot(from, new THREE.Vector3(shot.x, shot.y, shot.z));
    }
    // Pull the hit point slightly back towards the eye so sprites don't sink into walls.
    const hx = shot.x - dx * 0.08;
    const hy = shot.y - dy * 0.08;
    const hz = shot.z - dz * 0.08;
    if (shot.enemy >= 0) {
      const e = this.enemySim.enemies[shot.enemy];
      const killed = this.enemySim.damage(shot.enemy, 1);
      this.particles.flash(hx, hy, hz, "hit", killed ? 1.0 : 0.7, 5);
      if (normal) this.particles.sparksAt(this.fxRng, hx, hy, hz, 7, 4);
      if (killed) {
        this.particles.burst(normal ? this.fxRng : null, e.x, e.y, e.z);
        this.particles.flash(e.x, e.y, e.z, "ring", 2.2, 9);
        this.trauma = Math.min(1, this.trauma + 0.2);
        this.sound("enemyDestroy", 0.7);
      } else this.sound("enemyHurt", 0.6);
    } else if (normal) {
      this.particles.flash(hx, hy, hz, "hit", 0.35, 4);
      this.particles.sparksAt(this.fxRng, hx, hy, hz, 3, 2.5);
    }
  }

  private sound(name: Parameters<Sfx["play"]>[0], gain = 1, rate = 1): void {
    if (this.audio) this.sfx.play(name, gain, rate);
  }

  render(alpha: number): void {
    const a = this.mode === "recording" ? 1 : Math.max(0, Math.min(1, alpha));
    const p = this.player;
    const q = this.prev;
    const L = (x: number, y: number) => x + (y - x) * a;
    const dead = this.alive ? 0 : Math.min(1, (this.deadTicks + a) / GAME_OVER_TICKS);
    const ease = dead * dead * (3 - 2 * dead);
    const fell = p.y < FALL_Y + 2 || !p.grounded;
    const eye: EyeView = {
      x: L(q.x, p.x),
      y: L(q.y, p.y) + EYE - (fell ? 0 : ease * 0.6),
      z: L(q.z, p.z),
      yaw: q.yaw + wrapAngle(p.yaw - q.yaw) * a,
      pitch: L(q.pitch, p.pitch),
      roll: ease * 0.35,
    };
    // Game over: the view tips down towards the floor.
    eye.pitch = eye.pitch + (-1.05 - eye.pitch) * ease;
    if (this.mode === "normal") {
      eye.pitch += this.shake.y;
      eye.yaw += this.shake.x;
    }
    const t = (this.tick + a) / 30;
    this.world.update(eye, this.enemySim, this.pickups, t, a, (x, z, y) => groundHeight(this.boxes, x, z, 0.05, y, 0));
    this.particles.draw();

    const b = this.blaster;
    this.vm.update({
      sinceShot: this.tick - b.lastShot + a,
      reload: reloadProgress(b.reloading),
      walked: p.distance,
      moving: Math.min(1, Math.hypot(p.vx, p.vz) / 5.4),
      look: this.lastLook,
      grounded: p.grounded,
      dead: ease,
      flashRot: this.flashRot,
      flashScale: this.flashScale,
    });
    this.overlay.set(this.hurt / HURT_TICKS, ease * 0.9, this.alive);

    const r = this.renderer;
    r.autoClear = true;
    r.render(this.world.scene, this.world.camera);
    r.autoClear = false;
    r.clearDepth();
    r.render(this.vm.scene, this.vm.camera);
    r.render(this.overlay.scene, this.overlay.camera);
  }

  resize(width: number, height: number): void {
    if (!this.renderer) return;
    this.renderer.setSize(width, height, false);
    const aspect = width / Math.max(1, height);
    this.world.resize(aspect);
    this.vm.resize(aspect);
    this.overlay.resize(width, height, this.mode === "recording");
  }

  state(): GameState {
    const p = this.player;
    return {
      lane: null,
      x: p.x,
      y: p.z,
      vertical: this.alive ? verticalState(p) : "none",
      alive: this.alive,
      speed: Math.hypot(p.vx, p.vz),
      distance: p.distance,
      extra: {
        yaw: p.yaw,
        pitch: p.pitch,
        health: p.health,
        ammo: this.blaster.mag,
        enemies: this.enemySim.aliveCount(),
        wave: this.enemySim.waves.wave,
        reserve: this.blaster.reserve,
        height: p.y,
      },
    };
  }

  isOver(): boolean {
    return this.over;
  }

  hud(): Hud {
    const b = this.blaster;
    return {
      Health: String(Math.ceil(this.player.health)),
      Ammo: `${b.mag}/${b.reserve}`,
      Wave: String(Math.max(1, this.enemySim.waves.wave)),
    };
  }

  /** Read-only view of the simulation for the autopilot. */
  get sim(): OutpostSim {
    return {
      player: this.player,
      blaster: this.blaster,
      enemies: this.enemySim,
      pickups: this.pickups,
      boxes: this.boxes,
      alive: this.alive,
      episode: this.episode,
      tick: this.tick,
    };
  }

  /** True if nothing solid is between the player's eye and a point. */
  canSee(x: number, y: number, z: number): boolean {
    const p = this.player;
    return !lineBlocked(this.boxes, p.x, p.y + EYE, p.z, x, y, z);
  }

  createAutopilot(seed: number): Autopilot {
    return new OutpostAutopilot(seed);
  }

  dispose(): void {
    this.detach?.();
    this.detach = null;
    this.sfx.dispose();
    this.renderer?.dispose();
  }
}

function finite(v: number | undefined): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
}

export function createGame(): LucidGame {
  return new OutpostGame();
}
