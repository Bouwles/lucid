// Bramble: a side-scrolling platformer, one of LUCID's built-in games.
//
// The game is strictly 2D: the player, enemies and platforms only move along x and y, and all the
// rules live in world.ts / physics.ts with no rendering in them. It is drawn in 2.5D, though: the
// art is Kenney's 3D Platformer Starter Kit (CC0), so the level is built from those 3D models and
// seen side-on through a narrow-FOV perspective camera, with backdrop layers at increasing depth
// for parallax. That is a deliberate choice: the kit is 3D, and real depth and lighting make the
// world read clearly in both the normal view and LUCID's small recording view.
//
// Module map: levels.ts (level data) -> level.ts (parsing) -> physics.ts, enemies.ts, world.ts (the
// game) -> character.ts, stage.ts, backdrop.ts, camera.ts, effects.ts (the view) and autopilot.ts.

import * as THREE from "three";
import type { Action, Autopilot, GameOptions, GameState, Hud, LucidGame } from "../shared/types";
import { loadAssets, LOOK, type Assets } from "./assets";
import { BrambleAutopilot } from "./autopilot";
import { Backdrop } from "./backdrop";
import { CameraRig } from "./camera";
import { CharacterView } from "./character";
import { Particles, Sound } from "./effects";
import { Stage } from "./stage";
import { LOGIC_HZ, OVER_TICKS, PARSED, World } from "./world";

interface LevelView {
  stage: Stage;
  backdrop: Backdrop;
}

class Bramble implements LucidGame {
  readonly id = "bramble";
  readonly logicHz = LOGIC_HZ;
  readonly world = new World();

  private opts!: GameOptions;
  private assets!: Assets;
  private renderer!: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private rig!: CameraRig;
  private character!: CharacterView;
  private particles: Particles | null = null;
  private sound: Sound | null = null;
  private views = new Map<number, LevelView>();
  private view: LevelView | null = null;
  private fog = new THREE.Fog(0xffffff, 26, 125);
  private trailTick = 0;

  async init(opts: GameOptions): Promise<void> {
    this.opts = opts;
    const recording = opts.mode === "recording";
    this.assets = await loadAssets();
    this.renderer = new THREE.WebGLRenderer({ canvas: opts.canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(opts.canvas.width, opts.canvas.height, false);
    // Fixed exposure in both modes: no tone mapping or auto exposure.
    this.renderer.toneMapping = THREE.NoToneMapping;

    this.rig = new CameraRig(!recording);
    this.scene.add(this.rig.camera);
    this.scene.fog = this.fog;

    const hemi = new THREE.HemisphereLight(0xffffff, 0xb9a8e8, 1.9);
    const sun = new THREE.DirectionalLight(0xfff4e6, 1.5);
    sun.position.set(-4, 10, 7);
    this.scene.add(hemi, sun);

    this.character = new CharacterView(this.assets);
    this.scene.add(this.character.root);
    if (!recording) {
      this.particles = new Particles(this.assets);
      this.scene.add(this.particles.group);
      if (opts.audio) this.sound = new Sound();
    }
    this.resize(opts.canvas.width, opts.canvas.height);
    this.reset(opts.seed);
    // Build every level up front so a level change never stalls a frame.
    for (let i = 0; i < PARSED.length; i++) this.levelView(i);
  }

  private levelView(index: number): LevelView {
    let v = this.views.get(index);
    if (!v) {
      const level = PARSED[index];
      v = { stage: new Stage(level, this.assets), backdrop: new Backdrop(level, this.assets, this.rig.camera) };
      v.stage.group.visible = false;
      v.backdrop.group.visible = false;
      v.backdrop.sky.visible = false;
      this.scene.add(v.stage.group, v.backdrop.group);
      this.views.set(index, v);
    }
    return v;
  }

  private showLevel(): void {
    const v = this.levelView(this.world.level.index);
    if (this.view && this.view !== v) {
      this.view.stage.group.visible = false;
      this.view.backdrop.group.visible = false;
      this.view.backdrop.sky.visible = false;
    }
    this.view = v;
    v.stage.group.visible = true;
    v.backdrop.group.visible = true;
    v.backdrop.sky.visible = true;
    v.stage.reset();
    this.fog.color.copy(v.backdrop.fogColor);
    this.character.snap(this.world);
    this.rig.snap(this.world);
    this.particles?.clear();
  }

  reset(seed: number): void {
    this.world.events = [];
    this.world.reset(seed);
    this.world.events.length = 0;
    this.particles?.seed(seed);
    LOOK.uDesat.value = 0;
    LOOK.uFlash.value = 0;
    this.showLevel();
  }

  step(action: Action): void {
    if (this.sound) this.sound.start();
    const w = this.world;
    const events = (w.events ??= []);
    events.length = 0;
    w.step(action.buttons);

    let jumped = false;
    let dashed = false;
    let landSpeed = 0;
    for (const e of events) {
      switch (e.t) {
        case "level":
          this.showLevel();
          break;
        case "jump":
          jumped = true;
          this.sound?.play("jump", 0.55);
          this.particles?.dust(e.x, e.y, 2, 0.6);
          break;
        case "dash":
          dashed = true;
          this.sound?.play("jump", 0.35, 0.02);
          this.rig.addShake(0.06);
          break;
        case "land":
          landSpeed = e.v;
          if (e.v > 4) {
            this.particles?.dust(e.x, e.y, e.v > 12 ? 6 : 4);
            this.sound?.play("land", Math.min(1, 0.3 + e.v / 20));
          }
          break;
        case "coin":
          this.particles?.sparkle(e.x, e.y, 7);
          this.sound?.play("coin", 0.5, 0.04);
          break;
        case "block":
          this.particles?.sparkle(e.x, e.y + 0.4, 9);
          this.sound?.play("coin", 0.5, 0.04);
          break;
        case "stomp":
          this.particles?.dust(e.x, e.y - 0.3, 6, 1.3);
          this.rig.addShake(0.16);
          this.sound?.play("break", 0.6);
          break;
        case "smash":
          this.particles?.sparkle(e.x, e.y, 6);
          this.rig.addShake(0.14);
          this.sound?.play("break", 0.6);
          break;
        case "die":
          this.rig.addShake(0.4);
          this.sound?.play("fall", 0.7, 0);
          break;
        case "fall":
          this.sound?.play("fall", 0.7, 0);
          break;
        case "flag":
        case "win":
          this.particles?.sparkle(e.x, e.y + 1.5, 16);
          this.sound?.play("coin", 0.6, 0);
          break;
      }
    }

    const p = w.player;
    if (this.particles && p.dashT > 0 && ++this.trailTick % 2 === 0) this.particles.trail(p.x - p.facing * 0.3, p.y);
    this.character.tick(w, jumped, landSpeed, dashed);
    this.rig.tick(w);
    this.particles?.tick();
  }

  render(alpha: number): void {
    const w = this.world;
    const view = this.view;
    if (!view) return;
    const t = w.levelTick - 1 + alpha;
    this.character.render(alpha);
    this.rig.render(alpha);
    const pos = this.character.root.position;
    view.stage.update(w, t, alpha, pos.x, pos.y);
    view.backdrop.update(t);
    this.particles?.render();

    // Game-over looks: the world drains of colour (death, time up) or glows warm (the last flag).
    if (w.phase === "over") {
      const k = Math.min(1, (w.phaseT - 1 + alpha) / 10);
      if (w.overKind === "win") LOOK.uFlash.value = 0.45 * Math.min(1, w.phaseT / OVER_TICKS * 2);
      else LOOK.uDesat.value = 0.92 * Math.max(0, k);
    } else {
      LOOK.uDesat.value = 0;
      LOOK.uFlash.value = 0;
    }
    this.renderer.render(this.scene, this.rig.camera);
  }

  resize(width: number, height: number): void {
    this.renderer?.setSize(width, height, false);
    this.rig?.resize(width / Math.max(1, height));
  }

  state(): GameState {
    const w = this.world;
    const p = w.player;
    const alive = w.alive;
    return {
      lane: null,
      x: p.x,
      y: p.y,
      vertical: !alive ? "none" : p.grounded ? "grounded" : p.vy > 0 ? "jumping" : "falling",
      alive,
      speed: p.vx,
      distance: w.distance,
      extra: { level: w.level.index + 1, coins: w.coins, dashing: p.dashT > 0 ? 1 : 0 },
    };
  }

  isOver(): boolean {
    return this.world.over;
  }

  hud(): Hud {
    const s = Math.floor(this.world.tick / LOGIC_HZ);
    return {
      Level: String(this.world.level.index + 1),
      Coins: String(this.world.coins),
      Time: `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`,
    };
  }

  createAutopilot(seed: number): Autopilot {
    return new BrambleAutopilot(seed);
  }

  dispose(): void {
    this.sound?.dispose();
    this.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) mesh.geometry?.dispose();
    });
    this.renderer?.dispose();
    this.views.clear();
  }
}

export function createGame(): LucidGame {
  return new Bramble();
}
