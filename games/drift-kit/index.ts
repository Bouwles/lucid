// Drift Kit: a low-poly arcade time trial on a Kenney track. See docs/games.md for the contract.
//
// Normal mode: two-lap time trial with a best-lap ghost and three AI trucks on the racing line,
// tyre smoke, camera shake and sound.
// Recording mode: the same track and car, time trial alone. No AI cars and no ghost (fewer moving
// things for the dream to model), no particles or shake, one fixed camera rig, square canvas.

import * as THREE from "three";
import { mulberry32 } from "../shared/rng";
import { BUTTON, type Action, type Autopilot, type GameOptions, type GameState, type Hud, type LucidGame } from "../shared/types";
import { createAutopilot } from "./autopilot";
import { ChaseCamera } from "./camera";
import { Car, ObstacleGrid, type CarSnapshot } from "./car";
import { CarAudio, Smoke, Wash } from "./effects";
import { OFF_TRACK_LAT, ROAD_HALF, Track, angleDiff } from "./track";
import { PALETTE, buildWorld, loadAssets, placeSun, type Assets } from "./world";

const HZ = 30;
const DT = 1 / HZ;
export const LAPS = 2;
const TIME_LIMIT = 90 * HZ;
const OFF_LIMIT = 3 * HZ;
const STUCK_LIMIT = 3 * HZ;
const OVER_TICKS = 30;
/** The finish arch stands in the middle of the first tile. */
const LINE_S = 5;

type EndReason = "finish" | "crash" | "stuck" | "timeout";

interface CarView {
  root: THREE.Object3D;
  body: THREE.Object3D | null;
  wheels: THREE.Object3D[];
  front: THREE.Object3D[];
}

function makeCarView(src: THREE.Object3D, material?: THREE.Material): CarView {
  const root = src.clone(true);
  const wheels: THREE.Object3D[] = [];
  const front: THREE.Object3D[] = [];
  let body: THREE.Object3D | null = null;
  root.traverse((o) => {
    if (o.name === "body") body = o;
    if (o.name.startsWith("wheel-")) {
      o.rotation.order = "YXZ";
      wheels.push(o);
      if (o.name.startsWith("wheel-front")) front.push(o);
    }
    const m = o as THREE.Mesh;
    if (m.isMesh && material) {
      m.material = material;
      m.castShadow = false;
    }
  });
  return { root, body, wheels, front };
}

function poseView(v: CarView, a: CarSnapshot, b: CarSnapshot, t: number, spin: boolean): void {
  v.root.position.set(a.x + (b.x - a.x) * t, 0, a.z + (b.z - a.z) * t);
  v.root.rotation.y = a.heading + angleDiff(b.heading, a.heading) * t;
  if (v.body) {
    v.body.rotation.z = a.lean + (b.lean - a.lean) * t;
    v.body.rotation.x = a.pitch + (b.pitch - a.pitch) * t;
  }
  const steer = (a.steer + (b.steer - a.steer) * t) * 0.45;
  const roll = spin ? a.wheelSpin + (b.wheelSpin - a.wheelSpin) * t : 0;
  for (const w of v.wheels) w.rotation.x = roll;
  for (const w of v.front) w.rotation.y = steer;
}

/** An AI truck: kinematic, it rides the racing line with its own pace and lane offset. */
interface Rival {
  s: number;
  v: number;
  lane: number;
  pace: number;
  prev: CarSnapshot;
  cur: CarSnapshot;
  view: CarView;
}

function fmtTime(ticks: number): string {
  const t = ticks / HZ;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s < 10 ? "0" : ""}${s.toFixed(1)}`;
}

export class DriftKit implements LucidGame {
  readonly id = "drift-kit";
  readonly logicHz = HZ;

  readonly track = new Track();
  readonly car = new Car();
  private opts!: GameOptions;
  private recording = false;
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera = new THREE.PerspectiveCamera(60, 1, 0.1, 260);
  private sun!: THREE.DirectionalLight;
  private obstacles!: ObstacleGrid;
  private assets!: Assets;
  private chase!: ChaseCamera;
  private view!: CarView;
  private prevSnap!: CarSnapshot;
  private curSnap!: CarSnapshot;

  // Episode state (all of it is reset from the seed).
  /** Ticks since reset. */
  tick = 0;
  /** Metres along the track from the finish line, cumulative over laps (negative before the line). */
  progress = 0;
  private startProgress = 0;
  /** Lateral offset from the centreline (m, positive = left) and the projected track distance. */
  lat = 0;
  private lastS = 0;
  lapsDone = 0;
  private lapStart = 0;
  offTimer = 0;
  stuckTimer = 0;
  private moved = false;
  /** -1 while racing; then counts the game-over ticks. */
  overTicks = -1;
  reason: EndReason = "finish";
  lastButtons = 0;

  // Kept across episodes: the best lap and its ghost (normal mode only).
  private bestLap = 0;
  private lastLap = 0;
  private ghostBest: Float32Array | null = null;
  private ghostLap: number[] = [];
  private ghost: CarView | null = null;

  private rivals: Rival[] = [];
  private smoke: Smoke | null = null;
  private wash = new Wash();
  private audio: CarAudio | null = null;
  private rivalCircles: number[] = [];

  async init(opts: GameOptions): Promise<void> {
    this.opts = opts;
    this.recording = opts.mode === "recording";
    const r = new THREE.WebGLRenderer({ canvas: opts.canvas, antialias: true, powerPreference: "high-performance" });
    r.setPixelRatio(1);
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    // Fixed exposure in both modes: nothing adapts to the scene.
    r.toneMapping = THREE.NeutralToneMapping;
    r.toneMappingExposure = 1.0;
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = r;

    this.assets = await loadAssets();
    const world = buildWorld(this.track, this.assets, this.recording);
    this.scene = world.scene;
    this.sun = world.sun;
    this.obstacles = new ObstacleGrid(world.obstacles);
    this.chase = new ChaseCamera(this.recording);

    this.view = makeCarView(this.assets.cars.yellow);
    this.scene.add(this.view.root);

    if (!this.recording) {
      const ghostMat = new THREE.MeshStandardMaterial({
        map: this.assets.material.map,
        color: new THREE.Color("#dff4ff"),
        transparent: true,
        opacity: 0.45,
        depthWrite: false,
        roughness: 0.6,
      });
      this.ghost = makeCarView(this.assets.cars.yellow, ghostMat);
      this.ghost.root.visible = false;
      this.scene.add(this.ghost.root);
      const models = [this.assets.cars.red, this.assets.cars.green, this.assets.cars.purple];
      for (const m of models) {
        const view = makeCarView(m);
        this.scene.add(view.root);
        const snap: CarSnapshot = { x: 0, z: 0, heading: 0, steer: 0, wheelSpin: 0, lean: 0, pitch: 0 };
        this.rivals.push({ s: 0, v: 0, lane: 0, pace: 1, prev: snap, cur: { ...snap }, view });
      }
      this.smoke = new Smoke();
      this.scene.add(this.smoke.mesh);
      if (opts.audio) this.audio = new CarAudio();
    }
    this.camera.add(this.wash.mesh);
    this.scene.add(this.camera);

    this.resize(opts.canvas.width || 256, opts.canvas.height || 256);
    this.reset(opts.seed);
    // Compile shaders up front so the first frames don't stall.
    r.compile(this.scene, this.camera);
  }

  reset(seed: number): void {
    const rng = mulberry32(seed);
    // Start a few metres behind the line (the arch ahead), jittered a little along and across
    // the track and in heading. The first lap includes this short run-up.
    const along = LINE_S - 7 + rng() * 1.5;
    const across = (rng() - 0.5) * 2.0;
    const yaw = (rng() - 0.5) * 0.08;
    const p = this.track.pose(along);
    this.car.place(p.x + p.tz * across, p.z - p.tx * across, Math.atan2(p.tx, p.tz) + yaw);
    const proj = this.track.project(this.car.x, this.car.z);
    this.lastS = proj.s;
    this.lat = proj.lat;
    this.progress = along - LINE_S;
    this.startProgress = this.progress;
    this.tick = 0;
    this.lapsDone = 0;
    this.lapStart = 0;
    this.offTimer = 0;
    this.stuckTimer = 0;
    this.moved = false;
    this.overTicks = -1;
    this.reason = "finish";
    this.lastButtons = 0;
    this.ghostLap = [];
    this.curSnap = this.car.snapshot();
    this.prevSnap = { ...this.curSnap };
    this.chase.reset(this.car);

    // Rivals line up ahead of the player, each with its own pace and lane.
    const lanes = [1.4, -1.4, 0.4];
    this.rivals.forEach((rv, k) => {
      rv.s = along + 7 + k * 6 + rng() * 2;
      rv.v = 0;
      rv.lane = lanes[k] + (rng() - 0.5) * 0.6;
      rv.pace = 0.86 + k * 0.035 + rng() * 0.03;
      this.poseRival(rv);
      rv.prev = { ...rv.cur };
    });
    this.smoke?.clear();
    this.wash.set(PALETTE.finishWash, 0);
  }

  private poseRival(rv: Rival): void {
    const a = this.track.linePoint(rv.s, rv.lane);
    const b = this.track.linePoint(rv.s + 1.5, rv.lane);
    const heading = Math.atan2(b.x - a.x, b.z - a.z);
    const steer = Math.max(-1, Math.min(1, angleDiff(heading, rv.cur.heading) * 6));
    rv.cur = { x: a.x, z: a.z, heading, steer, wheelSpin: rv.cur.wheelSpin + (rv.v * DT) / 0.3, lean: -steer * 0.03, pitch: 0 };
  }

  step(action: Action): void {
    const b = action.buttons;
    const racing = this.overTicks < 0;
    this.lastButtons = racing ? b : 0;
    const controls = racing
      ? { left: !!(b & BUTTON.left), right: !!(b & BUTTON.right), up: !!(b & BUTTON.up), down: !!(b & BUTTON.down), drift: !!(b & BUTTON.actionA) }
      : { left: false, right: false, up: false, down: false, drift: false };

    this.prevSnap = this.curSnap;

    // Rivals move first; the player then collides with where they are now.
    this.rivalCircles.length = 0;
    for (const rv of this.rivals) {
      rv.prev = rv.cur;
      const ahead = this.track.lineSpeed[this.track.lineIndex(rv.s + rv.v * 0.4)] * rv.pace;
      let target = ahead;
      // Don't drive through the rival in front.
      for (const o of this.rivals) {
        if (o === rv) continue;
        let gap = (o.s - rv.s) % this.track.length;
        if (gap < 0) gap += this.track.length;
        if (gap < 4.5 && Math.abs(o.lane - rv.lane) < 1.6) target = Math.min(target, o.v * 0.95);
      }
      rv.v += Math.max(-14 * DT, Math.min(6.5 * DT, target - rv.v));
      rv.s += rv.v * DT;
      this.poseRival(rv);
      const sh = Math.sin(rv.cur.heading);
      const ch = Math.cos(rv.cur.heading);
      this.rivalCircles.push(rv.cur.x + sh * 0.7, rv.cur.z + ch * 0.7, 0.8, rv.cur.x - sh * 0.7, rv.cur.z - ch * 0.7, 0.8);
    }

    // Grass under the car slows it and loosens its grip.
    this.car.grass = Math.max(0, Math.min(1, (Math.abs(this.lat) - ROAD_HALF) / 0.6));
    if (!racing) {
      // Game over: roll to a stop (after a finish, roll on clear of the arch).
      const roll = this.reason === "finish" ? 0.96 : 0.9;
      this.car.vx *= roll;
      this.car.vz *= roll;
    }
    this.car.step(controls, DT, this.obstacles, this.rivalCircles);
    if (this.car.impact > 0) {
      this.chase.kick(Math.min(0.5, this.car.impact * 0.05));
      // Rivals that got hit lose a little speed.
      for (const rv of this.rivals) {
        if (Math.hypot(rv.cur.x - this.car.x, rv.cur.z - this.car.z) < 2.6) rv.v *= 0.9;
      }
    }
    this.curSnap = this.car.snapshot();

    // Progress along the track. Clamp per-tick jumps so cutting across the infield gains nothing.
    const proj = this.track.project(this.car.x, this.car.z);
    let ds = proj.s - this.lastS;
    if (ds > this.track.length / 2) ds -= this.track.length;
    if (ds < -this.track.length / 2) ds += this.track.length;
    this.progress += Math.max(-2, Math.min(2, ds));
    this.lastS = proj.s;
    this.lat = proj.lat;
    this.tick++;

    if (racing) {
      if (this.ghost) this.ghostLap.push(this.car.x, this.car.z, this.car.heading);
      const laps = Math.max(0, Math.floor(this.progress / this.track.length));
      if (laps > this.lapsDone) {
        const lapTicks = this.tick - this.lapStart;
        this.lastLap = lapTicks;
        if (this.bestLap === 0 || lapTicks < this.bestLap) {
          this.bestLap = lapTicks;
          if (this.ghost) this.ghostBest = Float32Array.from(this.ghostLap);
        }
        this.lapsDone = laps;
        this.lapStart = this.tick;
        this.ghostLap = [];
      }
      const off = Math.abs(this.lat) > OFF_TRACK_LAT;
      this.offTimer = off ? this.offTimer + 1 : 0;
      const speed = this.car.speed;
      if (speed > 2) this.moved = true;
      // Stuck only counts once the car has got going, so waiting on the line is fine.
      this.stuckTimer = this.moved && speed < 0.6 ? this.stuckTimer + 1 : 0;
      // The last lap is timed at the line, but the episode ends a few metres later so the
      // game-over camera starts clear of the finish arch.
      const finished = this.lapsDone >= LAPS;
      if (finished && this.progress >= LAPS * this.track.length + 9) this.end("finish");
      else if (finished && (this.stuckTimer >= 15 || this.offTimer >= 15)) this.end("finish");
      else if (this.offTimer >= OFF_LIMIT) this.end("crash");
      else if (this.stuckTimer >= STUCK_LIMIT) this.end("stuck");
      else if (this.tick >= TIME_LIMIT) this.end("timeout");
    } else {
      this.overTicks++;
    }

    this.chase.update(this.car, racing ? 0 : this.overTicks / OVER_TICKS);

    if (this.smoke) {
      const slide = Math.abs(this.car.sideSpeed);
      const sh = Math.sin(this.car.heading);
      const ch = Math.cos(this.car.heading);
      const emitting = racing && ((this.car.drifting && this.car.speed > 4) || slide > 3 || (this.car.grass > 0.3 && this.car.speed > 5));
      if (emitting) {
        for (const side of [1, -1]) {
          const wx = this.car.x - sh * 0.75 + ch * 0.55 * side;
          const wz = this.car.z - ch * 0.75 - sh * 0.55 * side;
          this.smoke.emit(wx, wz, this.car.vx, this.car.vz, this.car.grass > 0.3 && !this.car.drifting && slide < 3);
        }
      }
      this.smoke.tick(DT);
    }
    if (this.audio) {
      const skid = racing ? Math.min(1, Math.max(this.car.drifting ? 0.5 : 0, (Math.abs(this.car.sideSpeed) - 1.5) / 5)) : 0;
      this.audio.update(this.car.forwardSpeed, controls.up ? 1 : 0, this.car.speed > 3 ? skid : 0, this.car.impact, racing);
    }
  }

  private end(reason: EndReason): void {
    this.reason = reason;
    this.overTicks = 0;
    this.audio?.silence();
  }

  render(alpha: number): void {
    const t = Math.max(0, Math.min(1, alpha));
    poseView(this.view, this.prevSnap, this.curSnap, t, !this.recording);
    for (const rv of this.rivals) poseView(rv.view, rv.prev, rv.cur, t, true);

    if (this.ghost) {
      const g = this.ghostBest;
      const k = this.tick - this.lapStart;
      const show = g !== null && this.overTicks < 0 && k * 3 + 2 < g.length;
      this.ghost.root.visible = show;
      if (show && g) {
        this.ghost.root.position.set(g[k * 3], 0, g[k * 3 + 1]);
        this.ghost.root.rotation.y = g[k * 3 + 2];
      }
    }

    this.chase.apply(this.camera, t);
    const cx = this.view.root.position.x;
    const cz = this.view.root.position.z;
    placeSun(this.sun, cx, cz);
    // The sky dome follows the camera so it never clips.
    const sky = this.scene.getObjectByName("sky");
    if (sky) sky.position.copy(this.camera.position);

    if (this.overTicks >= 0) {
      const k = Math.min(1, (this.overTicks + t) / OVER_TICKS);
      const e = k * k * (3 - 2 * k);
      this.wash.set(this.reason === "finish" ? PALETTE.finishWash : PALETTE.crashWash, e * 0.55);
    } else {
      this.wash.set(PALETTE.finishWash, 0);
    }
    this.renderer.render(this.scene, this.camera);
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }

  state(): GameState {
    return {
      lane: null,
      x: this.car.x,
      y: this.car.z,
      vertical: "grounded",
      alive: this.overTicks < 0,
      speed: this.car.forwardSpeed,
      // Metres along the track since the start of the episode, cumulative over laps.
      distance: this.progress - this.startProgress,
      extra: {
        heading: angleDiff(this.car.heading, 0),
        offTrack: Math.abs(this.lat) > OFF_TRACK_LAT ? 1 : 0,
        lap: this.lapsDone + 1,
      },
    };
  }

  isOver(): boolean {
    return this.overTicks >= OVER_TICKS;
  }

  hud(): Hud {
    const lap = Math.min(LAPS, this.lapsDone + 1);
    const lapTicks = this.lapsDone >= LAPS ? this.lastLap : this.tick - this.lapStart;
    return {
      Lap: `${lap}/${LAPS}`,
      Time: fmtTime(lapTicks),
      Best: this.bestLap > 0 ? fmtTime(this.bestLap) : "-:--.-",
    };
  }

  createAutopilot(seed: number): Autopilot {
    return createAutopilot(seed);
  }

  dispose(): void {
    this.audio?.dispose();
    this.scene?.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) m.geometry.dispose();
    });
    this.renderer?.dispose();
  }
}

export function createGame(): LucidGame {
  return new DriftKit();
}

export default { createGame };
