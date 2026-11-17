// The player's robot (the kit's character.glb): its idle, walk and jump clips, plus procedural
// squash and stretch, a running bob, turning to face the direction of travel, and a tumble on death.
// Everything here advances per logic tick, so recordings look identical on every run.

import * as THREE from "three";
import type { Assets } from "./assets";
import { DT, PHYS } from "./physics";
import type { World } from "./world";

/** Turned a little towards the camera, so the face reads in a side view. */
const YAW = Math.PI / 2 - 0.5;
const MODEL_SCALE = 0.92;

type Clip = "idle" | "walk" | "jump";

interface Pose {
  x: number;
  y: number;
  z: number;
  yaw: number;
  roll: number;
  sx: number;
  sy: number;
  bob: number;
}

export class CharacterView {
  readonly root = new THREE.Group();
  private model: THREE.Object3D;
  private mixer: THREE.AnimationMixer;
  private actions = new Map<Clip, THREE.AnimationAction>();
  private clip: Clip = "idle";
  private prev: Pose = { x: 0, y: 0, z: 0, yaw: YAW, roll: 0, sx: 1, sy: 1, bob: 0 };
  private cur: Pose = { ...this.prev };
  /** Squash scale, eased back to 1 every tick. */
  private squash = new THREE.Vector3(1, 1, 1);
  private runPhase = 0;
  private animTicks = 0;
  private animTime = 0;
  private walkRate = 1;

  constructor(assets: Assets) {
    this.model = assets.character.scene.clone(true);
    this.model.scale.setScalar(MODEL_SCALE);
    this.root.add(this.model);
    this.mixer = new THREE.AnimationMixer(this.model);
    for (const clip of assets.character.animations) {
      if (clip.name === "idle" || clip.name === "walk" || clip.name === "jump") {
        const a = this.mixer.clipAction(clip);
        a.enabled = true;
        this.actions.set(clip.name, a);
      }
    }
    this.actions.get("idle")?.play();
  }

  /** Snap to the world's current state (new episode or level). */
  snap(w: World): void {
    const p = w.player;
    this.cur = { x: p.x, y: p.y, z: 0, yaw: p.facing * YAW, roll: 0, sx: 1, sy: 1, bob: 0 };
    this.prev = { ...this.cur };
    this.squash.set(1, 1, 1);
    this.runPhase = 0;
    this.animTicks = 0;
    this.animTime = 0;
    this.mixer.stopAllAction();
    this.clip = "idle";
    this.actions.get("idle")?.reset().play();
    this.mixer.setTime(0);
  }

  /** Called after each logic tick with that tick's events already applied to the world. */
  tick(w: World, jumped: boolean, landSpeed: number, dashed: boolean): void {
    const p = w.player;
    this.prev = { ...this.cur };
    this.animTicks++;

    if (jumped) this.squash.set(0.72, 1.32, 0.72);
    if (landSpeed > 2) {
      const k = Math.min(1, landSpeed / 18);
      this.squash.set(1 + 0.38 * k, 1 - 0.34 * k, 1 + 0.38 * k);
    }
    if (dashed) this.squash.set(1.3, 0.82, 1);
    if (p.dashT > 0) this.squash.lerp(new THREE.Vector3(1.25, 0.85, 1), 0.5);
    this.squash.lerp(new THREE.Vector3(1, 1, 1), 0.22);

    const dead = w.phase === "over";
    const speed = Math.abs(p.vx);
    let bob = 0;
    if (p.grounded && speed > 0.5 && !dead) {
      this.runPhase += speed * DT * 2.4;
      bob = Math.abs(Math.sin(this.runPhase * Math.PI)) * 0.07;
    }

    let yaw = this.cur.yaw;
    let roll = this.cur.roll;
    if (dead && w.overKind !== "win") {
      // Tumble: face the camera and spin.
      yaw += (0 - yaw) * 0.3;
      roll += w.overKind === "time" ? 0.03 * Math.max(0, 1 - w.phaseT / 20) : 0.32;
    } else if (w.phase === "clear" || (dead && w.overKind === "win")) {
      yaw += (0 - yaw) * 0.25; // turn to the camera to celebrate
    } else {
      const target = p.facing * YAW;
      yaw += (target - yaw) * 0.38;
      // Lean into the run and the dash.
      const lean = -Math.max(-1, Math.min(1, p.vx / PHYS.runMax)) * (p.dashT > 0 ? 0.35 : 0.08);
      roll += (lean - roll) * 0.3;
    }

    // A hit sends the robot tumbling out in front of the level, so it falls past the ground, not behind it.
    const z = dead && w.overKind === "hit" ? Math.min(2, w.phaseT * 0.25) : 0;
    this.cur = { x: p.x, y: p.y, z, yaw, roll, sx: this.squash.x, sy: this.squash.y, bob };

    let next: Clip = "idle";
    if (dead) next = w.overKind === "time" ? "idle" : "jump";
    else if (!p.grounded) next = "jump";
    else if (speed > 0.5) next = "walk";
    this.walkRate = Math.max(0.6, speed / PHYS.runMax) * 1.6;
    if (next !== this.clip) {
      const from = this.actions.get(this.clip);
      const to = this.actions.get(next);
      if (to) {
        to.reset().play();
        if (from) to.crossFadeFrom(from, 0.1, false);
      }
      this.clip = next;
    }
    const walk = this.actions.get("walk");
    if (walk) walk.timeScale = this.walkRate;
  }

  render(alpha: number): void {
    const a = this.prev;
    const b = this.cur;
    const lerp = (u: number, v: number) => u + (v - u) * alpha;
    this.root.position.set(lerp(a.x, b.x), lerp(a.y, b.y) + lerp(a.bob, b.bob), lerp(a.z, b.z));
    this.model.rotation.set(0, lerp(a.yaw, b.yaw), 0);
    this.root.rotation.z = lerp(a.roll, b.roll);
    // Squash around the feet; the tumble spins around the middle of the body.
    const sx = lerp(a.sx, b.sx);
    const sy = lerp(a.sy, b.sy);
    this.root.scale.set(sx, sy, sx);
    const roll = this.root.rotation.z;
    if (roll !== 0) {
      const mid = 0.45;
      this.root.position.x += Math.sin(roll) * mid;
      this.root.position.y += mid - Math.cos(roll) * mid;
    }
    // Clip time follows logic ticks, not the wall clock.
    const t = (this.animTicks - 1 + alpha) * DT;
    const dt = t - this.animTime;
    if (dt > 0) this.mixer.update(dt);
    this.animTime = Math.max(this.animTime, t);
  }
}
