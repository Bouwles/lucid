// Chase camera. Its state advances once per logic tick (so recording frames are a pure function
// of the game state) and rendering interpolates between the last two ticks for smooth play.

import * as THREE from "three";
import { angleDiff } from "./track";
import type { Car } from "./car";

interface Rig {
  px: number;
  py: number;
  pz: number;
  tx: number;
  ty: number;
  tz: number;
  fov: number;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (t: number) => t * t * (3 - 2 * t);

export class ChaseCamera {
  private yaw = 0;
  private prev: Rig = { px: 0, py: 0, pz: 0, tx: 0, ty: 0, tz: 0, fov: 60 };
  private cur: Rig = { ...this.prev };
  private shake = 0;
  private shakeT = 0;

  /**
   * recording: one fixed rig (constant distance, height and field of view, no shake).
   * normal: pulls back and widens a little with speed, and shakes on impacts.
   */
  constructor(private readonly recording: boolean) {}

  reset(car: Car): void {
    this.yaw = car.heading;
    this.shake = 0;
    this.compute(car, 0);
    this.prev = { ...this.cur };
  }

  /** Advance one tick. over in [0, 1] is the game-over pull-up. */
  update(car: Car, over: number): void {
    this.prev = { ...this.cur };
    // Follow the heading with a lag; while sliding, lean towards the direction of travel so a
    // drift reads as a drift.
    let aim = car.heading;
    if (car.speed > 3 && car.forwardSpeed > 0) {
      const travel = Math.atan2(car.vx, car.vz);
      aim = car.heading + angleDiff(travel, car.heading) * 0.45;
    }
    this.yaw += angleDiff(aim, this.yaw) * 0.14;
    this.compute(car, over);
  }

  kick(strength: number): void {
    if (!this.recording) this.shake = Math.min(0.6, this.shake + strength);
  }

  private compute(car: Car, over: number): void {
    const fast = this.recording ? 0 : Math.min(1, Math.max(0, car.forwardSpeed) / 18);
    let dist = 6.6 + fast * 1.0;
    let height = 3.3 + fast * 0.2;
    let lookUp = 0.7;
    const e = smooth(Math.min(1, over));
    // Game over: swing up and over to the front of the car and look back down at it, so the car
    // stays in view and anything it just drove under (the finish arch) ends up behind it.
    dist = lerp(dist, -6, e);
    height = lerp(height, 10, e);
    lookUp = lerp(lookUp, 0, e);
    const ahead = lerp(3.5, 0, e);
    const sy = Math.sin(this.yaw);
    const cy = Math.cos(this.yaw);
    this.cur = {
      px: car.x - sy * dist,
      py: height,
      pz: car.z - cy * dist,
      tx: car.x + sy * ahead,
      ty: lookUp,
      tz: car.z + cy * ahead,
      fov: (this.recording ? 62 : 58) + fast * 8,
    };
    this.shakeT += 1;
    this.shake *= 0.82;
  }

  apply(camera: THREE.PerspectiveCamera, alpha: number): void {
    const a = this.prev;
    const b = this.cur;
    camera.position.set(lerp(a.px, b.px, alpha), lerp(a.py, b.py, alpha), lerp(a.pz, b.pz, alpha));
    const tx = lerp(a.tx, b.tx, alpha);
    const ty = lerp(a.ty, b.ty, alpha);
    const tz = lerp(a.tz, b.tz, alpha);
    if (this.shake > 0.002) {
      // A deterministic wobble (sum of sines), not random noise.
      const t = this.shakeT + alpha;
      camera.position.x += Math.sin(t * 2.7) * this.shake * 0.5;
      camera.position.y += Math.sin(t * 3.9 + 1.3) * this.shake * 0.35;
      camera.position.z += Math.sin(t * 3.1 + 2.1) * this.shake * 0.5;
    }
    camera.lookAt(tx, ty, tz);
    const fov = lerp(a.fov, b.fov, alpha);
    if (Math.abs(camera.fov - fov) > 1e-3) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  }
}
