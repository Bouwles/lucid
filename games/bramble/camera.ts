// Side-scrolling camera: follows the player with a small look-ahead in the direction of travel,
// smoothed, and only re-centres vertically when the player lands somewhere new (the usual
// platformer trick that stops every jump from bobbing the view). Steps per logic tick, so recording
// mode gets exactly the same framing every run; it just never shakes.

import * as THREE from "three";
import { PHYS } from "./physics";
import type { World } from "./world";

export const FOV = 30;
export const DISTANCE = 19.5;
/** How far the camera sits above the point it looks at: a slight downward view shows the tops of tiles. */
export const RISE = 2.4;
/** Half the visible height at the play plane. */
/** The view centre sits this far above the ground the player last stood on. */
export const FRAME_UP = 1.7;
export const HALF_H = DISTANCE * Math.tan(((FOV / 2) * Math.PI) / 180);

export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  private x = 0;
  private y = 0;
  private px = 0;
  private py = 0;
  private look = 0;
  private anchorY = 0;
  private shake = 0;
  private shakeTick = 0;
  aspect = 1;

  constructor(private readonly allowShake: boolean) {
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.5, 400);
  }

  private bounds(w: World): { x: number; y: number } {
    const halfW = HALF_H * this.aspect;
    const lv = w.level;
    const lo = halfW - 0.5;
    const hi = lv.w - halfW + 0.5;
    const tx = lo > hi ? lv.w / 2 : Math.max(lo, Math.min(hi, this.x));
    // Keep the bottom of the view a little below the ground, the top at most at the level's top.
    const yMin = HALF_H - 1.6;
    const yMax = Math.max(yMin, lv.h - HALF_H);
    return { x: tx, y: Math.max(yMin, Math.min(yMax, this.y)) };
  }

  snap(w: World): void {
    const p = w.player;
    this.look = 0;
    this.anchorY = p.y;
    this.x = p.x + 1.5;
    this.y = p.y + FRAME_UP;
    const b = this.bounds(w);
    this.x = this.px = b.x;
    this.y = this.py = b.y;
    this.shake = 0;
  }

  addShake(amount: number): void {
    if (this.allowShake) this.shake = Math.max(this.shake, amount);
  }

  tick(w: World): void {
    const p = w.player;
    this.px = this.x;
    this.py = this.y;
    this.shakeTick++;
    this.shake *= 0.82;
    if (w.phase === "over" && w.overKind !== "win") {
      // Hold still and watch the tumble.
      return;
    }
    const lookTarget = Math.max(-1, Math.min(1, p.vx / PHYS.runMax)) * 2.4 + p.facing * 0.4;
    this.look += (lookTarget - this.look) * 0.06;
    if (p.grounded) this.anchorY = p.y;
    else if (p.y < this.anchorY - 0.5) this.anchorY = p.y; // follow falls straight away
    else if (p.y > this.anchorY + 3.5) this.anchorY = p.y - 3.5; // and very high jumps
    this.x += (p.x + this.look - this.x) * 0.2;
    this.y += (this.anchorY + FRAME_UP - this.y) * 0.09;
    const b = this.bounds(w);
    this.x = b.x;
    this.y = b.y;
  }

  render(alpha: number): void {
    let x = this.px + (this.x - this.px) * alpha;
    let y = this.py + (this.y - this.py) * alpha;
    if (this.shake > 0.005) {
      const t = this.shakeTick + alpha;
      x += Math.sin(t * 13.7) * this.shake;
      y += Math.sin(t * 17.3 + 1.3) * this.shake * 0.8;
    }
    this.camera.position.set(x, y + RISE, DISTANCE);
    this.camera.lookAt(x, y, 0);
  }

  resize(aspect: number): void {
    this.aspect = aspect;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  get center(): { x: number; y: number } {
    return { x: this.x, y: this.y };
  }
}
