// Arcade car physics. Pure maths, no rendering, no randomness: the same inputs give the same
// motion. World velocity is kept separately from heading, so when the car turns its momentum
// carries on and the tyres pull it back into line (grip). Lower the grip (handbrake) and it slides.

export interface CarControls {
  left: boolean;
  right: boolean;
  up: boolean;
  down: boolean;
  /** Handbrake: rear grip drops, the car rotates faster and slides. */
  drift: boolean;
}

/** Circles a car can hit: [x, z, radius]. */
export interface Obstacles {
  near(x: number, z: number, out: number[]): void;
}

export const CAR = {
  /** Two collision circles along the car, centred this far in front of / behind its middle. */
  circleOffset: 0.7,
  circleRadius: 0.78,
  topSpeed: 18,
  reverseTop: 6,
  accel: 11,
  brake: 24,
  reverseAccel: 7,
  coast: 2.2,
  drag: 0.045,
  grip: 9,
  driftGrip: 1.3,
  /** How quickly grip comes back after the handbrake is let go (per second). Forgiving. */
  gripRecover: 5,
  grassDrag: 0.8,
  grassGrip: 0.6,
};

export interface CarSnapshot {
  x: number;
  z: number;
  heading: number;
  steer: number;
  wheelSpin: number;
  lean: number;
  pitch: number;
}

export class Car {
  x = 0;
  z = 0;
  /** Facing (sin h, cos h). Turning left increases it. */
  heading = 0;
  vx = 0;
  vz = 0;
  yawRate = 0;
  /** Smoothed steering in [-1, 1], positive = left. */
  steer = 0;
  grip = CAR.grip;
  /** 0 on tarmac, 1 fully on grass. Set by the game from the track before each step. */
  grass = 0;
  /** Strongest impact speed (m/s into the obstacle) during the last step. */
  impact = 0;
  /** Rolling angle of the wheels (radians), for drawing. */
  wheelSpin = 0;
  /** Body roll and pitch (radians), for drawing. */
  lean = 0;
  pitch = 0;
  private lastFwd = 0;
  drifting = false;

  place(x: number, z: number, heading: number): void {
    this.x = x;
    this.z = z;
    this.heading = heading;
    this.vx = this.vz = this.yawRate = this.steer = 0;
    this.grip = CAR.grip;
    this.grass = 0;
    this.impact = 0;
    this.wheelSpin = this.lean = this.pitch = this.lastFwd = 0;
    this.drifting = false;
  }

  /** Signed speed along the car's facing (m/s). */
  get forwardSpeed(): number {
    return this.vx * Math.sin(this.heading) + this.vz * Math.cos(this.heading);
  }

  /** Sideways speed (m/s, positive = sliding left). */
  get sideSpeed(): number {
    return this.vx * Math.cos(this.heading) - this.vz * Math.sin(this.heading);
  }

  get speed(): number {
    return Math.hypot(this.vx, this.vz);
  }

  /** Angle between where the car points and where it is going (0 when gripping). */
  get slipAngle(): number {
    if (this.speed < 1) return 0;
    return Math.atan2(this.sideSpeed, Math.abs(this.forwardSpeed));
  }

  snapshot(): CarSnapshot {
    return { x: this.x, z: this.z, heading: this.heading, steer: this.steer, wheelSpin: this.wheelSpin, lean: this.lean, pitch: this.pitch };
  }

  step(c: CarControls, dt: number, obstacles: Obstacles | null, extra: number[] = []): void {
    this.impact = 0;
    const sub = 2;
    const h = dt / sub;
    for (let k = 0; k < sub; k++) this.substep(c, h, obstacles, extra);
    // Cosmetic body motion (deterministic, derived from the physics).
    const fwd = this.forwardSpeed;
    const accel = (fwd - this.lastFwd) / dt;
    this.lastFwd = fwd;
    this.wheelSpin += (fwd * dt) / 0.3;
    const targetLean = -this.yawRate * Math.min(1, Math.abs(fwd) / 10) * 0.07 - this.sideSpeed * 0.012;
    this.lean += (targetLean - this.lean) * Math.min(1, dt * 10);
    const targetPitch = Math.max(-0.08, Math.min(0.08, -accel * 0.006));
    this.pitch += (targetPitch - this.pitch) * Math.min(1, dt * 8);
  }

  private substep(c: CarControls, dt: number, obstacles: Obstacles | null, extra: number[]): void {
    const sh = Math.sin(this.heading);
    const ch = Math.cos(this.heading);
    let vF = this.vx * sh + this.vz * ch;
    let vL = this.vx * ch - this.vz * sh;

    // Steering eases in and out quickly, so a held key keeps turning the car every tick.
    const target = (c.left ? 1 : 0) - (c.right ? 1 : 0);
    this.steer += (target - this.steer) * Math.min(1, dt * (target === 0 ? 12 : 8));

    // Throttle, brake, reverse.
    if (c.up) {
      if (vF < -0.5) vF += CAR.brake * dt;
      else vF += CAR.accel * Math.max(0, 1 - vF / CAR.topSpeed) * dt;
    } else if (c.down) {
      if (vF > 0.5) vF -= CAR.brake * dt;
      else vF = Math.max(-CAR.reverseTop, vF - CAR.reverseAccel * dt);
    } else {
      vF -= Math.sign(vF) * Math.min(Math.abs(vF), CAR.coast * dt);
    }
    if (c.drift) vF -= Math.sign(vF) * Math.min(Math.abs(vF), 3.5 * dt);
    vF -= vF * (CAR.drag + CAR.grassDrag * this.grass) * dt;

    // Yaw: quick at low and middle speeds, calmer at the top end; faster with the handbrake.
    const absF = Math.abs(vF);
    const roll = Math.min(1, absF / 4);
    let maxYaw = (2.7 - 1.0 * Math.min(1, absF / CAR.topSpeed)) * roll;
    if (c.drift && absF > 3) maxYaw *= 1.5;
    let targetYaw = this.steer * maxYaw * (vF >= 0 ? 1 : -1);
    // Forgiving: past ~65 degrees of slide, stop feeding more rotation into it.
    const slip = this.speed > 2 ? Math.atan2(vL, Math.abs(vF)) : 0;
    if (Math.abs(slip) > 1.15 && Math.sign(targetYaw) === Math.sign(-slip)) targetYaw *= 0.3;
    this.yawRate += (targetYaw - this.yawRate) * Math.min(1, dt * 9);
    const oldSh = sh;
    const oldCh = ch;
    this.heading += this.yawRate * dt;

    // Momentum: the velocity stays put in the world while the body turns under it.
    const wx = vF * oldSh + vL * oldCh;
    const wz = vF * oldCh - vL * oldSh;
    const nsh = Math.sin(this.heading);
    const nch = Math.cos(this.heading);
    vF = wx * nsh + wz * nch;
    vL = wx * nch - wz * nsh;

    // Grip pulls the slide back in line; part of the lost sideways speed becomes forward speed,
    // which keeps drifts fast and recovery smooth.
    this.drifting = c.drift && absF > 3;
    const gripTarget = (this.drifting ? CAR.driftGrip : CAR.grip) * (1 - (1 - CAR.grassGrip) * this.grass);
    if (gripTarget < this.grip) this.grip = gripTarget;
    else this.grip += (gripTarget - this.grip) * Math.min(1, dt * CAR.gripRecover);
    const newL = vL * Math.exp(-this.grip * dt);
    const lost = Math.abs(vL) - Math.abs(newL);
    vL = newL;
    if (vF >= 0) vF += lost * 0.15;
    else vF -= lost * 0.15;

    this.vx = vF * nsh + vL * nch;
    this.vz = vF * nch - vL * nsh;
    this.x += this.vx * dt;
    this.z += this.vz * dt;

    if (obstacles) this.collide(obstacles, extra);
  }

  private collide(obstacles: Obstacles, extra: number[]): void {
    const near: number[] = [];
    const sh = Math.sin(this.heading);
    const ch = Math.cos(this.heading);
    for (const side of [1, -1]) {
      const cx = this.x + sh * CAR.circleOffset * side;
      const cz = this.z + ch * CAR.circleOffset * side;
      near.length = 0;
      obstacles.near(cx, cz, near);
      for (let k = 0; k < extra.length; k++) near.push(extra[k]);
      for (let k = 0; k < near.length; k += 3) {
        const dx = cx - near[k];
        const dz = cz - near[k + 1];
        const min = CAR.circleRadius + near[k + 2];
        const d2 = dx * dx + dz * dz;
        if (d2 >= min * min || d2 < 1e-9) continue;
        const d = Math.sqrt(d2);
        const nx = dx / d;
        const nz = dz / d;
        const pen = min - d;
        this.x += nx * pen;
        this.z += nz * pen;
        const vn = this.vx * nx + this.vz * nz;
        if (vn < 0) {
          // Bounce a little, scrub some speed along the obstacle, and knock the nose round.
          this.vx -= 1.3 * vn * nx;
          this.vz -= 1.3 * vn * nz;
          this.vx *= 0.8;
          this.vz *= 0.8;
          this.impact = Math.max(this.impact, -vn);
          this.yawRate *= 0.4;
          const tangential = (nx * ch - nz * sh) * side;
          this.yawRate += tangential * Math.min(2, -vn * 0.25);
        }
      }
    }
  }
}

/** A uniform grid over the obstacle circles, so a car only tests the ones nearby. */
export class ObstacleGrid implements Obstacles {
  private cells = new Map<number, number[]>();
  private static readonly CELL = 4;

  constructor(circles: ReadonlyArray<readonly [number, number, number]>) {
    for (const [x, z, r] of circles) {
      const i0 = Math.floor((x - r - 1) / ObstacleGrid.CELL);
      const i1 = Math.floor((x + r + 1) / ObstacleGrid.CELL);
      const j0 = Math.floor((z - r - 1) / ObstacleGrid.CELL);
      const j1 = Math.floor((z + r + 1) / ObstacleGrid.CELL);
      for (let i = i0; i <= i1; i++) {
        for (let j = j0; j <= j1; j++) {
          const key = ObstacleGrid.key(i, j);
          let list = this.cells.get(key);
          if (!list) this.cells.set(key, (list = []));
          list.push(x, z, r);
        }
      }
    }
  }

  private static key(i: number, j: number): number {
    return (i + 4096) * 8192 + (j + 4096);
  }

  near(x: number, z: number, out: number[]): void {
    const list = this.cells.get(ObstacleGrid.key(Math.floor(x / ObstacleGrid.CELL), Math.floor(z / ObstacleGrid.CELL)));
    if (list) for (let k = 0; k < list.length; k++) out.push(list[k]);
  }
}
