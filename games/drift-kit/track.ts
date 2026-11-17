// The circuit: a closed loop of Kenney track tiles on a 10 m grid, plus the maths the rest of the
// game needs (centreline, projection to track coordinates, racing line and its speed profile).
//
// Track coordinates: s = metres along the centreline from the finish line, lat = metres to the
// left of the centreline (positive = left when facing the direction of travel).
// World axes: x and z on the ground, y up. Heading h means facing (sin h, cos h); turning left
// increases h.

export const TILE = 10;
/** Radius of the centreline through a corner tile (the arc pivots on the tile's inner corner). */
const R = TILE / 2;
/** Tarmac (to 3.5), the raised yellow line (to 3.9) and the kerb (to 4.5). Beyond is grass. */
export const ROAD_HALF = 4.5;
/** Where the car's centre counts as off the track (half the car is on the grass). */
export const OFF_TRACK_LAT = 4.8;

export type TileKind = "straight" | "finish" | "corner";

export interface Tile {
  kind: TileKind;
  i: number;
  j: number;
  /** Direction index of travel in and out: 0 = +z, 1 = +x, 2 = -z, 3 = -x. */
  dirIn: number;
  dirOut: number;
  /** Track distance at the tile's entry edge, and the tile's length along the centreline. */
  s0: number;
  len: number;
  /** Rotation of the tile model about y. */
  rotY: number;
  /** Corner pivot (world), and +1 for a left turn, -1 for a right turn, 0 for a straight. */
  px: number;
  pz: number;
  turn: number;
}

/**
 * The layout as a list of moves from the finish line, starting at cell (0, 0) heading +z.
 * F = finish (a straight), S = straight, L / R = 90 degree corner. Clockwise overall: 7 rights,
 * 3 lefts. It has a long start straight, a kink, an S-bend, a chicane and a tight final sweep.
 */
const LAYOUT = "FSSSRSLRSSRSSLRSSRSSLRSSRS";

const DIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
];

export interface Projection {
  /** Distance along the centreline in [0, length). */
  s: number;
  lat: number;
  tile: number;
}

export interface Pose {
  x: number;
  z: number;
  /** Unit tangent (direction of travel). */
  tx: number;
  tz: number;
}

export class Track {
  readonly tiles: Tile[] = [];
  readonly length: number;
  private cellIndex = new Map<string, number>();

  // Racing line, sampled every SAMPLE metres of centreline distance.
  static readonly SAMPLE = 0.5;
  readonly lineOffset: Float64Array;
  readonly lineX: Float64Array;
  readonly lineZ: Float64Array;
  /** Target speed (m/s) along the racing line for a car with the given grip. */
  readonly lineSpeed: Float64Array;
  readonly n: number;

  constructor() {
    let i = 0;
    let j = 0;
    let d = 0;
    let s = 0;
    for (const m of LAYOUT) {
      const dirIn = d;
      let dirOut = d;
      let turn = 0;
      if (m === "L") {
        dirOut = (d + 1) % 4;
        turn = 1;
      } else if (m === "R") {
        dirOut = (d + 3) % 4;
        turn = -1;
      }
      const kind: TileKind = m === "F" ? "finish" : m === "S" ? "straight" : "corner";
      // The corner model joins its +z and -x edges, pivoting on (-5, +5). Unrotated it is a left
      // turn when entered heading -z, and a right turn when entered heading +x.
      const rotY = kind === "corner" ? ((turn > 0 ? dirIn - 2 : dirIn - 1) * Math.PI) / 2 : (dirIn * Math.PI) / 2;
      const len = kind === "corner" ? (Math.PI / 2) * R : TILE;
      const [ax, az] = DIRS[dirIn];
      const [bx, bz] = DIRS[dirOut];
      this.tiles.push({
        kind,
        i,
        j,
        dirIn,
        dirOut,
        s0: s,
        len,
        rotY,
        px: i * TILE + (bx - ax) * R,
        pz: j * TILE + (bz - az) * R,
        turn,
      });
      this.cellIndex.set(`${i},${j}`, this.tiles.length - 1);
      s += len;
      d = dirOut;
      i += DIRS[d][0];
      j += DIRS[d][1];
    }
    if (i !== 0 || j !== 0 || d !== 0) throw new Error("drift-kit: track layout does not close");
    this.length = s;

    this.n = Math.round(this.length / Track.SAMPLE);
    this.lineOffset = new Float64Array(this.n);
    this.lineX = new Float64Array(this.n);
    this.lineZ = new Float64Array(this.n);
    this.lineSpeed = new Float64Array(this.n);
    this.buildRacingLine();
  }

  /** Is grid cell (i, j) part of the track? */
  hasCell(i: number, j: number): boolean {
    return this.cellIndex.has(`${i},${j}`);
  }

  /** Centreline position and tangent at distance s (wraps). */
  pose(s: number): Pose {
    s = ((s % this.length) + this.length) % this.length;
    // Tiles are in order of s0; a linear scan over ~26 tiles is cheap and branch-friendly.
    let k = this.tiles.length - 1;
    for (let t = 1; t < this.tiles.length; t++) {
      if (this.tiles[t].s0 > s) {
        k = t - 1;
        break;
      }
    }
    return this.localPose(this.tiles[k], s - this.tiles[k].s0);
  }

  private localPose(t: Tile, u: number): Pose {
    const [ax, az] = DIRS[t.dirIn];
    const ex = t.i * TILE - ax * R;
    const ez = t.j * TILE - az * R;
    if (t.turn === 0) return { x: ex + ax * u, z: ez + az * u, tx: ax, tz: az };
    // Heading grows anticlockwise seen from above, which is a negative cross product in x-z.
    const a = (-u / R) * t.turn;
    const rx = ex - t.px;
    const rz = ez - t.pz;
    const c = Math.cos(a);
    const sn = Math.sin(a);
    const bx = rx * c - rz * sn;
    const bz = rz * c + rx * sn;
    return { x: t.px + bx, z: t.pz + bz, tx: (bz / R) * t.turn, tz: (-bx / R) * t.turn };
  }

  /** Project a world point onto one tile: local distance (clamped), lateral offset and squared distance. */
  private projectTile(t: Tile, x: number, z: number): { u: number; lat: number; err: number } {
    const [ax, az] = DIRS[t.dirIn];
    const ex = t.i * TILE - ax * R;
    const ez = t.j * TILE - az * R;
    let u: number;
    if (t.turn === 0) {
      u = Math.min(TILE, Math.max(0, (x - ex) * ax + (z - ez) * az));
    } else {
      const rx = ex - t.px;
      const rz = ez - t.pz;
      const qx = x - t.px;
      const qz = z - t.pz;
      let ang = -Math.atan2(rx * qz - rz * qx, rx * qx + rz * qz) * t.turn;
      // Angles just behind the entry come back slightly negative; far behind wraps past pi.
      if (ang < -Math.PI * 0.75) ang += Math.PI * 2;
      u = Math.min(Math.PI / 2, Math.max(0, ang)) * R;
    }
    const p = this.localPose(t, u);
    const dx = x - p.x;
    const dz = z - p.z;
    // Left of the travel direction (tx, tz) is (tz, -tx).
    return { u, lat: dx * p.tz - dz * p.tx, err: dx * dx + dz * dz };
  }

  /** Nearest point on the centreline. Checks the point's grid cell and its neighbours. */
  project(x: number, z: number): Projection {
    const ci = Math.round(x / TILE);
    const cj = Math.round(z / TILE);
    let best = -1;
    let bestScore = Infinity;
    let bestU = 0;
    let bestLat = 0;
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        const k = this.cellIndex.get(`${ci + di},${cj + dj}`);
        if (k === undefined) continue;
        const p = this.projectTile(this.tiles[k], x, z);
        const score = p.err;
        if (score < bestScore) {
          bestScore = score;
          best = k;
          bestU = p.u;
          bestLat = p.lat;
        }
      }
    }
    if (best < 0) {
      // Far from the track (should not happen inside the walls): fall back to a full scan.
      for (let k = 0; k < this.tiles.length; k++) {
        const p = this.projectTile(this.tiles[k], x, z);
        const score = p.err;
        if (score < bestScore) {
          bestScore = score;
          best = k;
          bestU = p.u;
          bestLat = p.lat;
        }
      }
    }
    const s = (this.tiles[best].s0 + bestU) % this.length;
    return { s, lat: bestLat, tile: best };
  }

  /** Racing line sample index for distance s. */
  lineIndex(s: number): number {
    const k = Math.floor((((s % this.length) + this.length) % this.length) / Track.SAMPLE);
    return k >= this.n ? 0 : k;
  }

  /** A point on the racing line, offset sideways by extra metres (positive = left). */
  linePoint(s: number, extra = 0): { x: number; z: number; tx: number; tz: number } {
    const p = this.pose(s);
    const k = this.lineIndex(s);
    const k2 = (k + 1) % this.n;
    const f = (((s % this.length) + this.length) % this.length) / Track.SAMPLE - k;
    const o = this.lineOffset[k] * (1 - Math.min(1, Math.max(0, f))) + this.lineOffset[k2] * Math.min(1, Math.max(0, f)) + extra;
    return { x: p.x + p.tz * o, z: p.z - p.tx * o, tx: p.tx, tz: p.tz };
  }

  /**
   * An "elastic band" racing line: each sample slides sideways towards the midpoint of its
   * neighbours, which straightens the path and cuts corners, but never past the kerbs.
   */
  private buildRacingLine(): void {
    const n = this.n;
    const cx = new Float64Array(n);
    const cz = new Float64Array(n);
    const nx = new Float64Array(n);
    const nz = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const p = this.pose(k * Track.SAMPLE);
      cx[k] = p.x;
      cz[k] = p.z;
      nx[k] = p.tz; // left normal
      nz[k] = -p.tx;
    }
    const off = this.lineOffset;
    const limit = ROAD_HALF - 1.2;
    for (let it = 0; it < 1500; it++) {
      for (let k = 0; k < n; k++) {
        const a = (k + n - 2) % n;
        const b = (k + 2) % n;
        const px = cx[k] + nx[k] * off[k];
        const pz = cz[k] + nz[k] * off[k];
        const mx = (cx[a] + nx[a] * off[a] + cx[b] + nx[b] * off[b]) / 2;
        const mz = (cz[a] + nz[a] * off[a] + cz[b] + nz[b] * off[b]) / 2;
        const o = off[k] + 0.5 * ((mx - px) * nx[k] + (mz - pz) * nz[k]);
        off[k] = Math.max(-limit, Math.min(limit, o));
      }
    }
    for (let k = 0; k < n; k++) {
      this.lineX[k] = cx[k] + nx[k] * off[k];
      this.lineZ[k] = cz[k] + nz[k] * off[k];
    }
    // Speed from curvature (v^2 * k <= lateral grip), then limit braking and acceleration.
    const aLat = 15;
    const vTop = 17.5;
    const brake = 13;
    const accel = 7;
    const v = this.lineSpeed;
    const step = 4;
    for (let k = 0; k < n; k++) {
      const a = (k + n - step) % n;
      const b = (k + step) % n;
      const h1 = Math.atan2(this.lineX[k] - this.lineX[a], this.lineZ[k] - this.lineZ[a]);
      const h2 = Math.atan2(this.lineX[b] - this.lineX[k], this.lineZ[b] - this.lineZ[k]);
      let dh = h2 - h1;
      dh = Math.atan2(Math.sin(dh), Math.cos(dh));
      const ds = Math.hypot(this.lineX[b] - this.lineX[a], this.lineZ[b] - this.lineZ[a]) / 2;
      const kappa = Math.abs(dh) / Math.max(1e-6, ds);
      v[k] = Math.min(vTop, Math.sqrt(aLat / Math.max(1e-6, kappa)));
    }
    const ds = Track.SAMPLE;
    for (let pass = 0; pass < 2; pass++) {
      for (let k = n - 1; k >= 0; k--) {
        const nxt = v[(k + 1) % n];
        v[k] = Math.min(v[k], Math.sqrt(nxt * nxt + 2 * brake * ds));
      }
      for (let k = 0; k < n; k++) {
        const prv = v[(k + n - 1) % n];
        v[k] = Math.min(v[k], Math.sqrt(prv * prv + 2 * accel * ds));
      }
    }
  }

  /** Grid cells within the given margin of the track that are not track. */
  decorationCells(margin: number): Array<[number, number]> {
    let i0 = Infinity;
    let i1 = -Infinity;
    let j0 = Infinity;
    let j1 = -Infinity;
    for (const t of this.tiles) {
      i0 = Math.min(i0, t.i);
      i1 = Math.max(i1, t.i);
      j0 = Math.min(j0, t.j);
      j1 = Math.max(j1, t.j);
    }
    const out: Array<[number, number]> = [];
    for (let i = i0 - margin; i <= i1 + margin; i++) {
      for (let j = j0 - margin; j <= j1 + margin; j++) if (!this.hasCell(i, j)) out.push([i, j]);
    }
    return out;
  }
}

/** Smallest signed difference a - b of two angles, in (-pi, pi]. */
export function angleDiff(a: number, b: number): number {
  let d = (a - b) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d <= -Math.PI) d += Math.PI * 2;
  return d;
}
