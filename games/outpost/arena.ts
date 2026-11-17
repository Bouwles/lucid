// The Outpost arena: one floating platform built from the kit's tiles and walls, plus collision.
//
// Axes: x is east, z is south, y is up. The four sides have their own accent colour so a dream
// can tell where the player is: north is red (the gate), east is blue (the raised deck),
// south is yellow (the walkway and crates) and west is purple (the radio mast).
// Collision is a list of axis-aligned boxes; there is no physics engine.

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { mulberry32 } from "../shared/rng";
import { type Assets, type ModelName, shiftPalette } from "./assets";

export interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z0: number;
  z1: number;
}

export interface Pad {
  kind: "health" | "ammo";
  x: number;
  y: number;
  z: number;
}

export interface Spawn {
  x: number;
  z: number;
}

/** Size of one large grass tile. */
export const TILE = 5.2;

// Palette shifts from the kit's red-orange (see shiftPalette).
const BLUE = 2;
const LIGHT_BLUE = 4;
const PURPLE = 6;
const YELLOW = -4;

interface Piece {
  model: ModelName;
  x: number;
  y: number;
  z: number;
  /** Quarter turns about y. */
  rot?: number;
  shift?: number;
  solid?: boolean;
}

// Tile heights by row (north to south) and column (west to east). "." is open sky.
const GRID = [
  ". 0 0 0 0",
  "0 0 0 0 1",
  "0 0 0 0 1",
  "0 0 0 0 0",
  "0 0 0 . .",
];

function layoutPieces(): Piece[] {
  const p: Piece[] = [];
  GRID.forEach((row, r) => {
    row.split(" ").forEach((c, col) => {
      if (c === ".") return;
      const x = (col - 2) * TILE;
      const z = (r - 2) * TILE;
      const level = Number(c);
      // Raised tiles are stacked so their sides read as solid ground.
      for (let k = 0; k <= level * 2; k++) p.push({ model: "platformLarge", x, y: -0.5 + k * 0.5, z, solid: k === level * 2 });
    });
  });

  // Centre: a light blue dais with the beacon mast.
  p.push({ model: "platform", x: 0, y: 0, z: 0, shift: LIGHT_BLUE, solid: true });

  // North: the red gate, two stacked pillars and a lintel, plus red low walls as cover.
  for (const gx of [-2.05, 2.05]) {
    p.push({ model: "wallHigh", x: gx, y: 0, z: -11.2, solid: true });
    p.push({ model: "wallHigh", x: gx, y: 1.7, z: -11.2, solid: true });
  }
  p.push({ model: "wallLow", x: -1.1, y: 3.4, z: -11.2, solid: true });
  p.push({ model: "wallLow", x: 1.1, y: 3.4, z: -11.2, solid: true });
  p.push({ model: "wallLow", x: -6.2, y: 0, z: -6.4, solid: true });
  p.push({ model: "wallLow", x: 4.6, y: 0, z: -6.8, rot: 1, solid: true });

  // East: the raised blue deck with two steps up, low walls along its lip and a tall cover block.
  p.push({ model: "platform", x: 6.6, y: 0, z: -2.6, shift: BLUE, solid: true });
  p.push({ model: "platform", x: 10.4, y: 0, z: 3.7, shift: BLUE, solid: true });
  p.push({ model: "wallLow", x: 8.45, y: 1, z: -6.3, rot: 1, shift: BLUE, solid: true });
  p.push({ model: "wallLow", x: 8.45, y: 1, z: 1.2, rot: 1, shift: BLUE, solid: true });
  p.push({ model: "wallHigh", x: 11.8, y: 1, z: -4.2, shift: BLUE, solid: true });

  // West: purple cover near the radio mast.
  p.push({ model: "wallHigh", x: -6.0, y: 0, z: 3.4, rot: 1, shift: PURPLE, solid: true });
  p.push({ model: "wallLow", x: -6.6, y: 0, z: -3.2, rot: 1, shift: PURPLE, solid: true });
  p.push({ model: "wallLow", x: -10.6, y: 0, z: 6.0, shift: PURPLE, solid: true });

  // South: yellow walls and a narrow yellow walkway out to a lone pad over the void.
  p.push({ model: "wallLow", x: -3.2, y: 0, z: 8.6, shift: YELLOW, solid: true });
  p.push({ model: "wallLow", x: 2.2, y: 0, z: 6.6, rot: 1, shift: YELLOW, solid: true });
  p.push({ model: "wallHigh", x: -8.0, y: 0, z: 11.2, shift: YELLOW, solid: true });
  p.push({ model: "platform", x: -10.4, y: -0.5, z: 14.1, shift: YELLOW, solid: true });
  p.push({ model: "platform", x: -10.4, y: -0.5, z: 16.3, shift: YELLOW, solid: true });
  return p;
}

export const PADS: Pad[] = [
  { kind: "health", x: -10.4, y: 0, z: 16.3 },
  { kind: "health", x: 11.2, y: 1, z: -0.6 },
  { kind: "health", x: -4.6, y: 0, z: -10.2 },
  { kind: "ammo", x: 10.6, y: 1, z: -6.6 },
  { kind: "ammo", x: 0.0, y: 0, z: 10.6 },
  { kind: "ammo", x: -8.4, y: 0, z: -6.4 },
  { kind: "ammo", x: 6.4, y: 0, z: -10.6 },
];

export const SPAWNS: Spawn[] = [
  { x: 0, z: 5.6 },
  { x: -5.6, z: 0.4 },
  { x: 4.4, z: -3.6 },
  { x: -2.6, z: -5.8 },
  { x: 5.8, z: 5.2 },
  { x: -8.4, z: 9.0 },
  { x: 10.6, z: -2.0 },
];

/** Places that give cover from the open middle, for the autopilot. */
export const COVER: Spawn[] = [
  { x: -6.2, z: -5.3 },
  { x: 5.7, z: -6.8 },
  { x: 9.4, z: -6.0 },
  { x: -5.0, z: 3.4 },
  { x: -5.5, z: -3.2 },
  { x: -3.2, z: 7.5 },
  { x: 3.2, z: 6.6 },
  { x: 11.8, z: -3.0 },
  { x: -10.4, z: -1.0 },
];

// The radio mast on the west side (procedural), as a solid box.
const MAST = { x: -10.6, z: -3.4 };

// ---------------------------------------------------------------- collision

function pieceBox(geo: THREE.BufferGeometry, pc: Piece): Box {
  const bb = geo.boundingBox!;
  const quarter = ((pc.rot ?? 0) % 2 + 2) % 2;
  const [ax0, ax1, az0, az1] = quarter ? [bb.min.z, bb.max.z, bb.min.x, bb.max.x] : [bb.min.x, bb.max.x, bb.min.z, bb.max.z];
  return { x0: pc.x + ax0, x1: pc.x + ax1, y0: pc.y + bb.min.y, y1: pc.y + bb.max.y, z0: pc.z + az0, z1: pc.z + az1 };
}

/** Highest surface top at or below feet + step under a footprint of half-size r. -Infinity over the void. */
export function groundHeight(boxes: Box[], x: number, z: number, r: number, feet: number, step: number): number {
  let best = -Infinity;
  for (const b of boxes) {
    if (x + r > b.x0 && x - r < b.x1 && z + r > b.z0 && z - r < b.z1 && b.y1 <= feet + step && b.y1 > best) best = b.y1;
  }
  return best;
}

/** Pushes a cylinder (radius r, from feet to feet + height) out of every box it overlaps. Mutates p. */
export function collideXZ(boxes: Box[], p: { x: number; z: number }, feet: number, r: number, height: number, step: number): boolean {
  let hit = false;
  for (const b of boxes) {
    if (b.y1 <= feet + step || b.y0 >= feet + height) continue;
    if (p.x <= b.x0 - r || p.x >= b.x1 + r || p.z <= b.z0 - r || p.z >= b.z1 + r) continue;
    const dl = p.x - (b.x0 - r);
    const dr = b.x1 + r - p.x;
    const dn = p.z - (b.z0 - r);
    const ds = b.z1 + r - p.z;
    const m = Math.min(dl, dr, dn, ds);
    if (m === dl) p.x = b.x0 - r;
    else if (m === dr) p.x = b.x1 + r;
    else if (m === dn) p.z = b.z0 - r;
    else p.z = b.z1 + r;
    hit = true;
  }
  return hit;
}

/** Lowest box bottom above y within a footprint, for head bumps. */
export function ceilingAbove(boxes: Box[], x: number, z: number, r: number, y: number): number {
  let best = Infinity;
  for (const b of boxes) {
    if (x + r > b.x0 && x - r < b.x1 && z + r > b.z0 && z - r < b.z1 && b.y0 >= y && b.y0 < best) best = b.y0;
  }
  return best;
}

/** Distance along a unit ray to the first box it enters, or maxT. */
export function rayBoxes(boxes: Box[], ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, maxT: number): number {
  let best = maxT;
  const ix = 1 / dx;
  const iy = 1 / dy;
  const iz = 1 / dz;
  for (const b of boxes) {
    let t0 = (b.x0 - ox) * ix;
    let t1 = (b.x1 - ox) * ix;
    let lo = Math.min(t0, t1);
    let hi = Math.max(t0, t1);
    t0 = (b.y0 - oy) * iy;
    t1 = (b.y1 - oy) * iy;
    lo = Math.max(lo, Math.min(t0, t1));
    hi = Math.min(hi, Math.max(t0, t1));
    t0 = (b.z0 - oz) * iz;
    t1 = (b.z1 - oz) * iz;
    lo = Math.max(lo, Math.min(t0, t1));
    hi = Math.min(hi, Math.max(t0, t1));
    if (hi >= Math.max(lo, 0) && lo < best) best = Math.max(lo, 0);
  }
  return best;
}

export function pointInBoxes(boxes: Box[], x: number, y: number, z: number): boolean {
  for (const b of boxes) if (x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1 && z > b.z0 && z < b.z1) return true;
  return false;
}

/** True if a straight line from a to b passes through any box. */
export function lineBlocked(boxes: Box[], ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-6) return false;
  return rayBoxes(boxes, ax, ay, az, dx / len, dy / len, dz / len, len) < len;
}

// ---------------------------------------------------------------- geometry

function placed(src: THREE.BufferGeometry, x: number, y: number, z: number, rotY = 0, scale = 1, shift = 0): THREE.BufferGeometry {
  const g = shiftPalette(src.clone(), shift);
  const m = new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotY),
    new THREE.Vector3(scale, scale, scale),
  );
  return g.applyMatrix4(m);
}

/** Non-indexed copy of g with a per-vertex colour (a constant, or a function of local y). */
function colored(g: THREE.BufferGeometry, color: THREE.ColorRepresentation | ((y: number) => THREE.Color)): THREE.BufferGeometry {
  const src = g.index ? g.toNonIndexed() : g;
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", src.getAttribute("position").clone());
  out.setAttribute("normal", src.getAttribute("normal").clone());
  const pos = out.getAttribute("position");
  const col = new Float32Array(pos.count * 3);
  const fixed = typeof color === "function" ? null : new THREE.Color(color);
  for (let i = 0; i < pos.count; i++) {
    const c = fixed ?? (color as (y: number) => THREE.Color)(pos.getY(i));
    col.set([c.r, c.g, c.b], i * 3);
  }
  out.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return out;
}

function at(g: THREE.BufferGeometry, x: number, y: number, z: number, rx = 0, ry = 0, rz = 0): THREE.BufferGeometry {
  return g.applyMatrix4(new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz))).translate(x, y, z);
}

const box = (w: number, h: number, d: number, c: THREE.ColorRepresentation) => colored(new THREE.BoxGeometry(w, h, d), c);
const cyl = (rt: number, rb: number, h: number, seg: number, c: THREE.ColorRepresentation) =>
  colored(new THREE.CylinderGeometry(rt, rb, h, seg), c);

/** An upside-down low-poly rock cone that hangs under a floating platform. */
function rock(r: number, h: number, seg: number, rot: number): THREE.BufferGeometry {
  const top = new THREE.Color("#8d86b8");
  const bottom = new THREE.Color("#3e3866");
  const g = new THREE.ConeGeometry(r, h, seg, 2);
  g.rotateX(Math.PI);
  g.rotateY(rot);
  g.translate(0, -h / 2, 0);
  return colored(g, (y) => bottom.clone().lerp(top, Math.max(0, Math.min(1, 1 + y / h))));
}

export interface ArenaView {
  group: THREE.Group;
  /** The spinning ring on the beacon; animated from the tick count. */
  beacon: THREE.Mesh;
  clouds: THREE.InstancedMesh;
  cloudBase: { x: number; y: number; z: number; s: number; r: number; speed: number }[];
  mastLight: THREE.Mesh;
}

export interface ArenaData {
  boxes: Box[];
  view: ArenaView;
}

/** Builds the arena's meshes (a few merged draw calls) and its collision boxes. */
export function buildArena(assets: Assets, kitMaterial: THREE.Material): ArenaData {
  const pieces = layoutPieces();
  const boxes: Box[] = [];
  const kitGeos: THREE.BufferGeometry[] = [];
  for (const pc of pieces) {
    const src = assets.geo[pc.model];
    kitGeos.push(placed(src, pc.x, pc.y, pc.z, ((pc.rot ?? 0) * Math.PI) / 2, 1, pc.shift ?? 0));
    if (pc.solid) {
      const b = pieceBox(src, pc);
      // Stacked tiles: one box from the bottom tile to the top.
      if (pc.model === "platformLarge") b.y0 = -0.5;
      boxes.push(b);
    }
  }

  // Mast collision.
  boxes.push({ x0: MAST.x - 0.75, x1: MAST.x + 0.75, y0: 0, y1: 5.2, z0: MAST.z - 0.75, z1: MAST.z + 0.75 });
  // Crates on the south-east tile.
  const crates = [
    { x: 4.6, y: 0, z: 4.0, s: 1.1, c: "#f2b33d" },
    { x: 5.8, y: 0, z: 4.3, s: 1.1, c: "#f08a3a" },
    { x: 5.1, y: 1.1, z: 4.1, s: 1.0, c: "#f6c75a" },
  ];
  for (const c of crates) {
    const h = c.s / 2;
    boxes.push({ x0: c.x - h, x1: c.x + h, y0: c.y, y1: c.y + c.s, z0: c.z - h, z1: c.z + h });
  }

  // Grass tufts from a fixed seed (the arena never changes between episodes).
  const grng = mulberry32(90210);
  let tufts = 0;
  for (let tries = 0; tries < 400 && tufts < 46; tries++) {
    const x = (grng() - 0.5) * 26;
    const z = (grng() - 0.5) * 26;
    const g = groundHeight(boxes, x, z, 0.3, 0.05, 0);
    if (g !== 0 || pointInBoxes(boxes, x, 0.2, z) || Math.hypot(x, z) < 2) continue;
    if (groundHeight(boxes, x, z, 0.3, 0.4, 0.3) > 0.05) continue;
    kitGeos.push(placed(assets.geo[grng() < 0.5 ? "grass" : "grassSmall"], x, 0, z, grng() * Math.PI * 2, 1.2 + grng() * 0.8));
    tufts++;
  }
  // A few tufts on the raised deck.
  for (const [x, z] of [[9.4, -1.5], [12.2, 1.6], [12.4, -7.0], [9.6, -4.2]]) {
    kitGeos.push(placed(assets.geo.grass, x, 1, z, x * 1.7, 1.5));
  }

  // Distant floating islands (decoration only).
  const islands = [
    { x: -32, y: -7, z: -24, s: 1.6 },
    { x: 34, y: -12, z: 6, s: 2.0 },
    { x: 10, y: -16, z: 38, s: 1.4 },
    { x: -30, y: -3, z: 20, s: 1.1 },
    { x: 18, y: 4, z: -36, s: 1.2 },
  ];
  const procGeos: THREE.BufferGeometry[] = [];
  const glowGeos: THREE.BufferGeometry[] = [];
  islands.forEach((is, i) => {
    kitGeos.push(placed(assets.geo.platformLarge, is.x, is.y, is.z, i * 0.7, is.s));
    procGeos.push(at(rock(2.9 * is.s, 4 + 3 * is.s, 6, i), is.x, is.y, is.z));
  });
  const floats = [
    { x: -20, y: 6, z: -12, r: 0.3, sh: 0 },
    { x: 22, y: 1, z: 20, r: 1.2, sh: LIGHT_BLUE },
    { x: -18, y: -6, z: -30, r: 2.0, sh: PURPLE },
    { x: 26, y: 8, z: -18, r: 0.8, sh: YELLOW },
  ];
  for (const f of floats) kitGeos.push(placed(assets.geo.platform, f.x, f.y, f.z, f.r, 1.6, f.sh));

  // Rock underside of the main platform.
  GRID.forEach((row, r) => {
    row.split(" ").forEach((c, col) => {
      if (c === ".") return;
      const x = (col - 2) * TILE;
      const z = (r - 2) * TILE;
      const h = 3.5 + ((col * 7 + r * 3) % 5) * 0.9;
      procGeos.push(at(rock(3.7, h, 6, col + r), x, -0.45, z));
    });
  });
  procGeos.push(at(rock(12.5, 15, 7, 0.4), -1, -0.6, 0));
  for (const z of [14.1, 16.3]) procGeos.push(at(rock(1.3, 2.2, 5, z), -10.4, -0.45, z));

  // The centre beacon: a pale mast with a glowing ring that spins.
  procGeos.push(at(cyl(0.5, 0.7, 0.3, 8, "#d8dcf0"), 0, 0.65, 0));
  procGeos.push(at(cyl(0.12, 0.16, 5.2, 6, "#e8ebf7"), 0, 3.1, 0));
  procGeos.push(at(cyl(0.3, 0.3, 0.25, 8, "#9aa0c8"), 0, 5.75, 0));
  glowGeos.push(at(colored(new THREE.IcosahedronGeometry(0.32, 0), "#7ff7ff"), 0, 6.15, 0));

  // The west radio mast: a stack of purple blocks with a dish and a red light on top.
  const purple = "#8a5ad8";
  const purpleDark = "#5d3f9e";
  procGeos.push(at(box(1.9, 0.35, 1.9, "#4b4560"), MAST.x, 0.175, MAST.z));
  procGeos.push(at(box(1.3, 2.4, 1.3, purple), MAST.x, 1.55, MAST.z));
  procGeos.push(at(box(1.5, 0.3, 1.5, purpleDark), MAST.x, 2.9, MAST.z));
  procGeos.push(at(box(1.0, 2.0, 1.0, purple), MAST.x, 4.05, MAST.z));
  procGeos.push(at(box(1.2, 0.25, 1.2, purpleDark), MAST.x, 5.15, MAST.z));
  procGeos.push(at(cyl(0.06, 0.08, 2.4, 5, "#d8dcf0"), MAST.x, 6.4, MAST.z));
  procGeos.push(at(cyl(0.9, 0.15, 0.35, 8, "#e8ebf7"), MAST.x + 0.75, 4.4, MAST.z, 0, 0, -1.0));
  glowGeos.push(at(box(0.9, 0.12, 0.05, "#ff9df5"), MAST.x, 2.0, MAST.z + 0.68));
  glowGeos.push(at(box(0.05, 0.12, 0.9, "#ff9df5"), MAST.x + 0.68, 2.0, MAST.z));

  // Crates.
  for (const c of crates) {
    procGeos.push(at(box(c.s, c.s, c.s, c.c), c.x, c.y + c.s / 2, c.z));
    procGeos.push(at(box(c.s * 1.02, c.s * 0.16, c.s * 1.02, "#5a4a3a"), c.x, c.y + c.s * 0.5, c.z));
  }

  // Glowing rim lights in each side's accent colour, so the edge of the world reads at a glance.
  const rimColor = { n: "#ff7a4a", e: "#4aa8ff", s: "#ffd23f", w: "#c77dff" };
  const solidAt = (col: number, r: number) => col >= 0 && col < 5 && r >= 0 && r < 5 && GRID[r].split(" ")[col] !== ".";
  GRID.forEach((row, r) => {
    row.split(" ").forEach((c, col) => {
      if (c === ".") return;
      const x = (col - 2) * TILE;
      const z = (r - 2) * TILE;
      const yOff = Number(c);
      const h = TILE / 2 + 0.07;
      const lights = (cx: number, cz: number, along: "x" | "z", colr: string) => {
        for (const o of [-1.6, 1.6]) {
          const px = along === "x" ? cx + o : cx;
          const pz = along === "z" ? cz + o : cz;
          glowGeos.push(at(box(along === "x" ? 1.4 : 0.12, 0.12, along === "x" ? 0.12 : 1.4, colr), px, yOff - 0.12, pz));
        }
      };
      if (!solidAt(col, r - 1)) lights(x, z - h, "x", rimColor.n);
      if (!solidAt(col, r + 1)) lights(x, z + h, "x", rimColor.s);
      if (!solidAt(col + 1, r)) lights(x + h, z, "z", rimColor.e);
      if (!solidAt(col - 1, r)) lights(x - h, z, "z", rimColor.w);
    });
  });
  // Deck lip lights facing west (the raised edge).
  for (const z of [-4.4, -0.8]) glowGeos.push(at(box(0.12, 0.12, 1.4, rimColor.e), TILE * 1.5 + 0.07, 0.88, z));

  // Glowing panel inside the north gate.
  glowGeos.push(at(box(2.6, 0.12, 0.2, "#ffb36b"), 0, 3.32, -11.2));
  glowGeos.push(at(box(0.12, 3.2, 0.2, "#ff8a4a"), -1.35, 1.7, -11.2));
  glowGeos.push(at(box(0.12, 3.2, 0.2, "#ff8a4a"), 1.35, 1.7, -11.2));

  // Pickup pad rings.
  for (const pad of PADS) {
    procGeos.push(at(cyl(0.75, 0.8, 0.08, 10, "#e9ecf8"), pad.x, pad.y + 0.04, pad.z));
    glowGeos.push(at(cyl(0.55, 0.55, 0.1, 10, pad.kind === "health" ? "#5df0a8" : "#ffc94a"), pad.x, pad.y + 0.05, pad.z));
  }

  const group = new THREE.Group();
  const kit = new THREE.Mesh(mergeGeometries(kitGeos, false)!, kitMaterial);
  kit.castShadow = kit.receiveShadow = true;
  const proc = new THREE.Mesh(mergeGeometries(procGeos, false)!, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }));
  proc.castShadow = proc.receiveShadow = true;
  const glow = new THREE.Mesh(mergeGeometries(glowGeos, false)!, new THREE.MeshBasicMaterial({ vertexColors: true }));
  group.add(kit, proc, glow);

  const beacon = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.07, 4, 16), new THREE.MeshBasicMaterial({ color: "#7ff7ff" }));
  beacon.position.set(0, 6.15, 0);
  group.add(beacon);

  const mastLight = new THREE.Mesh(new THREE.IcosahedronGeometry(0.16, 0), new THREE.MeshBasicMaterial({ color: "#ff4a6a" }));
  mastLight.position.set(MAST.x, 7.65, MAST.z);
  group.add(mastLight);

  // Kenney's floating cloud cubes around and below the platform.
  const crng = mulberry32(4242);
  const cloudBase: ArenaView["cloudBase"] = [];
  for (let i = 0; i < 22; i++) {
    const a = crng() * Math.PI * 2;
    const d = 24 + crng() * 34;
    cloudBase.push({ x: Math.cos(a) * d, y: -14 + crng() * 26, z: Math.sin(a) * d, s: 1.5 + crng() * 3.5, r: crng() * 6, speed: 0.2 + crng() * 0.5 });
  }
  const clouds = new THREE.InstancedMesh(assets.geo.cloud, kitMaterial, cloudBase.length);
  clouds.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  group.add(clouds);

  return { boxes, view: { group, beacon, clouds, cloudBase, mastLight } };
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();

/** Animates the beacon, mast light and clouds from the episode time (deterministic). */
export function animateArena(v: ArenaView, t: number): void {
  v.beacon.rotation.set(Math.PI / 2 + Math.sin(t * 0.7) * 0.25, t * 1.6, 0);
  v.mastLight.visible = Math.floor(t * 1.5) % 2 === 0;
  v.cloudBase.forEach((c, i) => {
    _p.set(c.x + Math.sin(t * 0.05 * c.speed + i) * 2, c.y + Math.sin(t * 0.3 * c.speed + i * 1.7) * 0.6, c.z);
    _q.setFromEuler(_e.set(c.r, c.r * 0.7 + t * 0.05 * c.speed, c.r * 0.3));
    _s.setScalar(c.s);
    v.clouds.setMatrixAt(i, _m.compose(_p, _q, _s));
  });
  v.clouds.instanceMatrix.needsUpdate = true;
}
