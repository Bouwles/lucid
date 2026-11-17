// Builds the static scene: Kenney track and decoration tiles merged into one mesh, a ground plane,
// a gradient sky, and one sun with a shadow map that follows the player.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { mulberry32 } from "../shared/rng";
import { TILE, type Track } from "./track";

import colormapUrl from "./assets/Textures/colormap.png?url";
import straightUrl from "./assets/track-straight.glb?url";
import cornerUrl from "./assets/track-corner.glb?url";
import finishUrl from "./assets/track-finish.glb?url";
import forestUrl from "./assets/decoration-forest.glb?url";
import tentsUrl from "./assets/decoration-tents.glb?url";
import emptyUrl from "./assets/decoration-empty.glb?url";
import yellowUrl from "./assets/vehicle-truck-yellow.glb?url";
import redUrl from "./assets/vehicle-truck-red.glb?url";
import greenUrl from "./assets/vehicle-truck-green.glb?url";
import purpleUrl from "./assets/vehicle-truck-purple.glb?url";

/** The limited palette everything procedural is drawn from. */
export const PALETTE = {
  skyTop: new THREE.Color("#3f9be8"),
  skyHorizon: new THREE.Color("#cdeeff"),
  // The kit's own grass texel, so the ground plane and the tiles meet without a seam.
  grass: new THREE.Color("#328260"),
  sun: new THREE.Color("#fff4dd"),
  hemiSky: new THREE.Color("#cfe9ff"),
  hemiGround: new THREE.Color("#4f8f5a"),
  finishWash: new THREE.Color("#ffd166"),
  crashWash: new THREE.Color("#3b3570"),
};

export interface Assets {
  straight: THREE.Object3D;
  corner: THREE.Object3D;
  finish: THREE.Object3D;
  forest: THREE.Object3D;
  tents: THREE.Object3D;
  empty: THREE.Object3D;
  cars: { yellow: THREE.Object3D; red: THREE.Object3D; green: THREE.Object3D; purple: THREE.Object3D };
  material: THREE.MeshStandardMaterial;
}

export async function loadAssets(): Promise<Assets> {
  const manager = new THREE.LoadingManager();
  // The models reference "Textures/colormap.png" next to themselves; point that at the bundled file.
  manager.setURLModifier((url) => (url.endsWith("colormap.png") ? colormapUrl : url));
  const loader = new GLTFLoader(manager);
  const load = async (url: string) => (await loader.loadAsync(url)).scene;
  const [straight, corner, finish, forest, tents, empty, yellow, red, green, purple] = await Promise.all(
    [straightUrl, cornerUrl, finishUrl, forestUrl, tentsUrl, emptyUrl, yellowUrl, redUrl, greenUrl, purpleUrl].map(load),
  );
  // Every kit model shares one palette texture, so one material serves the whole scene.
  let map: THREE.Texture | null = null;
  straight.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
    if (m?.map) map = m.map;
  });
  if (map) {
    (map as THREE.Texture).colorSpace = THREE.SRGBColorSpace;
    (map as THREE.Texture).anisotropy = 4;
  }
  const material = new THREE.MeshStandardMaterial({ map, roughness: 0.85, metalness: 0 });
  for (const root of [straight, corner, finish, forest, tents, empty, yellow, red, green, purple]) {
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        (mesh.material as THREE.Material).dispose();
        mesh.material = material;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
    });
  }
  return { straight, corner, finish, forest, tents, empty, cars: { yellow, red, green, purple }, material };
}

type TriFilter = (ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number) => boolean;

/**
 * All mesh geometry under root, transformed by matrix, keeping only position, normal and uv.
 * keep() sees each triangle in the model's own space and can drop it.
 */
function collect(root: THREE.Object3D, matrix: THREE.Matrix4, out: THREE.BufferGeometry[], keep?: TriFilter): void {
  root.updateMatrixWorld(true);
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    let g = new THREE.BufferGeometry();
    for (const name of ["position", "normal", "uv"]) g.setAttribute(name, mesh.geometry.getAttribute(name).clone());
    if (mesh.geometry.index) g.setIndex(mesh.geometry.index.clone());
    g.applyMatrix4(mesh.matrixWorld);
    if (keep) g = filterTriangles(g, keep);
    g.applyMatrix4(matrix);
    out.push(g);
  });
}

function filterTriangles(g: THREE.BufferGeometry, keep: TriFilter): THREE.BufferGeometry {
  const pos = g.getAttribute("position");
  const idx = g.index!;
  const kept: number[] = [];
  for (let t = 0; t < idx.count; t += 3) {
    const a = idx.getX(t);
    const b = idx.getX(t + 1);
    const c = idx.getX(t + 2);
    if (keep(pos.getX(a), pos.getY(a), pos.getZ(a), pos.getX(b), pos.getY(b), pos.getZ(b), pos.getX(c), pos.getY(c), pos.getZ(c))) kept.push(a, b, c);
  }
  g.setIndex(kept);
  return g;
}

// The kit's tiles come with guard rails (and a strip of tarmac under them) along both edges.
// Drift Kit drops them so a car that runs wide ends up on the grass (slow, and off the track)
// instead of bouncing along a wall.
const RAIL_HALF = 4.45;
const noRailStraight: TriFilter = (ax, _ay, _az, bx, _by, _bz, cx) =>
  !(Math.abs(ax) > RAIL_HALF && Math.abs(bx) > RAIL_HALF && Math.abs(cx) > RAIL_HALF && Math.abs(ax) < 5.2 && Math.abs(bx) < 5.2 && Math.abs(cx) < 5.2);
const noRailCorner: TriFilter = (ax, ay, az, bx, by, bz, cx, cy, cz) => {
  if (Math.max(ay, by, cy) > 0.8) return true;
  // The corner pivots on (-5, +5): its rails run from radius 0 to 0.5 and from 9.5 to 10.
  const r = (x: number, z: number) => Math.hypot(x + 5, z - 5);
  const ra = r(ax, az);
  const rb = r(bx, bz);
  const rc = r(cx, cz);
  const inner = ra < 0.56 && rb < 0.56 && rc < 0.56;
  const outer = ra > 9.44 && rb > 9.44 && rc > 9.44 && ra < 10.06 && rb < 10.06 && rc < 10.06;
  return !(inner || outer);
};

/**
 * Solid things in a model (tree trunks, tents, tyre stacks), found from its geometry: vertices
 * between knee and head height are binned on a 0.5 m grid and joined into blobs; each blob becomes
 * a circle. Returns [x, z, radius] in the model's space.
 */
function findObstacles(root: THREE.Object3D, isGreen: (u: number, v: number) => boolean, keep?: TriFilter): Array<[number, number, number]> {
  const parts: THREE.BufferGeometry[] = [];
  collect(root, new THREE.Matrix4(), parts, keep);
  const cell = 0.5;
  const occupied = new Map<string, [number, number]>();
  for (const g of parts) {
    const pos = g.getAttribute("position");
    const uv = g.getAttribute("uv");
    const idx = g.index!;
    for (let t = 0; t < idx.count; t++) {
      const v = idx.getX(t);
      const y = pos.getY(v);
      // Trunk height. Bushes and grass tufts (green) are soft: drive through them.
      if (y < 0.45 || y > 0.8 || isGreen(uv.getX(v), uv.getY(v))) continue;
      const i = Math.floor(pos.getX(v) / cell);
      const j = Math.floor(pos.getZ(v) / cell);
      occupied.set(`${i},${j}`, [i, j]);
    }
    g.dispose();
  }
  const seen = new Set<string>();
  const out: Array<[number, number, number]> = [];
  for (const [key, [i0, j0]] of occupied) {
    if (seen.has(key)) continue;
    let mini = i0;
    let maxi = i0;
    let minj = j0;
    let maxj = j0;
    const stack: Array<[number, number]> = [[i0, j0]];
    seen.add(key);
    while (stack.length) {
      const [i, j] = stack.pop()!;
      mini = Math.min(mini, i);
      maxi = Math.max(maxi, i);
      minj = Math.min(minj, j);
      maxj = Math.max(maxj, j);
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          const k = `${i + di},${j + dj}`;
          if (!seen.has(k) && occupied.has(k)) {
            seen.add(k);
            stack.push([i + di, j + dj]);
          }
        }
      }
    }
    const cx = ((mini + maxi + 1) / 2) * cell;
    const cz = ((minj + maxj + 1) / 2) * cell;
    const rad = (Math.max(maxi - mini, maxj - minj) + 1) * cell * 0.5;
    out.push([cx, cz, Math.max(0.3, rad * 0.85)]);
  }
  return out;
}

/** A test for "this texel of the palette is foliage green". */
function paletteGreenTest(map: THREE.Texture | null): (u: number, v: number) => boolean {
  const img = map?.image as CanvasImageSource & { width: number; height: number } | undefined;
  if (!img) return () => false;
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(img, 0, 0);
  const px = g.getImageData(0, 0, c.width, c.height).data;
  // glTF uv has v pointing down the image (flipY is false for glTF textures).
  return (u, v) => {
    const x = Math.min(c.width - 1, Math.max(0, Math.floor(u * c.width)));
    const y = Math.min(c.height - 1, Math.max(0, Math.floor(v * c.height)));
    const k = (y * c.width + x) * 4;
    return px[k + 1] > px[k] + 40 && px[k + 1] > px[k + 2] + 15;
  };
}

function tileMatrix(x: number, z: number, rotY: number): THREE.Matrix4 {
  return new THREE.Matrix4().makeRotationY(rotY).setPosition(x, 0, z);
}

export interface World {
  scene: THREE.Scene;
  sun: THREE.DirectionalLight;
  statics: THREE.Mesh;
  /** Tree trunks, tents and tyre stacks that the cars bump into: x, z, radius. */
  obstacles: Array<[number, number, number]>;
}

export function buildWorld(track: Track, assets: Assets, recording: boolean): World {
  const scene = new THREE.Scene();
  scene.background = PALETTE.skyHorizon.clone();
  scene.fog = new THREE.Fog(PALETTE.skyHorizon, 60, 150);

  const parts: THREE.BufferGeometry[] = [];
  const obstacles: Array<[number, number, number]> = [];
  const isGreen = paletteGreenTest(assets.material.map);
  const solid = new Map<THREE.Object3D, Array<[number, number, number]>>();
  const place = (src: THREE.Object3D, x: number, z: number, rot: number, keep?: TriFilter) => {
    const m = tileMatrix(x, z, rot);
    collect(src, m, parts, keep);
    let local = solid.get(src);
    if (!local) {
      local = findObstacles(src, isGreen, keep);
      solid.set(src, local);
    }
    const v = new THREE.Vector3();
    for (const [ox, oz, r] of local) {
      v.set(ox, 0, oz).applyMatrix4(m);
      obstacles.push([v.x, v.z, r]);
    }
  };
  for (const t of track.tiles) {
    if (t.kind === "corner") place(assets.corner, t.i * TILE, t.j * TILE, t.rotY, noRailCorner);
    else place(t.kind === "finish" ? assets.finish : assets.straight, t.i * TILE, t.j * TILE, t.rotY, noRailStraight);
  }

  // Decorations: the scenery is fixed (not per-episode), so it uses its own fixed seed.
  const rng = mulberry32(20261116);
  for (const [i, j] of track.decorationCells(3)) {
    // Mostly open grass right next to the track, so running wide is a slow detour, not a crash.
    let touching = false;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) if (track.hasCell(i + di, j + dj)) touching = true;
    const r = rng();
    const rot = Math.floor(rng() * 4) * (Math.PI / 2);
    let src: THREE.Object3D;
    if (touching) src = r < 0.8 ? assets.empty : assets.tents;
    else src = r < 0.75 ? assets.forest : r < 0.88 ? assets.tents : assets.empty;
    place(src, i * TILE, j * TILE, rot);
  }
  const merged = mergeGeometries(parts, false);
  for (const g of parts) g.dispose();
  merged.computeBoundingSphere();
  const statics = new THREE.Mesh(merged, assets.material);
  statics.castShadow = true;
  statics.receiveShadow = true;
  statics.matrixAutoUpdate = false;
  scene.add(statics);

  // Ground: a big plane just under the tiles in the kit's grass colour.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(800, 800).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: PALETTE.grass, roughness: 1 }),
  );
  ground.position.y = -0.02;
  ground.receiveShadow = true;
  ground.matrixAutoUpdate = false;
  ground.updateMatrix();
  scene.add(ground);

  scene.add(makeSky());

  scene.add(new THREE.HemisphereLight(PALETTE.hemiSky, PALETTE.hemiGround, 1.35));
  const sun = new THREE.DirectionalLight(PALETTE.sun, 2.4);
  sun.castShadow = true;
  const size = recording ? 1024 : 2048;
  sun.shadow.mapSize.set(size, size);
  const cam = sun.shadow.camera;
  cam.left = cam.bottom = -26;
  cam.right = cam.top = 26;
  cam.near = 1;
  cam.far = 120;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;
  sun.shadow.radius = 3;
  scene.add(sun);
  scene.add(sun.target);

  return { scene, sun, statics, obstacles };
}

/** A vertical gradient dome. Drawn first, never fogged, follows the camera. */
function makeSky(): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    uniforms: { top: { value: PALETTE.skyTop }, horizon: { value: PALETTE.skyHorizon } },
    vertexShader: `varying vec3 vDir; void main() { vDir = normalize(position); vec4 p = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * p; gl_Position.z = gl_Position.w; }`,
    fragmentShader: `uniform vec3 top; uniform vec3 horizon; varying vec3 vDir;
      void main() { float h = clamp(vDir.y, 0.0, 1.0); vec3 c = mix(horizon, top, pow(h, 0.55)); gl_FragColor = vec4(c, 1.0);
      #include <colorspace_fragment>
      }`,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(200, 24, 12), mat);
  sky.name = "sky";
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  return sky;
}

/** Keep the shadow frustum centred on the player, snapped to shadow-map texels so edges don't crawl. */
export function placeSun(sun: THREE.DirectionalLight, x: number, z: number): void {
  const dir = new THREE.Vector3(-0.55, 1, -0.35).normalize();
  const texel = (sun.shadow.camera.right - sun.shadow.camera.left) / sun.shadow.mapSize.x;
  // Snap in light space: project the focus point onto the light's right/up axes.
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
  const up = new THREE.Vector3().crossVectors(dir, right).normalize();
  const focus = new THREE.Vector3(x, 0, z);
  const a = Math.round(focus.dot(right) / texel) * texel;
  const b = Math.round(focus.dot(up) / texel) * texel;
  const c = focus.dot(dir);
  const snapped = right.multiplyScalar(a).add(up.multiplyScalar(b)).add(dir.clone().multiplyScalar(c));
  sun.target.position.copy(snapped);
  sun.position.copy(snapped).addScaledVector(dir, 60);
  sun.target.updateMatrixWorld();
}
