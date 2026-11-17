// Loads the Kenney FPS kit models, sprites and sounds used by Outpost (see CREDITS.md).
// Every model shares one palette texture, so the geometry is pulled out of each glTF and drawn
// with a single shared material. That keeps the arena down to a handful of draw calls.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

import blasterGlb from "./assets/blaster.glb?url";
import cloudGlb from "./assets/cloud.glb?url";
import enemyGlb from "./assets/enemy-flying.glb?url";
import grassSmallGlb from "./assets/grass-small.glb?url";
import grassGlb from "./assets/grass.glb?url";
import platformLargeGlb from "./assets/platform-large-grass.glb?url";
import platformGlb from "./assets/platform.glb?url";
import wallHighGlb from "./assets/wall-high.glb?url";
import wallLowGlb from "./assets/wall-low.glb?url";
import colormapPng from "./assets/Textures/colormap.png?url";
import blobShadowPng from "./assets/blob_shadow.png?url";
import burstPng from "./assets/burst.png?url";
import crosshairPng from "./assets/crosshair.png?url";
import hitPng from "./assets/hit.png?url";
import blasterOgg from "./assets/blaster.ogg?url";
import enemyAttackOgg from "./assets/enemy_attack.ogg?url";
import enemyDestroyOgg from "./assets/enemy_destroy.ogg?url";
import enemyHurtOgg from "./assets/enemy_hurt.ogg?url";
import jumpOgg from "./assets/jump_a.ogg?url";
import landOgg from "./assets/land.ogg?url";
import pickupOgg from "./assets/weapon_change.ogg?url";

const MODELS = {
  platformLarge: platformLargeGlb,
  platform: platformGlb,
  wallHigh: wallHighGlb,
  wallLow: wallLowGlb,
  grass: grassGlb,
  grassSmall: grassSmallGlb,
  cloud: cloudGlb,
  enemy: enemyGlb,
  blaster: blasterGlb,
} as const;

export type ModelName = keyof typeof MODELS;

export const SOUNDS = {
  blaster: blasterOgg,
  enemyAttack: enemyAttackOgg,
  enemyDestroy: enemyDestroyOgg,
  enemyHurt: enemyHurtOgg,
  jump: jumpOgg,
  land: landOgg,
  pickup: pickupOgg,
} as const;

export type SoundName = keyof typeof SOUNDS;

export interface Assets {
  /** Geometry of each model in its own space (position, normal, uv), all meshes baked together. */
  geo: Record<ModelName, THREE.BufferGeometry>;
  /** The kit's shared palette texture. */
  colormap: THREE.Texture;
  sprites: { hit: THREE.Texture; burst: THREE.Texture; crosshair: THREE.Texture; shadow: THREE.Texture };
}

/** Bakes every mesh under root into one non-indexed geometry with position, normal and uv. */
function bake(root: THREE.Object3D): THREE.BufferGeometry {
  root.updateMatrixWorld(true);
  const parts: THREE.BufferGeometry[] = [];
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const src = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry;
    const g = new THREE.BufferGeometry();
    for (const name of ["position", "normal", "uv"]) {
      const a = src.getAttribute(name);
      if (!a) throw new Error(`outpost: model mesh without ${name}`);
      g.setAttribute(name, (a as THREE.BufferAttribute).clone());
    }
    g.applyMatrix4(mesh.matrixWorld);
    parts.push(g);
  });
  const merged = parts.length === 1 ? parts[0] : mergeGeometries(parts, false);
  if (!merged) throw new Error("outpost: could not merge model meshes");
  merged.computeBoundingBox();
  return merged;
}

function loadTexture(loader: THREE.TextureLoader, url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (t) => {
        t.colorSpace = THREE.SRGBColorSpace;
        resolve(t);
      },
      undefined,
      reject,
    );
  });
}

export async function loadAssets(): Promise<Assets> {
  // The .glb files point at Textures/colormap.png next to them. After bundling the file names
  // carry a hash, so send that request to the bundled URL instead.
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => (/colormap\.png$/.test(url) ? colormapPng : url));
  const gltf = new GLTFLoader(manager);
  const tex = new THREE.TextureLoader();

  const names = Object.keys(MODELS) as ModelName[];
  const [scenes, colormap, hit, burst, crosshair, shadow] = await Promise.all([
    Promise.all(names.map((n) => gltf.loadAsync(MODELS[n]).then((g) => g.scene))),
    loadTexture(tex, colormapPng),
    loadTexture(tex, hitPng),
    loadTexture(tex, burstPng),
    loadTexture(tex, crosshairPng),
    loadTexture(tex, blobShadowPng),
  ]);

  const geo = {} as Record<ModelName, THREE.BufferGeometry>;
  names.forEach((n, i) => (geo[n] = bake(scenes[i])));

  // glTF UVs have v pointing down. The palette has hard colour cells, so no mipmaps: at 128 px,
  // mip levels would blend neighbouring cells into mud.
  colormap.flipY = false;
  colormap.generateMipmaps = false;
  colormap.minFilter = THREE.LinearFilter;
  colormap.magFilter = THREE.LinearFilter;
  colormap.needsUpdate = true;

  return { geo, colormap, sprites: { hit, burst, crosshair, shadow } };
}

/**
 * Recolours a palette-mapped geometry by moving the UVs of its saturated colours (the top half of
 * the Kenney colormap) along the palette. cells is a whole number of 32 px palette cells:
 * from the kit's red-orange, +2 is blue, +4 light blue, +6 purple and -4 yellow.
 */
export function shiftPalette(g: THREE.BufferGeometry, cells: number): THREE.BufferGeometry {
  if (!cells) return g;
  const uv = g.getAttribute("uv") as THREE.BufferAttribute;
  for (let i = 0; i < uv.count; i++) {
    if (uv.getY(i) < 0.75) uv.setX(i, uv.getX(i) + cells / 16);
  }
  uv.needsUpdate = true;
  return g;
}
