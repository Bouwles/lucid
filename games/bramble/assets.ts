// Loads Kenney's Starter Kit 3D Platformer models (CC0, see CREDITS.md) and builds the shared materials.

import * as THREE from "three";
import { GLTFLoader, type GLTF } from "three/addons/loaders/GLTFLoader.js";
import colormapUrl from "./assets/Textures/colormap.png?url";
import blobUrl from "./assets/blob_shadow.png?url";
import particleUrl from "./assets/particle.png?url";
import characterUrl from "./assets/character.glb?url";
import brickUrl from "./assets/brick.glb?url";
import blockCoinUrl from "./assets/block-coin.glb?url";
import coinUrl from "./assets/coin.glb?url";
import flagUrl from "./assets/flag.glb?url";
import cloudUrl from "./assets/cloud.glb?url";
import grassUrl from "./assets/grass.glb?url";
import grassSmallUrl from "./assets/grass-small.glb?url";
import platformUrl from "./assets/platform.glb?url";
import platformMediumUrl from "./assets/platform-medium.glb?url";
import platformFallingUrl from "./assets/platform-falling.glb?url";
import islandUrl from "./assets/platform-grass-large-round.glb?url";
import dustUrl from "./assets/dust.glb?url";

const MODELS = {
  brick: brickUrl,
  blockCoin: blockCoinUrl,
  coin: coinUrl,
  flag: flagUrl,
  cloud: cloudUrl,
  grass: grassUrl,
  grassSmall: grassSmallUrl,
  platform: platformUrl,
  platformMedium: platformMediumUrl,
  platformFalling: platformFallingUrl,
  island: islandUrl,
  dust: dustUrl,
} as const;

export type ModelName = keyof typeof MODELS;

/**
 * Uniforms shared by every material, for the game-over looks: desat fades the scene to grey,
 * flash fades it towards warm white (the win).
 */
export const LOOK = {
  uDesat: { value: 0 },
  uFlash: { value: 0 },
};

/** Adds the game-over looks to a built-in material. */
export function withLook<M extends THREE.Material>(m: M): M {
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uDesat = LOOK.uDesat;
    shader.uniforms.uFlash = LOOK.uFlash;
    shader.fragmentShader =
      "uniform float uDesat;\nuniform float uFlash;\n" +
      shader.fragmentShader.replace(
        "#include <dithering_fragment>",
        `#include <dithering_fragment>
        float lum = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));
        gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(lum) * vec3(0.86, 0.86, 0.95), uDesat);
        gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(1.0, 0.96, 0.86), uFlash);`,
      );
  };
  m.customProgramCacheKey = () => "bramble-look";
  return m;
}

export interface Assets {
  character: GLTF;
  geo: Record<ModelName, THREE.BufferGeometry>;
  colormap: THREE.Texture;
  blob: THREE.Texture;
  particle: THREE.Texture;
  /** The kit's palette texture with soft lighting: used by every model. */
  kit: THREE.MeshLambertMaterial;
}

export async function loadAssets(): Promise<Assets> {
  const manager = new THREE.LoadingManager();
  // The models point at "Textures/colormap.png" next to them. Bundled files get hashed names, so
  // send that request to the bundled texture instead.
  manager.setURLModifier((url) => (url.endsWith("colormap.png") ? colormapUrl : url));
  const loader = new GLTFLoader(manager);
  const texLoader = new THREE.TextureLoader();

  const [character, colormap, blob, particle, ...models] = await Promise.all([
    loader.loadAsync(characterUrl),
    texLoader.loadAsync(colormapUrl),
    texLoader.loadAsync(blobUrl),
    texLoader.loadAsync(particleUrl),
    ...Object.values(MODELS).map((u) => loader.loadAsync(u)),
  ]);

  colormap.colorSpace = THREE.SRGBColorSpace;
  // The texture is a palette of flat colour strips: no mipmaps, so small renders don't blend strips.
  colormap.generateMipmaps = false;
  colormap.minFilter = THREE.LinearFilter;
  colormap.magFilter = THREE.LinearFilter;
  colormap.flipY = false;
  blob.colorSpace = THREE.SRGBColorSpace;
  particle.colorSpace = THREE.SRGBColorSpace;

  const kit = withLook(new THREE.MeshLambertMaterial({ map: colormap }));

  const geo = {} as Record<ModelName, THREE.BufferGeometry>;
  Object.keys(MODELS).forEach((name, i) => {
    geo[name as ModelName] = firstGeometry(models[i]);
  });

  // Every character part uses the shared kit material.
  character.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh) mesh.material = kit;
  });

  return { character, geo, colormap, blob, particle, kit };
}

/** The single mesh of a one-mesh kit model, with its node transform baked in. */
function firstGeometry(g: GLTF): THREE.BufferGeometry {
  g.scene.updateMatrixWorld(true);
  let out: THREE.BufferGeometry | null = null;
  g.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!out && mesh.isMesh) out = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
  });
  if (!out) throw new Error("model has no mesh");
  return out;
}
