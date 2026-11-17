// Layered backdrop for depth: a sky gradient, far and near hill silhouettes, floating kit islands
// and drifting clouds. Layers sit at different depths behind the play plane, so the perspective
// camera gives the parallax for free. Hills and sky are procedural geometry; islands and clouds are
// kit models. All motion is a function of the tick, so recording mode is repeatable.

import * as THREE from "three";
import { withLook, type Assets } from "./assets";
import { hash } from "./stage";
import type { Level } from "./level";

function mix(a: number, b: number, t: number): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), t);
}

/** A strip of rounded hills: top edge from a sum of sines, filled down to `bottom`. */
function hills(x0: number, x1: number, base: number, amp: number, bottom: number, seed: number, top: THREE.Color, low: THREE.Color): THREE.BufferGeometry {
  const pos: number[] = [];
  const col: number[] = [];
  const idx: number[] = [];
  const step = 1.5;
  let n = 0;
  for (let x = x0; x <= x1; x += step) {
    const u = x / 18;
    const h = base + amp * (0.55 * Math.sin(u + seed) + 0.3 * Math.sin(u * 2.3 + seed * 1.7) + 0.15 * Math.sin(u * 5.1 + seed * 3.1));
    pos.push(x, h, 0, x, bottom, 0);
    col.push(top.r, top.g, top.b, low.r, low.g, low.b);
    if (n > 0) {
      const a = (n - 1) * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    n++;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  return g;
}

export class Backdrop {
  readonly group = new THREE.Group();
  readonly sky: THREE.Mesh;
  private clouds: THREE.InstancedMesh;
  private cloudBase: { x: number; y: number; z: number; s: number; speed: number }[] = [];
  private islands: THREE.InstancedMesh;
  private islandBase: { x: number; y: number; z: number; s: number; ph: number }[] = [];
  readonly fogColor: THREE.Color;
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();

  constructor(lv: Level, assets: Assets, camera: THREE.Camera) {
    const [skyTop, skyLow] = lv.sky;
    this.fogColor = new THREE.Color(skyLow);

    // Sky: a big gradient card that travels with the camera.
    const skyGeo = new THREE.PlaneGeometry(900, 420, 1, 8);
    const sc: number[] = [];
    const p = skyGeo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const t = THREE.MathUtils.clamp((p.getY(i) + 70) / 160, 0, 1);
      const c = mix(skyLow, skyTop, Math.pow(t, 0.7));
      sc.push(c.r, c.g, c.b);
    }
    skyGeo.setAttribute("color", new THREE.Float32BufferAttribute(sc, 3));
    this.sky = new THREE.Mesh(skyGeo, withLook(new THREE.MeshBasicMaterial({ vertexColors: true, fog: false, depthWrite: false })));
    this.sky.position.set(0, 0, -330);
    this.sky.renderOrder = -10;
    camera.add(this.sky);

    const x0 = -120;
    const x1 = lv.w + 120;
    const hillMat = withLook(new THREE.MeshBasicMaterial({ vertexColors: true, fog: true }));
    // Far hills: pale, close to the horizon colour.
    const farTop = mix(skyLow, skyTop, 0.55).multiplyScalar(0.93);
    const far = new THREE.Mesh(hills(x0 - 60, x1 + 60, -2, 9, -60, 1.3, farTop, mix(skyLow, skyTop, 0.2)), hillMat);
    far.position.z = -95;
    // Near hills: the kit's soft violet and mint, darker than the sky so the play area stands out.
    const nearTop = mix(0x9b8ae0, skyTop, 0.35);
    const near = new THREE.Mesh(hills(x0, x1, -3.5, 5, -40, 4.1, nearTop, mix(0x9b8ae0, skyLow, 0.5)), hillMat);
    near.position.z = -48;
    this.group.add(far, near);

    // Floating islands between the hills and the play plane.
    for (let i = 0, x = -20; x < lv.w + 30; i++) {
      x += 14 + 18 * hash(i, lv.index, 7);
      this.islandBase.push({
        x, y: -6 + 10 * hash(i, lv.index, 8), z: -62 - 20 * hash(i, lv.index, 9), s: 1.3 + 0.9 * hash(i, lv.index, 10), ph: 6.3 * hash(i, lv.index, 11),
      });
    }
    this.islands = new THREE.InstancedMesh(assets.geo.island, assets.kit, this.islandBase.length);
    this.islands.frustumCulled = false;
    this.group.add(this.islands);

    for (let i = 0, x = -60; x < lv.w + 80; i++) {
      x += 10 + 16 * hash(i, lv.index, 3);
      this.cloudBase.push({
        x, y: 9 + 12 * hash(i, lv.index, 4), z: -28 - 30 * hash(i, lv.index, 5), s: 2.5 + 3 * hash(i, lv.index, 6), speed: 0.004 + 0.006 * hash(i, lv.index, 12),
      });
    }
    // Clouds glow a little so they stay white when lit from the side.
    const cloudMat = withLook(new THREE.MeshLambertMaterial({ map: assets.colormap, emissive: 0xffffff, emissiveIntensity: 0.45 }));
    this.clouds = new THREE.InstancedMesh(assets.geo.cloud, cloudMat, this.cloudBase.length);
    this.clouds.frustumCulled = false;
    this.group.add(this.clouds);
    this.update(0);
  }

  update(t: number): void {
    const { m, q } = this;
    const v = new THREE.Vector3();
    const s = new THREE.Vector3();
    this.islandBase.forEach((b, i) => {
      // Tipped towards the camera so the grassy tops read.
      q.setFromEuler(new THREE.Euler(0.32, b.ph, 0, "XYZ"));
      m.compose(v.set(b.x, b.y + Math.sin(t * 0.03 + b.ph) * 0.25, b.z), q, s.set(b.s, b.s, b.s));
      this.islands.setMatrixAt(i, m);
    });
    this.islands.instanceMatrix.needsUpdate = true;
    q.identity();
    this.cloudBase.forEach((b, i) => {
      m.compose(v.set(b.x + t * b.speed, b.y, b.z), q, s.set(b.s * 1.6, b.s * 0.8, b.s));
      this.clouds.setMatrixAt(i, m);
    });
    this.clouds.instanceMatrix.needsUpdate = true;
  }

  dispose(camera: THREE.Camera): void {
    camera.remove(this.sky);
  }
}
