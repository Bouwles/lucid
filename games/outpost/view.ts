// The 3D scene: sky, lights, arena, robots, orbs and pickups. Reads the simulation; never changes it.

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { animateArena, type ArenaView, buildArena } from "./arena";
import type { Assets } from "./assets";
import { chargeLevel, type EnemySim, MAX_ENEMIES } from "./enemies";
import type { PickupState } from "./pickups";

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = p.xyww;
}`;

const SKY_FRAG = /* glsl */ `
uniform vec3 uTop;
uniform vec3 uMid;
uniform vec3 uHorizon;
uniform vec3 uBottom;
uniform vec3 uSunDir;
uniform vec3 uSun;
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 0.28, h));
  c = mix(c, uTop, smoothstep(0.28, 0.95, h));
  c = mix(c, uBottom, smoothstep(0.0, 0.5, -h));
  float s = max(dot(d, uSunDir), 0.0);
  c += uSun * (pow(s, 24.0) * 0.6 + pow(s, 400.0) * 1.5);
  gl_FragColor = vec4(c, 1.0);
  #include <colorspace_fragment>
}`;

export const SKY = {
  top: "#6c6fd6",
  mid: "#b49ce8",
  horizon: "#f4d3e6",
  bottom: "#a993d6",
  fog: "#e9cde6",
};

const SUN_DIR = new THREE.Vector3(-0.55, 0.62, 0.56).normalize();

function coloredBox(w: number, h: number, d: number, c: string, x = 0, y = 0, z = 0): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d).toNonIndexed();
  g.deleteAttribute("uv");
  const col = new THREE.Color(c);
  const arr = new Float32Array(g.getAttribute("position").count * 3);
  for (let i = 0; i < arr.length; i += 3) arr.set([col.r, col.g, col.b], i);
  g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
  return g.translate(x, y, z);
}

function coloredCyl(r: number, h: number, c: string, x: number, y: number, z: number): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r, r, h, 6).toNonIndexed();
  g.deleteAttribute("uv");
  const col = new THREE.Color(c);
  const arr = new Float32Array(g.getAttribute("position").count * 3);
  for (let i = 0; i < arr.length; i += 3) arr.set([col.r, col.g, col.b], i);
  g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
  return g.translate(x, y, z);
}

function healthGeometry(): THREE.BufferGeometry {
  const w = "#ffffff";
  return mergeGeometries([
    coloredBox(0.5, 0.5, 0.5, "#35d08a"),
    coloredBox(0.34, 0.11, 0.54, w),
    coloredBox(0.11, 0.34, 0.54, w),
    coloredBox(0.54, 0.11, 0.34, w),
    coloredBox(0.54, 0.34, 0.11, w),
  ])!;
}

function ammoGeometry(): THREE.BufferGeometry {
  return mergeGeometries([
    coloredBox(0.62, 0.34, 0.4, "#ffbf2e"),
    coloredBox(0.64, 0.08, 0.42, "#5a4632", 0, 0.02, 0),
    coloredCyl(0.06, 0.24, "#e98a3c", -0.17, 0.27, 0),
    coloredCyl(0.06, 0.24, "#e98a3c", 0, 0.27, 0),
    coloredCyl(0.06, 0.24, "#e98a3c", 0.17, 0.27, 0),
  ])!;
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler(0, 0, 0, "YXZ");
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _c = new THREE.Color();
const ZERO = new THREE.Matrix4().makeScale(0, 0, 0);

export interface EyeView {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
}

export class World {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly kitMaterial: THREE.MeshLambertMaterial;
  readonly arena: { boxes: import("./arena").Box[]; view: ArenaView };
  private sky: THREE.Mesh;
  private robots: THREE.InstancedMesh;
  private orbCore: THREE.InstancedMesh;
  private orbHalo: THREE.InstancedMesh;
  private health: THREE.InstancedMesh;
  private ammo: THREE.InstancedMesh;
  private shadows: THREE.InstancedMesh;

  constructor(assets: Assets) {
    this.camera = new THREE.PerspectiveCamera(74, 1, 0.05, 400);
    this.camera.rotation.order = "YXZ";
    this.kitMaterial = new THREE.MeshLambertMaterial({ map: assets.colormap });
    const fogColor = new THREE.Color(SKY.fog);
    this.scene.fog = new THREE.Fog(fogColor, 32, 110);
    this.scene.background = fogColor;

    this.sky = new THREE.Mesh(
      new THREE.SphereGeometry(300, 32, 16),
      new THREE.ShaderMaterial({
        vertexShader: SKY_VERT,
        fragmentShader: SKY_FRAG,
        uniforms: {
          uTop: { value: new THREE.Color(SKY.top) },
          uMid: { value: new THREE.Color(SKY.mid) },
          uHorizon: { value: new THREE.Color(SKY.horizon) },
          uBottom: { value: new THREE.Color(SKY.bottom) },
          uSunDir: { value: SUN_DIR },
          uSun: { value: new THREE.Color("#fff3d6") },
        },
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
      }),
    );
    this.sky.renderOrder = -1;
    this.sky.frustumCulled = false;
    this.scene.add(this.sky);

    this.scene.add(new THREE.HemisphereLight("#e6deff", "#7b6a9c", 1.9));
    const sun = new THREE.DirectionalLight("#fff0da", 2.3);
    sun.position.copy(SUN_DIR).multiplyScalar(30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    const sc = sun.shadow.camera;
    sc.left = -19;
    sc.right = 19;
    sc.top = 19;
    sc.bottom = -19;
    sc.near = 5;
    sc.far = 70;
    sun.shadow.bias = -0.0008;
    sun.shadow.normalBias = 0.03;
    this.scene.add(sun, sun.target);

    this.arena = buildArena(assets, this.kitMaterial);
    this.scene.add(this.arena.view.group);

    // Robots: one instanced draw for all of them. instanceColor carries the hit flash and the
    // warning glow before they fire.
    this.robots = new THREE.InstancedMesh(assets.geo.enemy, this.kitMaterial.clone(), MAX_ENEMIES);
    this.robots.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.robots.setColorAt(0, new THREE.Color(1, 1, 1));
    this.robots.castShadow = true;
    this.robots.frustumCulled = false;

    this.orbCore = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.13, 1), new THREE.MeshBasicMaterial({ color: "#fff4e0" }), 24);
    this.orbHalo = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.26, 1),
      new THREE.MeshBasicMaterial({ color: "#ff5a3c", transparent: true, opacity: 0.75, depthWrite: false }),
      24,
    );
    for (const m of [this.orbCore, this.orbHalo]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.frustumCulled = false;
    }

    const pickMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
    this.health = new THREE.InstancedMesh(healthGeometry(), pickMat, 8);
    this.ammo = new THREE.InstancedMesh(ammoGeometry(), pickMat, 8);
    for (const m of [this.health, this.ammo]) {
      m.castShadow = true;
      m.frustumCulled = false;
    }

    // Soft blob shadows under the robots, so their height reads even in flat light.
    this.shadows = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: assets.sprites.shadow, transparent: true, depthWrite: false, opacity: 0.35, color: "#2a2050" }),
      MAX_ENEMIES,
    );
    this.shadows.frustumCulled = false;
    this.scene.add(this.robots, this.orbCore, this.orbHalo, this.health, this.ammo, this.shadows);
  }

  resize(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  /** Places everything for a frame. lerp(a, b) is applied by the caller where needed. */
  update(eye: EyeView, sim: EnemySim, pickups: PickupState[], t: number, alpha: number, ground: (x: number, z: number, y: number) => number): void {
    this.camera.position.set(eye.x, eye.y, eye.z);
    this.camera.rotation.set(eye.pitch, -eye.yaw, eye.roll);
    this.sky.position.copy(this.camera.position);

    animateArena(this.arena.view, t);

    const L = (a: number, b: number) => a + (b - a) * alpha;
    sim.enemies.forEach((e, i) => {
      if (!e.alive) {
        this.robots.setMatrixAt(i, ZERO);
        this.shadows.setMatrixAt(i, ZERO);
        return;
      }
      const x = L(e.px, e.x);
      const y = L(e.py, e.y);
      const z = L(e.pz, e.z);
      const dx = eye.x - x;
      const dz = eye.z - z;
      const dy = eye.y - y;
      const yaw = Math.atan2(dx, dz);
      const pitch = -Math.atan2(dy, Math.hypot(dx, dz)) * 0.5;
      // Bank into turns a little.
      const roll = Math.max(-0.4, Math.min(0.4, (e.vx * Math.cos(yaw) - e.vz * Math.sin(yaw)) * -0.08));
      _q.setFromEuler(_e.set(pitch, yaw, roll));
      this.robots.setMatrixAt(i, _m.compose(_p.set(x, y, z), _q, _s.setScalar(1.35)));
      const charge = chargeLevel(e);
      if (e.flash > 0) _c.setRGB(3, 3, 3);
      else _c.setRGB(1 + charge * 1.2, 1 + charge * 0.35, 1 - charge * 0.3);
      this.robots.setColorAt(i, _c);
      const g = ground(x, z, y);
      if (g > -50) {
        const hgt = y - g;
        const s = Math.max(0.4, 1.6 - hgt * 0.12);
        _q.identity();
        this.shadows.setMatrixAt(i, _m.compose(_p.set(x, g + 0.02, z), _q, _s.set(s, 1, s)));
      } else this.shadows.setMatrixAt(i, ZERO);
    });
    this.robots.instanceMatrix.needsUpdate = true;
    this.robots.instanceColor!.needsUpdate = true;
    this.shadows.instanceMatrix.needsUpdate = true;

    sim.orbs.forEach((o, i) => {
      if (!o.alive) {
        this.orbCore.setMatrixAt(i, ZERO);
        this.orbHalo.setMatrixAt(i, ZERO);
        return;
      }
      _q.identity();
      _m.compose(_p.set(L(o.px, o.x), L(o.py, o.y), L(o.pz, o.z)), _q, _s.setScalar(1));
      this.orbCore.setMatrixAt(i, _m);
      this.orbHalo.setMatrixAt(i, _m);
    });
    this.orbCore.instanceMatrix.needsUpdate = true;
    this.orbHalo.instanceMatrix.needsUpdate = true;

    let nh = 0;
    let na = 0;
    pickups.forEach((it, i) => {
      const { pad } = it;
      const back = it.wait > 0 ? Math.max(0, 1 - it.wait / 15) : 1;
      const mesh = pad.kind === "health" ? this.health : this.ammo;
      const idx = pad.kind === "health" ? nh++ : na++;
      if (back <= 0) {
        mesh.setMatrixAt(idx, ZERO);
        return;
      }
      _q.setFromEuler(_e.set(0, t * 1.8 + i, 0));
      _m.compose(_p.set(pad.x, pad.y + 0.62 + Math.sin(t * 2.5 + i) * 0.08, pad.z), _q, _s.setScalar(back));
      mesh.setMatrixAt(idx, _m);
    });
    this.health.count = nh;
    this.ammo.count = na;
    this.health.instanceMatrix.needsUpdate = true;
    this.ammo.instanceMatrix.needsUpdate = true;
  }
}
