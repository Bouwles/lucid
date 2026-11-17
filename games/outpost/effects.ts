// Feedback: the blaster viewmodel, muzzle flash, hit sparks, robot debris, screen tint and sound.
//
// Nothing here touches the simulation. Normal mode draws its random bits from a separate
// generator, so the game plays identically in both modes. Recording mode keeps only
// deterministic, minimal effects (no shake, no random particles, a fixed muzzle flash).

import * as THREE from "three";
import type { Rng } from "../shared/rng";
import { type Box, groundHeight } from "./arena";
import { type Assets, SOUNDS, type SoundName } from "./assets";
import { RELOAD_TICKS } from "./weapons";

const MAX_SPARKS = 64;
const MAX_DEBRIS = 48;
const MAX_FLASHES = 8;

interface Spark {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  life: number;
  max: number;
}

interface Debris extends Spark {
  rx: number;
  ry: number;
  spin: number;
  size: number;
  color: THREE.Color;
}

interface Flash {
  sprite: THREE.Sprite;
  age: number;
  life: number;
  size: number;
}

// ---------------------------------------------------------------- sound

export class Sfx {
  private ctx: AudioContext | null = null;
  private buffers: Partial<Record<SoundName, AudioBuffer>> = {};
  private master: GainNode | null = null;
  private unlock = () => void this.ctx?.resume();

  /** Creates the AudioContext and starts decoding; called on the first step that wants sound. */
  ensure(): void {
    if (this.ctx) return;
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) return;
    this.ctx = new Ctx();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.55;
    this.master.connect(this.ctx.destination);
    // Browsers start audio suspended until the player interacts.
    window.addEventListener("pointerdown", this.unlock);
    window.addEventListener("keydown", this.unlock);
    for (const [name, url] of Object.entries(SOUNDS) as [SoundName, string][]) {
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((data) => this.ctx!.decodeAudioData(data))
        .then((buf) => (this.buffers[name] = buf))
        .catch(() => {
          /* a sound that can't be decoded is skipped */
        });
    }
  }

  play(name: SoundName, gain = 1, rate = 1): void {
    const ctx = this.ctx;
    const buf = this.buffers[name];
    if (!ctx || !buf || !this.master || ctx.state !== "running") return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = rate;
    const g = ctx.createGain();
    g.gain.value = gain;
    src.connect(g).connect(this.master);
    src.start();
  }

  dispose(): void {
    window.removeEventListener("pointerdown", this.unlock);
    window.removeEventListener("keydown", this.unlock);
    void this.ctx?.close();
    this.ctx = null;
  }
}

// ---------------------------------------------------------------- overlay (tint + crosshair)

const TINT_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const TINT_FRAG = /* glsl */ `
uniform float uHurt;
uniform float uDeath;
varying vec2 vUv;
void main() {
  vec2 d = vUv - 0.5;
  float edge = smoothstep(0.25, 0.75, length(d) * 1.4);
  vec3 red = vec3(0.85, 0.08, 0.12);
  float a = uHurt * edge * 0.7 + uDeath * (0.35 + 0.5 * edge);
  gl_FragColor = vec4(red * (1.0 - uDeath * 0.45 * edge), clamp(a, 0.0, 0.9));
}`;

export class Overlay {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  private tint: THREE.ShaderMaterial;
  private crosshair: THREE.Sprite;

  constructor(assets: Assets) {
    this.tint = new THREE.ShaderMaterial({
      vertexShader: TINT_VERT,
      fragmentShader: TINT_FRAG,
      uniforms: { uHurt: { value: 0 }, uDeath: { value: 0 } },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.tint);
    quad.frustumCulled = false;
    quad.renderOrder = 2;
    this.crosshair = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: assets.sprites.crosshair, depthTest: false, transparent: true, opacity: 0.9 }),
    );
    this.crosshair.renderOrder = 1;
    this.scene.add(quad, this.crosshair);
    this.camera.position.z = 1;
  }

  resize(w: number, h: number, recording: boolean): void {
    const aspect = w / h;
    this.camera.left = -aspect;
    this.camera.right = aspect;
    this.camera.updateProjectionMatrix();
    // About 22 px across in normal play; a little larger in relative terms at 128 px.
    const px = recording ? Math.max(9, h * 0.075) : Math.max(18, Math.min(30, h * 0.035));
    this.crosshair.scale.setScalar((px / h) * 2);
  }

  set(hurt: number, death: number, showCrosshair: boolean): void {
    this.tint.uniforms.uHurt.value = hurt;
    this.tint.uniforms.uDeath.value = death;
    this.crosshair.visible = showCrosshair;
    (this.tint as THREE.Material).visible = hurt > 0.001 || death > 0.001;
  }
}

// ---------------------------------------------------------------- viewmodel

export interface ViewmodelInput {
  /** Ticks since the last shot (fractional with interpolation). */
  sinceShot: number;
  /** 0..1 progress of the current reload, or -1. */
  reload: number;
  /** Distance walked, for the bob. */
  walked: number;
  /** Horizontal speed over max speed, 0..1. */
  moving: number;
  /** Last look input, for sway. */
  look: [number, number];
  grounded: boolean;
  /** 0..1 progress into the game-over hold. */
  dead: number;
  /** Muzzle flash rotation in radians. */
  flashRot: number;
  flashScale: number;
}

export class Viewmodel {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10);
  private gun = new THREE.Group();
  private flash: THREE.Sprite;
  private sway = new THREE.Vector2();

  constructor(assets: Assets, material: THREE.Material) {
    const mesh = new THREE.Mesh(assets.geo.blaster, material);
    // The kit's blaster points along +z; turn it to face down the camera's -z.
    mesh.rotation.y = Math.PI;
    mesh.position.set(-0.1, -0.3, 0.0);
    this.gun.add(mesh);
    const tex = assets.sprites.burst.clone();
    tex.repeat.set(0.5, 1);
    tex.needsUpdate = true;
    this.flash = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: tex, color: "#ffffff", depthWrite: false, depthTest: false, transparent: true }),
    );
    // Drawn over the gun: from behind, the barrel's tip is hidden by the body.
    this.flash.renderOrder = 10;
    this.flash.position.set(-0.12, 0.42, -1.2);
    this.flash.visible = false;
    this.gun.add(this.flash);
    this.gun.scale.setScalar(0.3);
    this.scene.add(this.gun);
    this.scene.add(new THREE.HemisphereLight("#f0eaff", "#6f5f90", 2.0));
    const sun = new THREE.DirectionalLight("#fff1dc", 2.2);
    sun.position.set(-1, 2, 1.2);
    this.scene.add(sun);
  }

  resize(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  update(v: ViewmodelInput): void {
    const kick = Math.exp(-v.sinceShot * 0.45);
    const bobT = v.walked * 1.9;
    const bob = v.grounded ? v.moving : 0;
    this.sway.x += (v.look[0] - this.sway.x) * 0.25;
    this.sway.y += (v.look[1] - this.sway.y) * 0.25;
    // A reload dips the gun down and tilts it, then brings it back.
    const r = v.reload >= 0 ? Math.sin(Math.min(1, v.reload) * Math.PI) : 0;
    const x = 0.29 + Math.sin(bobT) * 0.01 * bob - this.sway.x * 0.018;
    const y = -0.25 - Math.abs(Math.cos(bobT)) * 0.009 * bob - r * 0.14 - v.dead * 0.3 + this.sway.y * 0.012 - kick * 0.006;
    const z = -0.84 + kick * 0.07;
    this.gun.position.set(x, y, z);
    this.gun.rotation.set(kick * 0.14 + r * 0.5 - v.dead * 0.6, this.sway.x * 0.05, r * 0.35 + this.sway.x * 0.03);
    const on = v.sinceShot < 1.6 && v.reload < 0;
    this.flash.visible = on;
    if (on) {
      this.flash.material.rotation = v.flashRot;
      this.flash.scale.setScalar(0.9 * v.flashScale * (0.75 + v.sinceShot * 0.3));
    }
  }
}

export function reloadProgress(reloading: number): number {
  return reloading > 0 ? 1 - reloading / RELOAD_TICKS : -1;
}

// ---------------------------------------------------------------- particles

export class Particles {
  readonly group = new THREE.Group();
  private sparks: Spark[] = [];
  private debris: Debris[] = [];
  private flashes: Flash[] = [];
  private points: THREE.Points;
  private pieces: THREE.InstancedMesh;
  private tracer: THREE.Mesh;
  private tracerLife = 0;
  private hitTex: THREE.Texture;
  private ringTex: THREE.Texture;

  constructor(assets: Assets, private recording: boolean) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_SPARKS * 3), 3));
    geo.setDrawRange(0, 0);
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        map: assets.sprites.hit,
        color: "#ffe9a8",
        size: 0.22,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.points.frustumCulled = false;
    this.pieces = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshLambertMaterial(), MAX_DEBRIS);
    this.pieces.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.pieces.setColorAt(0, new THREE.Color("#ffffff"));
    this.pieces.count = 0;
    this.pieces.castShadow = false;
    this.pieces.frustumCulled = false;
    this.tracer = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1).translate(0, 0, 0.5),
      new THREE.MeshBasicMaterial({ color: "#aef6ff", transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    this.tracer.visible = false;
    this.hitTex = assets.sprites.hit;
    this.ringTex = assets.sprites.burst.clone();
    this.ringTex.repeat.set(0.5, 1);
    this.ringTex.needsUpdate = true;
    for (let i = 0; i < MAX_FLASHES; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      s.visible = false;
      this.flashes.push({ sprite: s, age: 0, life: 0, size: 1 });
      this.group.add(s);
    }
    this.group.add(this.points, this.pieces, this.tracer);
  }

  clear(): void {
    this.sparks.length = 0;
    this.debris.length = 0;
    this.tracerLife = 0;
    this.tracer.visible = false;
    for (const f of this.flashes) {
      f.life = 0;
      f.sprite.visible = false;
    }
  }

  /** A short-lived additive sprite: hit marks (white star) or rings (orange burst). */
  flash(x: number, y: number, z: number, kind: "hit" | "ring" | "pop", size: number, life: number): void {
    const f = this.flashes.find((q) => q.life === 0) ?? this.flashes[0];
    const m = f.sprite.material;
    m.map = kind === "hit" ? this.hitTex : this.ringTex;
    m.color.set(kind === "hit" ? "#fff6c8" : kind === "pop" ? "#ff8a5c" : "#ffd27a");
    m.needsUpdate = true;
    f.sprite.position.set(x, y, z);
    f.age = 0;
    f.life = life;
    f.size = size;
    f.sprite.visible = true;
  }

  sparksAt(rng: Rng, x: number, y: number, z: number, n: number, speed: number): void {
    for (let i = 0; i < n; i++) {
      if (this.sparks.length >= MAX_SPARKS) this.sparks.shift();
      const a = rng() * Math.PI * 2;
      const u = rng() * 2 - 1;
      const s = speed * (0.4 + rng() * 0.8);
      const c = Math.sqrt(1 - u * u);
      const life = 6 + Math.floor(rng() * 7);
      this.sparks.push({ x, y, z, vx: Math.cos(a) * c * s, vy: u * s + 1.5, vz: Math.sin(a) * c * s, life, max: life });
    }
  }

  /**
   * A destroyed robot bursts into a few pieces. With rng (normal mode) directions are random;
   * without it (recording) they are a fixed pattern.
   */
  burst(rng: Rng | null, x: number, y: number, z: number): void {
    const colors = ["#f6b73c", "#e2483f", "#5b5f7a", "#f6b73c", "#c9cde3", "#e2483f", "#f6b73c", "#5b5f7a", "#e2483f", "#c9cde3"];
    const n = rng ? 10 : 6;
    for (let i = 0; i < n; i++) {
      if (this.debris.length >= MAX_DEBRIS) this.debris.shift();
      const a = rng ? rng() * Math.PI * 2 : (i / n) * Math.PI * 2;
      const up = rng ? 2 + rng() * 4 : 3.5;
      const s = rng ? 1.5 + rng() * 3 : 2.6;
      const life = rng ? 30 + Math.floor(rng() * 25) : 36;
      this.debris.push({
        x, y, z, vx: Math.cos(a) * s, vy: up, vz: Math.sin(a) * s, life, max: life,
        rx: a, ry: i, spin: rng ? (rng() - 0.5) * 0.8 : 0.3, size: rng ? 0.12 + rng() * 0.16 : 0.18,
        color: new THREE.Color(colors[i % colors.length]),
      });
    }
  }

  tracerShot(from: THREE.Vector3, to: THREE.Vector3): void {
    if (this.recording) return;
    const len = from.distanceTo(to);
    this.tracer.position.copy(from);
    this.tracer.lookAt(to);
    this.tracer.scale.set(0.035, 0.035, len);
    this.tracerLife = 2;
    this.tracer.visible = true;
  }

  /** Advance one tick. */
  step(boxes: Box[]): void {
    const dt = 1 / 30;
    for (const s of this.sparks) {
      s.vy -= 12 * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      s.life--;
    }
    this.sparks = this.sparks.filter((s) => s.life > 0);
    for (const d of this.debris) {
      d.vy -= 16 * dt;
      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.z += d.vz * dt;
      d.rx += d.spin;
      d.ry += d.spin * 0.7;
      const g = groundHeight(boxes, d.x, d.z, 0.01, d.y + 0.3, 0);
      if (d.y - d.size / 2 < g && d.vy < 0) {
        d.y = g + d.size / 2;
        d.vy *= -0.35;
        d.vx *= 0.6;
        d.vz *= 0.6;
        d.spin *= 0.5;
      }
      d.life--;
    }
    this.debris = this.debris.filter((d) => d.life > 0 && d.y > -30);
    for (const f of this.flashes) {
      if (f.life === 0) continue;
      f.age++;
      if (f.age >= f.life) {
        f.life = 0;
        f.sprite.visible = false;
      }
    }
    if (this.tracerLife > 0 && --this.tracerLife === 0) this.tracer.visible = false;
  }

  /** Write the current particles into their GPU buffers. */
  draw(): void {
    const pos = this.points.geometry.getAttribute("position") as THREE.BufferAttribute;
    this.sparks.forEach((s, i) => pos.setXYZ(i, s.x, s.y, s.z));
    pos.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.sparks.length);
    this.points.visible = this.sparks.length > 0;

    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const p = new THREE.Vector3();
    const sc = new THREE.Vector3();
    this.debris.forEach((d, i) => {
      const k = Math.min(1, d.life / 10);
      m.compose(p.set(d.x, d.y, d.z), q.setFromEuler(e.set(d.rx, d.ry, 0)), sc.setScalar(d.size * k));
      this.pieces.setMatrixAt(i, m);
      this.pieces.setColorAt(i, d.color);
    });
    this.pieces.count = this.debris.length;
    this.pieces.instanceMatrix.needsUpdate = true;
    if (this.pieces.instanceColor) this.pieces.instanceColor.needsUpdate = true;
    this.pieces.visible = this.debris.length > 0;

    for (const f of this.flashes) {
      if (f.life === 0) continue;
      const t = f.age / f.life;
      f.sprite.scale.setScalar(f.size * (0.5 + t * 0.9));
      f.sprite.material.opacity = 1 - t * t;
    }
  }
}
