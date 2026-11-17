// Juice for normal play (tyre smoke, grass dust, sound) and the game-over colour wash, which
// both modes show. Recording mode never creates the smoke or the audio.

import * as THREE from "three";
import { mulberry32, type Rng } from "../shared/rng";

import engineUrl from "./assets/engine.ogg?url";
import skidUrl from "./assets/skid.ogg?url";
import impactUrl from "./assets/impact.ogg?url";

const SMOKE_MAX = 96;
const SMOKE_LIFE = 26; // ticks

/** Low-poly puffs (one instanced mesh, one draw call), matching the faceted look of the kit. */
export class Smoke {
  readonly mesh: THREE.InstancedMesh;
  private x = new Float32Array(SMOKE_MAX);
  private y = new Float32Array(SMOKE_MAX);
  private z = new Float32Array(SMOKE_MAX);
  private vx = new Float32Array(SMOKE_MAX);
  private vy = new Float32Array(SMOKE_MAX);
  private vz = new Float32Array(SMOKE_MAX);
  private age = new Int32Array(SMOKE_MAX).fill(SMOKE_LIFE);
  private size = new Float32Array(SMOKE_MAX);
  private next = 0;
  // Effects have their own generator, so they never disturb the game's seeded randomness.
  private rng: Rng = mulberry32(99);
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private s = new THREE.Vector3();
  private p = new THREE.Vector3();
  private white = new THREE.Color("#f4f1ea");
  private dust = new THREE.Color("#d8d0a0");

  constructor() {
    const geo = new THREE.IcosahedronGeometry(0.42, 0);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true });
    this.mesh = new THREE.InstancedMesh(geo, mat, SMOKE_MAX);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    for (let i = 0; i < SMOKE_MAX; i++) this.mesh.setColorAt(i, this.white);
    this.clear();
  }

  clear(): void {
    this.age.fill(SMOKE_LIFE);
    this.m.makeScale(0, 0, 0);
    for (let i = 0; i < SMOKE_MAX; i++) this.mesh.setMatrixAt(i, this.m);
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  emit(x: number, z: number, vx: number, vz: number, grass: boolean): void {
    const i = this.next;
    this.next = (this.next + 1) % SMOKE_MAX;
    const r = this.rng;
    this.x[i] = x + (r() - 0.5) * 0.3;
    this.y[i] = 0.25;
    this.z[i] = z + (r() - 0.5) * 0.3;
    this.vx[i] = vx * 0.25 + (r() - 0.5) * 1.2;
    this.vy[i] = 0.6 + r() * 0.6;
    this.vz[i] = vz * 0.25 + (r() - 0.5) * 1.2;
    this.size[i] = (0.7 + r() * 0.6) * (grass ? 0.55 : 1);
    this.age[i] = 0;
    this.mesh.setColorAt(i, grass ? this.dust : this.white);
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  tick(dt: number): void {
    for (let i = 0; i < SMOKE_MAX; i++) {
      if (this.age[i] >= SMOKE_LIFE) continue;
      this.age[i]++;
      this.x[i] += this.vx[i] * dt;
      this.y[i] += this.vy[i] * dt;
      this.z[i] += this.vz[i] * dt;
      this.vx[i] *= 0.92;
      this.vz[i] *= 0.92;
      const t = this.age[i] / SMOKE_LIFE;
      // Puff up fast, then shrink away (no transparency needed).
      const sc = t >= 1 ? 0 : this.size[i] * Math.sin(Math.min(1, t * 1.6) * Math.PI * 0.5) * (1 - t * t);
      this.q.setFromAxisAngle(this.p.set(0, 1, 0), i * 1.7 + t * 2);
      this.m.compose(this.p.set(this.x[i], this.y[i], this.z[i]), this.q, this.s.set(sc, sc, sc));
      this.mesh.setMatrixAt(i, this.m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** A full-screen tint drawn last. Used for the game-over view: no text, just a clear colour. */
export class Wash {
  readonly mesh: THREE.Mesh;
  private mat: THREE.ShaderMaterial;

  constructor() {
    this.mat = new THREE.ShaderMaterial({
      uniforms: { color: { value: new THREE.Color() }, amount: { value: 0 } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: `uniform vec3 color; uniform float amount; varying vec2 vUv;
        void main() {
          vec2 d = vUv - 0.5;
          float vig = smoothstep(0.2, 0.75, length(d));
          gl_FragColor = vec4(color, clamp(amount * (0.75 + 0.25 * vig), 0.0, 1.0));
          #include <colorspace_fragment>
        }`,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1000;
    this.mesh.visible = false;
  }

  set(color: THREE.Color, amount: number): void {
    this.mesh.visible = amount > 0.001;
    this.mat.uniforms.color.value.copy(color);
    this.mat.uniforms.amount.value = amount;
  }
}

/**
 * Engine, skid and impact sounds from the kit. The AudioContext is created on the first step
 * (never at load), and resumed once the page has had a user gesture.
 */
export class CarAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private engine: { src: AudioBufferSourceNode; gain: GainNode } | null = null;
  private skid: { src: AudioBufferSourceNode; gain: GainNode } | null = null;
  private impactBuf: AudioBuffer | null = null;
  private lastImpact = -999;
  private tick = 0;
  private failed = false;

  private start(): void {
    try {
      this.ctx = new AudioContext();
    } catch {
      this.failed = true;
      return;
    }
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.55;
    this.master.connect(ctx.destination);
    const load = async (url: string) => ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    const loop = (buf: AudioBuffer) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      src.connect(gain).connect(this.master!);
      src.start();
      return { src, gain };
    };
    void load(engineUrl).then((b) => (this.engine = loop(b))).catch(() => undefined);
    void load(skidUrl).then((b) => (this.skid = loop(b))).catch(() => undefined);
    void load(impactUrl).then((b) => (this.impactBuf = b)).catch(() => undefined);
  }

  /** speed in m/s, throttle 0..1, skid 0..1, impact = hit speed this tick (0 if none). */
  update(speed: number, throttle: number, skid: number, impact: number, alive: boolean): void {
    if (this.failed) return;
    if (!this.ctx) this.start();
    const ctx = this.ctx;
    if (!ctx) return;
    this.tick++;
    if (ctx.state === "suspended" && this.tick % 15 === 0) void ctx.resume().catch(() => undefined);
    const now = ctx.currentTime;
    const sf = Math.min(1, Math.abs(speed) / 18);
    if (this.engine) {
      this.engine.src.playbackRate.setTargetAtTime(0.55 + sf * 1.5 + (throttle > 0 ? 0.12 : 0), now, 0.08);
      this.engine.gain.gain.setTargetAtTime(alive ? 0.16 + sf * 0.22 + throttle * 0.08 : 0.05, now, 0.1);
    }
    if (this.skid) {
      this.skid.src.playbackRate.setTargetAtTime(0.9 + sf * 0.4, now, 0.1);
      this.skid.gain.gain.setTargetAtTime(alive ? skid * 0.5 : 0, now, 0.05);
    }
    if (impact > 1.5 && this.impactBuf && this.tick - this.lastImpact > 6) {
      this.lastImpact = this.tick;
      const src = ctx.createBufferSource();
      src.buffer = this.impactBuf;
      src.playbackRate.value = 0.9 + Math.min(0.3, impact * 0.02);
      const g = ctx.createGain();
      g.gain.value = Math.min(1, 0.25 + impact * 0.08);
      src.connect(g).connect(this.master!);
      src.start();
    }
  }

  silence(): void {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.engine?.gain.gain.setTargetAtTime(0, now, 0.05);
    this.skid?.gain.gain.setTargetAtTime(0, now, 0.05);
  }

  dispose(): void {
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}
