// Normal-mode juice: dust puffs (the kit's dust model), coin sparkles (the kit's particle sprite)
// and sound effects through WebAudio. Recording mode never creates any of this.

import * as THREE from "three";
import { mulberry32, type Rng } from "../shared/rng";
import type { Assets } from "./assets";
import coinOgg from "./assets/coin.ogg?url";
import jumpOgg from "./assets/jump.ogg?url";
import landOgg from "./assets/land.ogg?url";
import fallOgg from "./assets/fall.ogg?url";
import breakOgg from "./assets/break.ogg?url";

interface Puff {
  x: number; y: number; z: number; vx: number; vy: number; life: number; max: number; s: number;
}

interface Spark {
  x: number; y: number; vx: number; vy: number; life: number; max: number;
}

const MAX_PUFFS = 48;
const MAX_SPARKS = 64;

export class Particles {
  readonly group = new THREE.Group();
  private puffs: Puff[] = [];
  private sparks: Spark[] = [];
  private puffMesh: THREE.InstancedMesh;
  private sparkGeo: THREE.BufferGeometry;
  private sparkPos: Float32Array;
  private sparkCol: Float32Array;
  private rng: Rng = mulberry32(99);
  private m = new THREE.Matrix4();

  constructor(assets: Assets) {
    const dustMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
    this.puffMesh = new THREE.InstancedMesh(assets.geo.dust, dustMat, MAX_PUFFS);
    this.puffMesh.count = 0;
    this.puffMesh.frustumCulled = false;
    this.sparkGeo = new THREE.BufferGeometry();
    this.sparkPos = new Float32Array(MAX_SPARKS * 3);
    this.sparkCol = new Float32Array(MAX_SPARKS * 3);
    this.sparkGeo.setAttribute("position", new THREE.BufferAttribute(this.sparkPos, 3));
    this.sparkGeo.setAttribute("color", new THREE.BufferAttribute(this.sparkCol, 3));
    const sparkMat = new THREE.PointsMaterial({
      map: assets.particle, size: 0.5, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const points = new THREE.Points(this.sparkGeo, sparkMat);
    points.frustumCulled = false;
    this.group.add(this.puffMesh, points);
  }

  seed(s: number): void {
    this.rng = mulberry32(s ^ 0xd057);
  }

  clear(): void {
    this.puffs.length = 0;
    this.sparks.length = 0;
  }

  dust(x: number, y: number, n: number, spread = 1): void {
    for (let i = 0; i < n && this.puffs.length < MAX_PUFFS; i++) {
      const side = i % 2 === 0 ? 1 : -1;
      const max = 12 + Math.floor(this.rng() * 8);
      this.puffs.push({
        x: x + side * 0.15, y: y + 0.05, z: (this.rng() - 0.5) * 0.6,
        vx: side * (1.2 + this.rng() * 1.8) * spread, vy: 0.4 + this.rng() * 0.9, life: max, max, s: 0.6 + this.rng() * 0.6,
      });
    }
  }

  trail(x: number, y: number): void {
    if (this.puffs.length < MAX_PUFFS) {
      this.puffs.push({ x, y: y + 0.3, z: 0, vx: 0, vy: 0.3, life: 10, max: 10, s: 0.7 });
    }
  }

  sparkle(x: number, y: number, n: number): void {
    for (let i = 0; i < n && this.sparks.length < MAX_SPARKS; i++) {
      const a = (i / n) * Math.PI * 2 + this.rng();
      const sp = 1.5 + this.rng() * 2.5;
      const max = 10 + Math.floor(this.rng() * 8);
      this.sparks.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp + 1, life: max, max });
    }
  }

  /** Advance one logic tick. */
  tick(): void {
    const dt = 1 / 30;
    for (const p of this.puffs) {
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= 0.88;
      p.life--;
    }
    this.puffs = this.puffs.filter((p) => p.life > 0);
    for (const s of this.sparks) {
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.vy -= 6 * dt;
      s.vx *= 0.92;
      s.life--;
    }
    this.sparks = this.sparks.filter((s) => s.life > 0);
  }

  render(): void {
    const v = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const sc = new THREE.Vector3();
    this.puffs.forEach((p, i) => {
      const k = p.life / p.max;
      const s = p.s * (0.5 + 0.8 * (1 - k)) * Math.min(1, k * 3);
      this.m.compose(v.set(p.x, p.y, p.z), q, sc.setScalar(s));
      this.puffMesh.setMatrixAt(i, this.m);
    });
    this.puffMesh.count = this.puffs.length;
    this.puffMesh.instanceMatrix.needsUpdate = true;
    for (let i = 0; i < MAX_SPARKS; i++) {
      const s = this.sparks[i];
      const k = s ? s.life / s.max : 0;
      this.sparkPos.set(s ? [s.x, s.y, 0.4] : [0, -999, 0], i * 3);
      this.sparkCol.set([k, k * 0.85, k * 0.45], i * 3);
    }
    this.sparkGeo.attributes.position.needsUpdate = true;
    this.sparkGeo.attributes.color.needsUpdate = true;
  }
}

const SOUNDS = { coin: coinOgg, jump: jumpOgg, land: landOgg, fall: fallOgg, break: breakOgg } as const;
export type SoundName = keyof typeof SOUNDS;

/** WebAudio player. The context is created lazily (browsers only allow audio after a user gesture). */
export class Sound {
  private ctx: AudioContext | null = null;
  private buffers = new Map<SoundName, AudioBuffer>();
  private gain: GainNode | null = null;
  private rng: Rng = mulberry32(7);

  private tried = false;

  start(): void {
    if (this.tried) return;
    this.tried = true;
    try {
      this.ctx = new AudioContext();
      this.gain = this.ctx.createGain();
      this.gain.gain.value = 0.5;
      this.gain.connect(this.ctx.destination);
    } catch {
      return;
    }
    const ctx = this.ctx;
    for (const [name, url] of Object.entries(SOUNDS)) {
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((b) => ctx.decodeAudioData(b))
        .then((buf) => this.buffers.set(name as SoundName, buf))
        .catch(() => {}); // a missing decoder just means silence
    }
  }

  play(name: SoundName, volume = 1, pitchJitter = 0.08): void {
    const ctx = this.ctx;
    const buf = this.buffers.get(name);
    if (!ctx || !buf || !this.gain) return;
    if (ctx.state === "suspended") void ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = 1 + (this.rng() * 2 - 1) * pitchJitter;
    const g = ctx.createGain();
    g.gain.value = volume;
    src.connect(g).connect(this.gain);
    src.start();
  }

  dispose(): void {
    void this.ctx?.close();
    this.ctx = null;
  }
}
