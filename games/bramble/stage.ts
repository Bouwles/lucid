// The level as 3D geometry: repeated tiles are instanced (one draw call per kind of tile), and
// moving things (platforms, coins, enemies, the flag) are updated every frame from the world.

import * as THREE from "three";
import { withLook, type Assets, type ModelName } from "./assets";
import { ENEMY } from "./enemies";
import { isSolid, moverPos, TILE, tileAt, type Level } from "./level";
import type { World } from "./world";

/** The play plane is z = 0; ground tiles span this depth so their tops read as a walkway. */
const GROUND_DEPTH = 3;
/** Visual ground extends this far below the level, so pits look like drops rather than ledges. */
const GROUND_BELOW = 5;

/** Height of the burr's body centre above its feet. */
const BODY_Y = 0.52;

type Placement = { x: number; y: number; z: number; sx: number; sy: number; sz: number; ry?: number; depth?: number };

/** Cheap deterministic hash of a tile position, in [0, 1). */
export function hash(x: number, y: number, k = 0): number {
  let h = (x * 374761393 + y * 668265263 + k * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function instanced(geo: THREE.BufferGeometry, mat: THREE.Material, list: Placement[], shade?: (p: Placement) => number): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geo, mat, Math.max(1, list.length));
  mesh.count = list.length;
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  list.forEach((p, i) => {
    e.set(0, p.ry ?? 0, 0);
    q.setFromEuler(e);
    m.compose(new THREE.Vector3(p.x, p.y, p.z), q, new THREE.Vector3(p.sx, p.sy, p.sz));
    mesh.setMatrixAt(i, m);
    if (shade) mesh.setColorAt(i, new THREE.Color().setScalar(shade(p)));
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
  mesh.frustumCulled = false;
  return mesh;
}

interface EnemyView {
  root: THREE.Group;
  body: THREE.Mesh;
  legL: THREE.Object3D;
  legR: THREE.Object3D;
}

export class Stage {
  readonly group = new THREE.Group();
  private coins: THREE.InstancedMesh;
  private blocks: THREE.InstancedMesh;
  private usedBlocks: THREE.InstancedMesh;
  private movers: THREE.Mesh[] = [];
  private fallers: THREE.Mesh[] = [];
  private enemies: EnemyView[] = [];
  private flag: THREE.Mesh;
  private shadow: THREE.Mesh;
  /** Tick each coin was collected (for the pop animation), -1 if not collected. */
  private coinTakenAt: Int32Array;
  private blockBumpAt: Int32Array;
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private v = new THREE.Vector3();
  private s = new THREE.Vector3();
  private e = new THREE.Euler();

  constructor(readonly level: Level, private assets: Assets) {
    const lv = level;
    const geo = assets.geo;
    const kit = assets.kit;
    const lists: Partial<Record<ModelName, Placement[]>> = {};
    const add = (name: ModelName, p: Placement) => (lists[name] ??= []).push(p);
    const fill: Placement[] = [];

    for (let x = 0; x < lv.w; x++) {
      for (let y = 0; y < lv.h; y++) {
        const t = tileAt(lv, x, y);
        if (t === TILE.ground) {
          const exposed = tileAt(lv, x, y + 1) !== TILE.ground;
          if (exposed) {
            add("platform", { x: x + 0.5, y: y + 1 - 0.55, z: 0, sx: 0.5, sy: 1, sz: GROUND_DEPTH / 2 });
            this.decorate(add, x, y + 1);
          }
          // Earth below the top: one brick column per tile, down past the bottom of the level.
          const bottom = y === 0 ? -GROUND_BELOW : y;
          let surface = y;
          while (tileAt(lv, x, surface + 1) === TILE.ground) surface++;
          for (let fy = y; fy >= bottom; fy--) {
            fill.push({ x: x + 0.5, y: fy, z: 0, sx: 1, sy: 1, sz: GROUND_DEPTH * 0.92, depth: surface - fy });
          }
        } else if (t === TILE.brick) {
          add("brick", { x: x + 0.5, y, z: 0, sx: 1, sy: 1, sz: 1.3 });
        } else if (t === TILE.oneWay) {
          // Runs of ledges: 2-wide kit platforms, plus a half one for odd lengths.
          if (tileAt(lv, x - 1, y) !== TILE.oneWay) {
            let n = 0;
            while (tileAt(lv, x + n, y) === TILE.oneWay) n++;
            for (let i = 0; i < n; i += 2) {
              const wdt = Math.min(2, n - i);
              add("platform", { x: x + i + wdt / 2, y: y + 1 - 0.55 * 0.8, z: 0, sx: wdt / 2, sy: 0.8, sz: 0.75 });
            }
          }
        }
      }
    }
    for (const [name, list] of Object.entries(lists)) this.group.add(instanced(geo[name as ModelName], kit, list));
    // Earth darkens with depth, so the walkway on top stands out.
    this.group.add(instanced(geo.brick, kit, fill, (p) => Math.max(0.5, 1 - 0.12 * (p.depth ?? 0))));

    const blockList = lv.coinBlocks.map((b) => ({ x: b.x + 0.5, y: b.y, z: 0, sx: 0.92, sy: 1, sz: 1.2 }));
    this.blocks = instanced(geo.blockCoin, kit, blockList);
    this.usedBlocks = instanced(geo.brick, kit, blockList);
    this.group.add(this.blocks, this.usedBlocks);
    this.blockBumpAt = new Int32Array(lv.coinBlocks.length).fill(-1000);

    const gold = withLook(new THREE.MeshLambertMaterial({ map: assets.colormap, emissive: 0xffb020, emissiveIntensity: 0.28 }));
    this.coins = instanced(geo.coin, gold, lv.coins.map((c) => ({ x: c.x, y: c.y, z: 0, sx: 1, sy: 1, sz: 1 })));
    this.group.add(this.coins);
    this.coinTakenAt = new Int32Array(lv.coins.length).fill(-1);

    // Moving platforms in the kit's orange, so they stand apart from the yellow ledges.
    const moverMat = withLook(new THREE.MeshLambertMaterial({ map: assets.colormap, color: 0xff9a84 }));
    for (const mv of lv.movers) {
      const mesh = new THREE.Mesh(geo.platformMedium, moverMat);
      mesh.scale.set(mv.w / 3, 0.8, 0.5);
      this.movers.push(mesh);
      this.group.add(mesh);
    }
    for (let i = 0; i < lv.fallers.length; i++) {
      const mesh = new THREE.Mesh(geo.platformFalling, kit);
      mesh.scale.set(2 / 2.2, 1, 0.68);
      this.fallers.push(mesh);
      this.group.add(mesh);
    }

    // Burrs: a puffy bramble burr (the kit's dust puff, tinted) walking on two of the robot's legs.
    const legSrc = assets.character.scene.getObjectByName("leg-left") as THREE.Mesh | undefined;
    const burr = withLook(new THREE.MeshLambertMaterial({ map: assets.colormap, color: 0x7444c4 }));
    for (let i = 0; i < lv.enemies.length; i++) {
      const root = new THREE.Group();
      const body = new THREE.Mesh(geo.dust, burr);
      body.scale.setScalar(2.2);
      body.position.y = BODY_Y;
      root.add(body);
      const mkLeg = (z: number) => {
        const leg = legSrc ? new THREE.Mesh(legSrc.geometry, kit) : new THREE.Object3D();
        leg.position.set(0, 0.2, z);
        root.add(leg);
        return leg;
      };
      this.enemies.push({ root, body, legL: mkLeg(0.2), legR: mkLeg(-0.2) });
      this.group.add(root);
    }

    this.flag = new THREE.Mesh(geo.flag, kit);
    this.flag.scale.setScalar(1.7);
    this.flag.position.set(lv.flag.x, lv.flag.y, -0.2);
    this.flag.rotation.y = -Math.PI / 2;
    this.group.add(this.flag);

    // Soft blob shadow under the player (the kit's own shadow sprite).
    this.shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: assets.blob, transparent: true, depthWrite: false, opacity: 0.55 }),
    );
    this.shadow.renderOrder = 1;
    this.group.add(this.shadow);
  }

  /** Grass tufts along the top of the ground: some behind the play plane, a few small ones in front. */
  private decorate(add: (n: ModelName, p: Placement) => void, x: number, top: number): void {
    const r = hash(x, top);
    if (r < 0.55) {
      add("grass", { x: x + 0.2 + 0.6 * hash(x, top, 1), y: top, z: -0.9 - 0.4 * hash(x, top, 2), sx: 1.4, sy: 1.4, sz: 1.4, ry: 6 * r });
    }
    if (hash(x, top, 3) < 0.3) {
      add("grassSmall", { x: x + 0.5, y: top, z: 1.05, sx: 1.2, sy: 1.2, sz: 1.2, ry: 5 * r });
    }
  }

  /** Called on events: start the pop animations. */
  coinTaken(index: number, tick: number): void {
    this.coinTakenAt[index] = tick;
  }

  blockBumped(index: number, tick: number): void {
    this.blockBumpAt[index] = tick;
  }

  reset(): void {
    this.coinTakenAt.fill(-1);
    this.blockBumpAt.fill(-1000);
  }

  /** Update moving parts. t is the level tick plus the interpolation fraction. */
  update(w: World, t: number, alpha: number, playerX: number, playerY: number): void {
    const lv = this.level;
    const { m, q, v, s, e } = this;

    // Coins spin and, once taken, pop up and shrink away.
    for (let i = 0; i < lv.coins.length; i++) {
      const c = lv.coins[i];
      let scale = 1;
      let lift = 0;
      if (w.coinTaken[i]) {
        const at = this.coinTakenAt[i];
        const k = at < 0 ? 1 : Math.min(1, (t - at) / 8);
        scale = 1 - k;
        lift = k * 0.9;
      }
      e.set(0, t * 0.11 + c.x * 0.7, 0);
      q.setFromEuler(e);
      m.compose(v.set(c.x, c.y - 0.31 * 1.5 + lift, 0), q, s.setScalar(1.5 * scale));
      this.coins.setMatrixAt(i, m);
    }
    this.coins.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < lv.coinBlocks.length; i++) {
      const b = lv.coinBlocks[i];
      const dt = t - this.blockBumpAt[i];
      const bump = dt >= 0 && dt < 8 ? Math.sin((dt / 8) * Math.PI) * 0.22 : 0;
      const used = w.blockUsed[i] === 1;
      q.identity();
      m.compose(v.set(b.x + 0.5, b.y + bump, 0), q, s.set(used ? 0 : 0.92, used ? 0 : 1, used ? 0 : 1.2));
      this.blocks.setMatrixAt(i, m);
      m.compose(v.set(b.x + 0.5, b.y + bump, 0), q, s.set(used ? 1 : 0, used ? 1 : 0, used ? 1.2 : 0));
      this.usedBlocks.setMatrixAt(i, m);
    }
    this.blocks.instanceMatrix.needsUpdate = true;
    this.usedBlocks.instanceMatrix.needsUpdate = true;

    lv.movers.forEach((mv, i) => {
      const p = moverPos(mv, t);
      this.movers[i].position.set(p.x + mv.w / 2, p.y - 0.55 * 0.8, 0);
    });
    w.fallers.forEach((f, i) => {
      const mesh = this.fallers[i];
      mesh.visible = f.state !== 3;
      // Fallers move in logic ticks only; shaking is a small deterministic wobble.
      const wob = f.state === 1 ? Math.sin(f.t * 2.6) * 0.05 : 0;
      mesh.position.set(f.x + 1 + wob, f.top - 0.5, 0);
    });

    w.enemies.forEach((en, i) => {
      const view = this.enemies[i];
      const gone = !en.alive && en.deadT >= ENEMY.squashTicks;
      view.root.visible = !gone;
      if (gone) return;
      view.root.position.set(en.x + (en.alive ? en.dir * (alpha - 1) * ENEMY.speed / 30 : 0), en.y, 0);
      view.root.rotation.set(0, en.dir > 0 ? Math.PI / 2 - 0.35 : -Math.PI / 2 + 0.35, 0);
      if (en.alive) {
        const ph = en.walk * 7;
        view.legL.rotation.x = Math.sin(ph) * 0.6;
        view.legR.rotation.x = -Math.sin(ph) * 0.6;
        view.body.position.y = BODY_Y + Math.abs(Math.cos(ph)) * 0.05;
        view.body.rotation.y = Math.sin(ph * 0.5) * 0.2;
        view.root.scale.set(1, 1, 1);
        view.root.rotation.z = 0;
      } else if (en.how === "stomp") {
        const k = Math.min(1, en.deadT / 4);
        view.root.scale.set(1 + 0.35 * k, 1 - 0.75 * k, 1 + 0.35 * k);
      } else {
        // Knocked away by a dash: flies off spinning.
        const k = en.deadT;
        view.root.position.x += (playerX < en.x ? 1 : -1) * k * 0.25;
        view.root.position.y += k * 0.3 - k * k * 0.03;
        view.root.rotation.z = k * 0.5;
      }
    });

    // Flag waves gently (deterministic); jumps when reached.
    this.flag.rotation.y = -Math.PI / 2 + Math.sin(t * 0.08) * 0.12;
    const cheer = w.phase === "clear" ? Math.abs(Math.sin(w.phaseT * 0.35)) * 0.25 : 0;
    this.flag.position.y = lv.flag.y + cheer;

    // Shadow on the highest surface below the player.
    const ground = this.groundBelow(w, playerX, playerY);
    if (ground === null || w.phase === "over") {
      this.shadow.visible = false;
    } else {
      const hgt = playerY - ground;
      const k = Math.max(0, 1 - hgt / 6);
      this.shadow.visible = k > 0;
      this.shadow.position.set(playerX, ground + 0.02, 0);
      this.shadow.scale.setScalar(0.5 + 0.5 * k);
      (this.shadow.material as THREE.MeshBasicMaterial).opacity = 0.5 * k;
    }
  }

  private groundBelow(w: World, x: number, y: number): number | null {
    const lv = this.level;
    const cx = Math.floor(x);
    let best: number | null = null;
    for (let cy = Math.min(lv.h - 1, Math.floor(y + 0.01)); cy >= 0; cy--) {
      const t = tileAt(lv, cx, cy);
      if ((isSolid(t) || t === TILE.oneWay) && cy + 1 <= y + 0.01) {
        best = cy + 1;
        break;
      }
    }
    for (const s of w.surfaces()) {
      if (x >= s.x && x <= s.x + s.w && s.top <= y + 0.05 && (best === null || s.top > best)) best = s.top;
    }
    return best;
  }
}
