// Headless recording harness for the built-in games (driven by scripts/record_web_game.py).
//
// /record.html?game=drift-kit&collector=http://127.0.0.1:PORT&token=T&worker=0&episodes=20&seed=1000&size=128
//
// Runs the game in recording mode with its autopilot, as fast as possible, and streams every
// logic tick (frame, action, labels) to the collector, which writes the standard raw format.

import { GAMES } from "@games/registry";
import { mulberry32, type Rng } from "@games/shared/rng";
import type { Action, GameState } from "@games/shared/types";

/** Alternates between the autopilot and random held inputs, in segments of 4 to 24 ticks. */
class RandomHolds {
  private left = 0;
  private randomNow = false;
  private held: Action = { buttons: 0, look: [0, 0] };
  private bits: number[];

  constructor(
    private rng: Rng,
    private share: number,
    mask: number,
    private look: boolean,
    private idle = 0.15,
  ) {
    this.bits = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => 1 << i).filter((b) => mask & b);
  }

  act(autopilot: () => Action): Action {
    if (this.left <= 0) {
      this.left = 4 + Math.floor(this.rng() * 21);
      this.randomNow = this.rng() < this.share;
      if (this.randomNow) {
        let b = 0;
        // usually one or two buttons, sometimes none (letting go is an action too)
        const n = this.rng() < this.idle ? 0 : this.rng() < 0.6 ? 1 : 2;
        for (let i = 0; i < n; i++) b |= this.bits[Math.floor(this.rng() * this.bits.length)];
        const lx = this.look && this.rng() < 0.6 ? (this.rng() * 2 - 1) * 0.6 : 0;
        const ly = this.look && this.rng() < 0.3 ? (this.rng() * 2 - 1) * 0.3 : 0;
        this.held = { buttons: b, look: [lx, ly] };
      }
    }
    this.left--;
    const a = autopilot(); // always step the autopilot so its own plans stay consistent
    return this.randomNow ? this.held : a;
  }
}

const q = new URLSearchParams(location.search);
const gameId = q.get("game") ?? "drift-kit";
const collector = q.get("collector") ?? "";
const token = q.get("token") ?? "";
const worker = Number(q.get("worker") ?? 0);
const episodes = Number(q.get("episodes") ?? 10);
const seed0 = Number(q.get("seed") ?? 1);
const size = Number(q.get("size") ?? 128);
const maxTicks = Number(q.get("maxTicks") ?? 4000);
// Share of time spent on random held inputs instead of the autopilot. A competent autopilot's
// keys are predictable from the picture, so a dream trained only on it can ignore the keys and
// still do well. Random holds make the keys carry information the frames don't.
const randomShare = Number(q.get("random") ?? 0);
const buttonMask = Number(q.get("mask") ?? 0xff); // buttons this game uses
// Share of random holds with no key at all. Standing still is an action the dream has to learn too.
const idleShare = Number(q.get("idle") ?? 0.15);
const useLook = q.get("look") === "1";
const BATCH = 64;
const logEl = document.getElementById("log") as HTMLPreElement;
const log = (s: string) => (logEl.textContent = s);

interface StepRecord {
  step: number;
  time: number;
  buttons: number;
  look: [number, number];
  state: Record<string, unknown>;
  episode_end: boolean;
}

function stateJson(s: GameState): Record<string, unknown> {
  const out: Record<string, unknown> = {
    lane: s.lane, x: s.x, y: s.y, vertical: s.vertical, alive: s.alive, speed: s.speed, distance: s.distance,
  };
  for (const [k, v] of Object.entries(s.extra ?? {})) out[k] = v;
  return out;
}

async function post(path: string, body: BodyInit, contentType: string) {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(`${collector}${path}`, { method: "POST", body, headers: { "Content-Type": contentType, "X-Token": token } });
      if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
      return;
    } catch (e) {
      if (attempt >= 4) throw e;
      await new Promise((res) => setTimeout(res, 200 * (attempt + 1)));
    }
  }
}

async function sendBatch(episode: number, steps: StepRecord[], pixels: Uint8Array[]) {
  // Body: 4-byte little-endian JSON length, JSON header, then size*size*3 RGB bytes per step.
  const header = new TextEncoder().encode(JSON.stringify({ worker, episode, size, steps }));
  const body = new Uint8Array(4 + header.length + pixels.length * size * size * 3);
  new DataView(body.buffer).setUint32(0, header.length, true);
  body.set(header, 4);
  let off = 4 + header.length;
  for (const p of pixels) {
    body.set(p, off);
    off += p.length;
  }
  await post("/steps", body, "application/octet-stream");
}

async function main() {
  const game = (await GAMES[gameId]()).createGame();
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  canvas.width = canvas.height = size;
  await game.init({ canvas, mode: "recording", seed: seed0, audio: false });
  game.resize(size, size);
  const grab = document.createElement("canvas");
  grab.width = grab.height = size;
  const g2 = grab.getContext("2d", { willReadFrequently: true })!;
  const dt = 1 / game.logicHz;
  const t0 = performance.now();
  let total = 0;

  for (let e = 0; e < episodes; e++) {
    const seed = seed0 + e;
    game.reset(seed);
    const pilot = game.createAutopilot(seed * 7919 + 13);
    const mixer = new RandomHolds(mulberry32(seed * 104729 + 7), randomShare, buttonMask, useLook, idleShare);
    const settings = { ...pilot.describe(), random_share: randomShare, idle_share: idleShare, button_mask: buttonMask };
    await post("/episode_start", JSON.stringify({ worker, episode: e, seed, action_source: { name: randomShare > 0 ? "autopilot+random" : "autopilot", settings } }), "application/json");
    let steps: StepRecord[] = [];
    let pixels: Uint8Array[] = [];
    for (let t = 0; ; t++) {
      game.render(1);
      g2.drawImage(canvas, 0, 0); // read back in the same task as the render, before the buffer clears
      const rgba = g2.getImageData(0, 0, size, size).data;
      const rgb = new Uint8Array(size * size * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        rgb[j] = rgba[i];
        rgb[j + 1] = rgba[i + 1];
        rgb[j + 2] = rgba[i + 2];
      }
      const state = stateJson(game.state());
      const action = mixer.act(() => pilot.act(game));
      game.step(action);
      const end = game.isOver() || t === maxTicks - 1;
      steps.push({ step: t, time: t * dt, buttons: action.buttons & 0xff, look: action.look, state, episode_end: end });
      pixels.push(rgb);
      total++;
      if (steps.length === BATCH || end) {
        await sendBatch(e, steps, pixels);
        steps = [];
        pixels = [];
      }
      if (end) break;
    }
    await post("/episode_end", JSON.stringify({ worker, episode: e }), "application/json");
    log(`${gameId} worker ${worker}: episode ${e + 1}/${episodes}, ${total} ticks, ${(total / ((performance.now() - t0) / 1000)).toFixed(0)} ticks/s`);
  }
  game.dispose();
  await post("/done", JSON.stringify({ worker, ticks: total }), "application/json");
}

main().catch(async (err) => {
  log(String(err?.stack ?? err));
  try {
    await post("/error", JSON.stringify({ worker, error: String(err?.stack ?? err) }), "application/json");
  } catch {
    /* the collector is gone; nothing more to do */
  }
});
