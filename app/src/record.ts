// Headless recording harness for the built-in games (driven by scripts/record_web_game.py).
//
// /record.html?game=drift-kit&collector=http://127.0.0.1:PORT&token=T&worker=0&episodes=20&seed=1000&size=128
//
// Runs the game in recording mode with its autopilot, as fast as possible, and streams every
// logic tick (frame, action, labels) to the collector, which writes the standard raw format.

import { GAMES } from "@games/registry";
import type { GameState } from "@games/shared/types";

const q = new URLSearchParams(location.search);
const gameId = q.get("game") ?? "drift-kit";
const collector = q.get("collector") ?? "";
const token = q.get("token") ?? "";
const worker = Number(q.get("worker") ?? 0);
const episodes = Number(q.get("episodes") ?? 10);
const seed0 = Number(q.get("seed") ?? 1);
const size = Number(q.get("size") ?? 128);
const maxTicks = Number(q.get("maxTicks") ?? 4000);
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
    await post("/episode_start", JSON.stringify({ worker, episode: e, seed, action_source: { name: "autopilot", settings: pilot.describe() } }), "application/json");
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
      const action = pilot.act(game);
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
