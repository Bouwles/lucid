// Lets a Python policy play a real built-in game from pixels (driven by scripts/eval_policy_real.py).
// /remote.html?game=bramble&server=http://127.0.0.1:PORT&token=T&episodes=20&seed=500&size=64&skip=2
// Each dream step is `skip` logic ticks, so the policy sees the game at the same rate as in the dream.

import { GAMES } from "@games/registry";

const q = new URLSearchParams(location.search);
const gameId = q.get("game") ?? "bramble";
const server = q.get("server") ?? "";
const token = q.get("token") ?? "";
const episodes = Number(q.get("episodes") ?? 10);
const seed0 = Number(q.get("seed") ?? 500);
const size = Number(q.get("size") ?? 64);
const skip = Number(q.get("skip") ?? 2);
const maxSteps = Number(q.get("maxSteps") ?? 900);
const log = (s: string) => ((document.getElementById("log") as HTMLPreElement).textContent = s);

async function call(path: string, body: BodyInit, type = "application/json"): Promise<Response> {
  const r = await fetch(`${server}${path}`, { method: "POST", body, headers: { "Content-Type": type, "X-Token": token } });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r;
}

async function main() {
  const game = (await GAMES[gameId]()).createGame();
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  canvas.width = canvas.height = 128;
  await game.init({ canvas, mode: "recording", seed: seed0, audio: false });
  game.resize(128, 128);
  const grab = document.createElement("canvas");
  grab.width = grab.height = size;
  const g2 = grab.getContext("2d", { willReadFrequently: true })!;
  g2.imageSmoothingQuality = "high";
  for (let e = 0; e < episodes; e++) {
    game.reset(seed0 + e);
    await call("/start", JSON.stringify({ episode: e }));
    let steps = 0;
    let best = 0;
    while (!game.isOver() && steps < maxSteps) {
      game.render(1);
      g2.drawImage(canvas, 0, 0, size, size);
      const rgba = g2.getImageData(0, 0, size, size).data;
      const rgb = new Uint8Array(size * size * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        rgb[j] = rgba[i];
        rgb[j + 1] = rgba[i + 1];
        rgb[j + 2] = rgba[i + 2];
      }
      const action = (await (await call("/act", rgb, "application/octet-stream")).json()) as { buttons: number; look: [number, number] };
      for (let t = 0; t < skip && !game.isOver(); t++) game.step(action);
      best = Math.max(best, game.state().distance);
      steps++;
    }
    const s = game.state();
    await call("/end", JSON.stringify({ episode: e, distance: best, steps, alive: s.alive, extra: s.extra ?? {} }));
    log(`${gameId}: episode ${e + 1}/${episodes}, distance ${best.toFixed(1)}`);
  }
  await call("/done", "{}");
}

main().catch(async (err) => {
  log(String(err?.stack ?? err));
  try {
    await call("/error", JSON.stringify({ error: String(err?.stack ?? err) }));
  } catch {
    /* nothing more to do */
  }
});
