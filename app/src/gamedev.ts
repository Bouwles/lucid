// Developer page for the built-in games: /gamedev.html?game=drift-kit[&mode=recording][&autopilot=1][&size=256]
// Recording mode shows the square, calm view LUCID trains on.

import { GAMES, POINTER_LOCK } from "@games/registry";
import { GameRunner } from "@games/shared/runner";

const params = new URLSearchParams(location.search);
const id = params.get("game") ?? "drift-kit";
const mode = params.get("mode") === "recording" ? "recording" : "normal";
const canvas = document.getElementById("c") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const info = document.getElementById("info") as HTMLDivElement;

async function main() {
  const mod = await GAMES[id]();
  const game = mod.createGame();
  if (mode === "recording") {
    const size = Number(params.get("size") ?? 256);
    canvas.style.width = canvas.style.height = `${Math.min(innerWidth, innerHeight)}px`;
    canvas.width = canvas.height = size;
    canvas.style.imageRendering = "pixelated";
  } else {
    canvas.width = innerWidth * devicePixelRatio;
    canvas.height = innerHeight * devicePixelRatio;
  }
  await game.init({ canvas, mode, seed: 1, audio: mode === "normal" });
  game.resize(canvas.width, canvas.height);
  const autopilot = params.get("autopilot") ? game.createAutopilot(7) : null;
  const runner = new GameRunner(game, canvas, {
    autopilot,
    pointerLock: POINTER_LOCK.has(id),
    onTick: (g) => {
      hud.textContent = Object.entries(g.hud()).map(([k, v]) => `${k}: ${v}`).join("\n");
      const s = g.state();
      info.textContent = `${id} ${mode}${autopilot ? " autopilot" : ""}  x ${s.x.toFixed(1)} y ${s.y.toFixed(1)} ${s.vertical} speed ${s.speed.toFixed(1)} dist ${s.distance.toFixed(0)} ${s.alive ? "" : "DEAD"}`;
    },
  });
  if (mode === "normal") {
    addEventListener("resize", () => {
      canvas.width = innerWidth * devicePixelRatio;
      canvas.height = innerHeight * devicePixelRatio;
      game.resize(canvas.width, canvas.height);
    });
  }
  runner.start();
  (window as unknown as { lucidGame: unknown }).lucidGame = game;
}

main().catch((e) => {
  document.body.textContent = String(e?.stack ?? e);
});
