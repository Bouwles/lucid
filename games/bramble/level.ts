// Turns an ASCII level (levels.ts) into a tile grid plus lists of things that move or can be collected.

import type { LevelDef } from "./levels";

export const TILE = { empty: 0, ground: 1, brick: 2, coinBlock: 3, oneWay: 4 } as const;

/** A platform whose top-left corner follows a smooth back-and-forth path, purely as a function of time. */
export interface Mover {
  /** Left edge and top surface at the start of the path. */
  x0: number;
  y0: number;
  /** Displacement to the far end of the path. */
  dx: number;
  dy: number;
  w: number;
  /** Ticks for one full there-and-back cycle. */
  period: number;
}

export interface Level {
  index: number;
  name: string;
  sky: [number, number];
  w: number;
  h: number;
  /** Row-major tile codes, tiles[y * w + x], y = 0 at the bottom. */
  tiles: Uint8Array;
  coins: { x: number; y: number }[];
  enemies: { x: number; y: number }[];
  movers: Mover[];
  /** Falling platforms, given by their left edge and top surface. 2 tiles wide. */
  fallers: { x: number; y: number }[];
  /** Coin blocks, in the order their "used" flags are stored. */
  coinBlocks: { x: number; y: number }[];
  spawn: { x: number; y: number };
  flag: { x: number; y: number };
}

export function parseLevel(def: LevelDef, index: number): Level {
  const h = def.rows.length;
  const w = Math.max(...def.rows.map((r) => r.length));
  const at = (x: number, y: number) => def.rows[h - 1 - y]?.[x] ?? " ";
  const tiles = new Uint8Array(w * h);
  const lv: Level = {
    index, name: def.name, sky: def.sky, w, h, tiles,
    coins: [], enemies: [], movers: [], fallers: [], coinBlocks: [],
    spawn: { x: 2.5, y: 3 }, flag: { x: w - 4, y: 3 },
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = at(x, y);
      const i = y * w + x;
      switch (c) {
        case "#": tiles[i] = TILE.ground; break;
        case "B": tiles[i] = TILE.brick; break;
        case "?": tiles[i] = TILE.coinBlock; lv.coinBlocks.push({ x, y }); break;
        case "=": tiles[i] = TILE.oneWay; break;
        case "o": lv.coins.push({ x: x + 0.5, y: y + 0.5 }); break;
        case "e": lv.enemies.push({ x: x + 0.5, y }); break;
        case "P": lv.spawn = { x: x + 0.5, y }; break;
        case "F": lv.flag = { x: x + 0.5, y }; break;
        case "f": {
          // Each "ff" pair is one platform, so a platform starts after an even number of f's.
          let run = 0;
          while (at(x - 1 - run, y) === "f") run++;
          if (run % 2 === 0) lv.fallers.push({ x, y: y + 1 });
          break;
        }
        case "M": {
          let end = x;
          while (at(end + 1, y) === "-") end++;
          const span = end + 1 - x - 2;
          lv.movers.push({ x0: x, y0: y + 1, dx: span, dy: 0, w: 2, period: travelTicks(span) });
          break;
        }
        case "V": {
          let top = y;
          while (at(x, top + 1) === "|") top++;
          const span = top - y;
          lv.movers.push({ x0: x, y0: y + 1, dx: 0, dy: span, w: 2, period: travelTicks(span) });
          break;
        }
      }
    }
  }
  return lv;
}

/** About 2.4 tiles a second, with a short pause folded into the easing at each end. */
function travelTicks(span: number): number {
  return Math.round(30 * (2 * span / 2.4 + 1.2));
}

/** Position of a mover's top-left corner at a given tick. Smooth (cosine) easing between the two ends. */
export function moverPos(m: Mover, tick: number): { x: number; y: number } {
  const s = 0.5 - 0.5 * Math.cos((2 * Math.PI * tick) / m.period);
  return { x: m.x0 + m.dx * s, y: m.y0 + m.dy * s };
}

export function tileAt(lv: Level, x: number, y: number): number {
  if (x < 0 || x >= lv.w) return TILE.ground; // the level edges are walls
  if (y < 0 || y >= lv.h) return TILE.empty;
  return lv.tiles[y * lv.w + x];
}

export const isSolid = (t: number) => t === TILE.ground || t === TILE.brick || t === TILE.coinBlock;
