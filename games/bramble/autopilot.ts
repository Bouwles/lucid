// Bramble's autopilot, for recording training data.
//
// Its competent play is a small planner that uses full level knowledge: every few ticks it clones the
// world, tries about fifteen short input patterns (run, jump now or a bit later with different
// heights, wait, dash, step back) about a second ahead, and keeps the best one that survives and makes
// progress. Gaps, enemies, moving and falling platforms all fall out of that without special cases.
//
// On top of that it switches between behaviours, mixed differently in every episode: competent,
// sloppy (late, short or missed jumps that lead to real deaths), random held inputs, idling,
// backtracking and dashing. Everything comes from mulberry32(seed), so it replays exactly.

import { mulberry32, randInt, type Rng } from "../shared/rng";
import { BUTTON, type Action, type Autopilot, type LucidGame } from "../shared/types";
import type { World } from "./world";

const MODES = ["competent", "sloppy", "random", "idle", "backtrack", "dash"] as const;
type Mode = (typeof MODES)[number];

const BASE_WEIGHTS: Record<Mode, number> = {
  competent: 0.32, sloppy: 0.2, random: 0.13, idle: 0.06, backtrack: 0.14, dash: 0.15,
};

/** Ticks of planned input, then ticks of no input to check the move ends somewhere safe. */
const MACRO = 24;
const SETTLE = 12;
const REPLAN = 3;

const L = BUTTON.left;
const R = BUTTON.right;
const J = BUTTON.jump;
const D = BUTTON.actionA;

interface PlanOptions {
  dir: 1 | -1;
  /** Score bonus for plans that dash (negative = avoid dashing). */
  dashBias: number;
}

/** Candidate input patterns, each MACRO ticks long. */
function candidates(dir: 1 | -1): number[][] {
  const fwd = dir > 0 ? R : L;
  const back = dir > 0 ? L : R;
  const make = (f: (i: number) => number) => Array.from({ length: MACRO }, (_, i) => f(i));
  const jump = (at: number, hold: number, i: number) => (i >= at && i < at + hold ? J : 0);
  const stopAt = (n: number, f: (i: number) => number) => (i: number) => (i < n ? f(i) : f(i) & J);
  const out: number[][] = [
    make(() => fwd),
    make((i) => fwd | jump(0, 4, i)),
    make((i) => fwd | jump(0, 8, i)),
    make((i) => fwd | jump(0, 15, i)),
    make((i) => fwd | jump(2, 8, i)),
    make((i) => fwd | jump(2, 15, i)),
    make((i) => fwd | jump(5, 15, i)),
    make((i) => fwd | jump(8, 15, i)),
    make((i) => fwd | jump(12, 12, i)),
    make((i) => fwd | jump(16, 8, i)),
    // Walk up to an edge and stop there.
    make(stopAt(5, () => fwd)),
    make(stopAt(10, () => fwd)),
    // Hop onto something and stop, or hop twice in a row.
    make(stopAt(12, (i) => fwd | jump(0, 15, i))),
    make(stopAt(9, (i) => fwd | jump(0, 8, i))),
    make((i) => fwd | jump(0, 12, i) | jump(15, 9, i)),
    make(stopAt(22, (i) => fwd | jump(0, 12, i) | jump(15, 9, i))),
    make(() => 0),
    make((i) => (i < 6 ? 0 : fwd)),
    make((i) => (i < 12 ? 0 : fwd | jump(12, 12, i))),
    make((i) => fwd | jump(0, 15, i) | (i === 6 || i === 7 ? D : 0)),
    make((i) => fwd | (i < 2 ? D : 0)),
    make((i) => (i < 8 ? fwd | jump(0, 15, i) : back | jump(0, 15, i))),
    make((i) => (i < 6 ? back : 0)),
    make((i) => jump(0, 15, i)),
  ];
  return out;
}
const CANDIDATES = { 1: candidates(1), [-1]: candidates(-1) } as Record<number, number[][]>;

/** Pick the best candidate by simulating each on a clone of the world. */
export function planBest(w: World, opt: PlanOptions): number[] {
  const startX = w.player.x;
  const startCoins = w.coins;
  const startLevel = w.level.index;
  let best = CANDIDATES[opt.dir][16];
  let bestScore = -Infinity;
  for (const cand of CANDIDATES[opt.dir]) {
    const sim = w.clone();
    let died = -1;
    for (let i = 0; i < MACRO + SETTLE; i++) {
      sim.step(i < MACRO ? cand[i] : 0);
      if (sim.phase === "over" && sim.overKind !== "win" && sim.overKind !== "time") {
        died = i;
        break;
      }
    }
    let score: number;
    if (died >= 0) {
      score = -1000 + died;
    } else {
      const won = sim.phase === "over" && sim.overKind === "win";
      const levelUp = sim.level.index !== startLevel || sim.phase === "clear" || won;
      score = levelUp ? 500 : opt.dir * (sim.player.x - startX);
      score += 0.6 * (sim.coins - startCoins);
      if (cand.some((b) => b & J)) score -= 0.35;
      if (cand.some((b) => b & D)) score += opt.dashBias;
      // Still in the air after settling usually means a fall that the horizon didn't see the end of.
      if (!sim.player.grounded && sim.phase === "play" && sim.player.vy < -4) score -= 6;
      // Standing on a platform that is about to drop counts against a plan.
      const f = sim.player.platform >= 100 ? sim.fallers[sim.player.platform - 100] : null;
      if (f && f.state !== 0) score -= 2;
    }
    if (score > bestScore) {
      bestScore = score;
      best = cand;
    }
  }
  return best;
}

export class BrambleAutopilot implements Autopilot {
  private rng: Rng;
  private weights: Record<Mode, number> = { ...BASE_WEIGHTS };
  private mode: Mode = "competent";
  private modeLeft = 0;
  private plan: number[] = [];
  private planPos = 0;
  /** Ticks during which the current plan is followed without replanning (for sloppy mistakes). */
  private commit = 0;
  private held = 0;
  private heldLeft = 0;
  private lastTick = -1;
  private episodes = 0;
  private sloppyMiss = 0.35;
  private competentMiss = 0.03;

  constructor(private readonly seed: number) {
    this.rng = mulberry32(seed ^ 0xa11ce);
  }

  describe(): Record<string, unknown> {
    const round = (v: number) => Math.round(v * 1000) / 1000;
    return {
      policy: "bramble-planner",
      seed: this.seed,
      episode: this.episodes,
      weights: Object.fromEntries(MODES.map((m) => [m, round(this.weights[m])])),
      sloppyMissRate: round(this.sloppyMiss),
      competentMissRate: round(this.competentMiss),
      planner: { candidates: CANDIDATES[1].length, macroTicks: MACRO, settleTicks: SETTLE, replanTicks: REPLAN },
    };
  }

  private newEpisode(): void {
    this.episodes++;
    let total = 0;
    for (const m of MODES) {
      this.weights[m] = BASE_WEIGHTS[m] * (0.35 + 1.3 * this.rng());
      total += this.weights[m];
    }
    for (const m of MODES) this.weights[m] /= total;
    this.sloppyMiss = 0.25 + 0.3 * this.rng();
    this.competentMiss = 0.01 + 0.04 * this.rng();
    this.modeLeft = 0;
    this.plan = [];
    this.commit = 0;
  }

  private pickMode(): void {
    let r = this.rng();
    let mode: Mode = "competent";
    for (const m of MODES) {
      r -= this.weights[m];
      if (r <= 0) {
        mode = m;
        break;
      }
    }
    this.mode = mode;
    this.plan = [];
    this.commit = 0;
    this.heldLeft = 0;
    const ticks: Record<Mode, [number, number]> = {
      competent: [60, 180], sloppy: [60, 160], dash: [45, 120], random: [12, 75], idle: [10, 50], backtrack: [15, 60],
    };
    const [lo, hi] = ticks[mode];
    this.modeLeft = randInt(this.rng, lo, hi);
  }

  act(game: LucidGame): Action {
    const w = (game as unknown as { world: World }).world;
    const none: Action = { buttons: 0, look: [0, 0] };
    if (w.tick < this.lastTick || this.lastTick < 0) this.newEpisode();
    this.lastTick = w.tick;
    if (w.phase !== "play") return none;
    if (--this.modeLeft <= 0) this.pickMode();
    return { buttons: this.choose(w), look: [0, 0] };
  }

  private choose(w: World): number {
    switch (this.mode) {
      case "idle":
        return 0;
      case "random":
        // Random held inputs, each held for 0.2 to 1.5 s.
        if (this.heldLeft <= 0) {
          let b = 0;
          if (this.rng() < 0.5) b |= L;
          else if (this.rng() < 0.75) b |= R;
          if (this.rng() < 0.4) b |= J;
          if (this.rng() < 0.35) b |= D;
          this.held = b;
          this.heldLeft = randInt(this.rng, 6, 46);
        }
        this.heldLeft--;
        return this.held;
      case "backtrack":
        return this.planned(w, { dir: -1, dashBias: -0.6 }, 0.05);
      case "dash": {
        const b = this.planned(w, { dir: 1, dashBias: 1.2 }, this.competentMiss);
        // Extra dash taps on open ground.
        return w.player.grounded && this.rng() < 0.06 ? b | D : b;
      }
      case "sloppy":
        return this.planned(w, { dir: 1, dashBias: -0.6 }, this.sloppyMiss);
      default:
        return this.planned(w, { dir: 1, dashBias: -0.6 }, this.competentMiss);
    }
  }

  /** Follow the planner, sometimes botching a jump: too late, too short, or not at all. */
  private planned(w: World, opt: PlanOptions, missRate: number): number {
    if (this.commit > 0) {
      this.commit--;
    } else if (this.planPos >= REPLAN || this.planPos >= this.plan.length) {
      this.plan = planBest(w, opt).slice();
      this.planPos = 0;
      const jumpsNow = (this.plan[0] & J) !== 0 && w.player.grounded;
      if (jumpsNow && this.rng() < missRate) {
        this.botch();
        this.commit = 18;
      }
    }
    const b = this.plan[this.planPos] ?? 0;
    this.planPos++;
    return b;
  }

  private botch(): void {
    const p = this.plan;
    const kind = this.rng();
    if (kind < 0.4) {
      // Late: keep running for a few more ticks before jumping.
      const delay = randInt(this.rng, 3, 7);
      const shifted = p.map((_, i) => (i < delay ? p[0] & ~J : p[i - delay]));
      this.plan = shifted;
    } else if (kind < 0.75) {
      // Short: let go of jump almost at once.
      this.plan = p.map((b, i) => (i < 3 ? b : b & ~J));
    } else {
      // Missed: never jump.
      this.plan = p.map((b) => b & ~J);
    }
  }
}
