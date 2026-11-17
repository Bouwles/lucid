// The Drift Kit autopilot. It drives the racing line by aiming at a point ahead on it, but mixes
// in the kinds of driving a dream needs to see: clean laps, sloppy laps with late steering and
// trips onto the grass, drifting through corners, random held keys, short idles and reversing
// out of trouble. Everything comes from its own seeded generator and the game state, so the same
// seed and the same game give the same inputs.

import { mulberry32, type Rng } from "../shared/rng";
import { BUTTON, type Action, type Autopilot, type LucidGame } from "../shared/types";
import type { DriftKit } from "./index";
import { OFF_TRACK_LAT, angleDiff } from "./track";

type Style = "good" | "sloppy" | "drifty" | "chaotic";

/** Chance of each base style per episode (it can also switch mid-episode). */
const STYLE_MIX: Record<Style, number> = { good: 0.34, sloppy: 0.26, drifty: 0.24, chaotic: 0.16 };

/** Per-tick chances of an interruption, by style. */
const INTERRUPTS: Record<Style, { hold: number; idle: number; switch: number }> = {
  good: { hold: 0.002, idle: 0.0015, switch: 0.002 },
  sloppy: { hold: 0.005, idle: 0.003, switch: 0.002 },
  drifty: { hold: 0.004, idle: 0.0015, switch: 0.002 },
  chaotic: { hold: 0.02, idle: 0.008, switch: 0.003 },
};

interface Params {
  /** Look-ahead = base + perSpeed * speed (m). Short look-ahead oversteers. */
  lookBase: number;
  lookPerSpeed: number;
  /** Steering dead band (radians) and how many ticks a steering decision is held. */
  deadband: number;
  react: number;
  /** Target speed as a multiple of the racing line's speed (above 1 overcooks corners). */
  pace: number;
  /** Chance of using the handbrake through a given corner. */
  driftChance: number;
  /** Sideways offset from the racing line (m). */
  lane: number;
}

const pickStyle = (rng: Rng): Style => {
  let r = rng();
  for (const k of Object.keys(STYLE_MIX) as Style[]) {
    r -= STYLE_MIX[k];
    if (r < 0) return k;
  }
  return "good";
};

function paramsFor(style: Style, rng: Rng): Params {
  const u = (a: number, b: number) => a + (b - a) * rng();
  switch (style) {
    case "good":
      return { lookBase: u(3.5, 4.5), lookPerSpeed: u(0.4, 0.5), deadband: 0.03, react: 1, pace: u(0.92, 1.02), driftChance: 0.3, lane: u(-0.4, 0.4) };
    case "sloppy":
      return { lookBase: u(2.2, 3.2), lookPerSpeed: u(0.24, 0.34), deadband: u(0.06, 0.12), react: Math.floor(u(2, 5)), pace: u(1.04, 1.22), driftChance: 0.35, lane: u(-1.2, 1.2) };
    case "drifty":
      return { lookBase: u(3, 4.5), lookPerSpeed: u(0.35, 0.5), deadband: 0.05, react: 2, pace: u(1.0, 1.15), driftChance: 0.85, lane: u(-0.6, 0.6) };
    default:
      return { lookBase: u(2, 4), lookPerSpeed: u(0.2, 0.45), deadband: u(0.05, 0.12), react: Math.floor(u(2, 5)), pace: u(0.9, 1.3), driftChance: 0.5, lane: u(-1.5, 1.5) };
  }
}

export function createAutopilot(seed: number): Autopilot {
  const rng = mulberry32(seed);
  let lastTick = Number.POSITIVE_INFINITY;
  let style: Style = "good";
  let p: Params = paramsFor("good", rng);
  // Current interruption: a held button mask or an idle, for a number of ticks.
  let holdMask = 0;
  let holdLeft = 0;
  let reverseLeft = 0;
  let reverseSteer = 0;
  let steerHeld = 0;
  let steerTicks = 0;
  let driftLeft = 0;
  let cornerArmed = true;
  let slowTicks = 0;
  let throttleOn = true;

  const newEpisode = () => {
    style = pickStyle(rng);
    p = paramsFor(style, rng);
    holdMask = holdLeft = reverseLeft = driftLeft = steerTicks = slowTicks = 0;
    steerHeld = 0;
    cornerArmed = true;
    throttleOn = true;
    // Sometimes sit still for a moment at the start.
    if (rng() < 0.15) {
      holdMask = 0;
      holdLeft = 4 + Math.floor(rng() * 20);
    }
  };

  const act = (game: LucidGame): Action => {
    const g = game as unknown as DriftKit;
    if (g.tick < lastTick || g.tick === 0) newEpisode();
    lastTick = g.tick;
    const out = (buttons: number): Action => ({ buttons, look: [0, 0] });
    if (g.overTicks >= 0) return out(0);

    const car = g.car;
    const track = g.track;
    const speed = car.forwardSpeed;

    // Occasionally change style mid-episode.
    const chance = INTERRUPTS[style];
    if (rng() < chance.switch) {
      style = pickStyle(rng);
      p = paramsFor(style, rng);
    }

    // 1. Reversing out of trouble takes priority.
    const proj = track.project(car.x, car.z);
    const tan = track.pose(proj.s);
    const wrongWay = Math.abs(angleDiff(car.heading, Math.atan2(tan.tx, tan.tz))) > 1.9;
    if (reverseLeft > 0) {
      reverseLeft--;
      return out(BUTTON.down | reverseSteer);
    }
    slowTicks = Math.abs(speed) < 1.2 ? slowTicks + 1 : 0;
    if ((slowTicks > 12 && (g.lastButtons & BUTTON.up) && car.impact >= 0 && g.tick > 20) || (wrongWay && Math.abs(speed) < 2.5 && g.tick > 20)) {
      if (rng() < 0.6 || wrongWay) {
        reverseLeft = 12 + Math.floor(rng() * 30);
        // Reversing turns the car the other way: steer away from the track direction we want.
        const want = angleDiff(Math.atan2(tan.tx, tan.tz), car.heading);
        reverseSteer = want > 0 ? BUTTON.right : BUTTON.left;
        if (rng() < 0.25) reverseSteer = 0;
        slowTicks = 0;
        return out(BUTTON.down | reverseSteer);
      }
      slowTicks = 0;
    }

    // 2. Random held inputs and idles.
    if (holdLeft > 0) {
      holdLeft--;
      return out(holdMask);
    }
    const r = rng();
    if (r < chance.hold) {
      holdLeft = 6 + Math.floor(rng() * 40);
      holdMask = 0;
      const bits = [BUTTON.left, BUTTON.right, BUTTON.up, BUTTON.down, BUTTON.actionA];
      for (const b of bits) if (rng() < 0.4) holdMask |= b;
      if (holdMask & BUTTON.left && holdMask & BUTTON.right) holdMask &= ~(rng() < 0.5 ? BUTTON.left : BUTTON.right);
      if (holdMask & BUTTON.up && holdMask & BUTTON.down) holdMask &= ~(rng() < 0.5 ? BUTTON.up : BUTTON.down);
      // A long full-lock hold at speed is a guaranteed spin into the trees; keep those short.
      if (holdMask & (BUTTON.left | BUTTON.right) && speed > 8) holdLeft = Math.min(holdLeft, 6 + Math.floor(rng() * 12));
      return out(holdMask);
    }
    if (r < chance.hold + chance.idle) {
      holdLeft = 4 + Math.floor(rng() * 28);
      holdMask = 0;
      return out(0);
    }

    // 3. Drive the racing line. On the grass, head back to the tarmac first.
    const offTrack = Math.abs(g.lat) > OFF_TRACK_LAT;
    const look = offTrack ? 5 : p.lookBase + p.lookPerSpeed * Math.max(0, speed);
    const aim = track.linePoint(proj.s + look, offTrack ? 0 : p.lane);
    const want = angleDiff(Math.atan2(aim.x - car.x, aim.z - car.z), car.heading);
    if (steerTicks <= 0 || (offTrack && style !== "chaotic")) {
      steerHeld = want > p.deadband ? BUTTON.left : want < -p.deadband ? BUTTON.right : 0;
      steerTicks = p.react;
    }
    steerTicks--;
    let buttons = steerHeld;

    const k = track.lineIndex(proj.s + Math.max(0, speed) * 0.7);
    const kNow = track.lineIndex(proj.s);
    const target = track.lineSpeed[k] * p.pace;
    // Throttle with a little hysteresis; brake hard when well over the target.
    if (speed > target + 0.8) {
      buttons |= BUTTON.down;
      throttleOn = false;
    } else if (speed < target - 0.4 || offTrack) throttleOn = true;
    else if (speed > target + 0.2) throttleOn = false;
    if (throttleOn && !(buttons & BUTTON.down)) buttons |= BUTTON.up;

    // Handbrake through corners: fire when a tight bit of line is coming up at speed.
    const tight = track.lineSpeed[k] < 11 && track.lineSpeed[kNow] > track.lineSpeed[k] - 3;
    if (tight && cornerArmed && speed > 8) {
      cornerArmed = false;
      if (rng() < p.driftChance) driftLeft = 6 + Math.floor(rng() * 14);
    }
    if (!tight && track.lineSpeed[kNow] > 13) cornerArmed = true;
    if (driftLeft > 0) {
      driftLeft--;
      buttons |= BUTTON.actionA;
      buttons &= ~BUTTON.down;
      if (!(buttons & (BUTTON.left | BUTTON.right))) buttons |= want >= 0 ? BUTTON.left : BUTTON.right;
    }
    return out(buttons);
  };

  // A local first: `style` is a debugging extra beyond the Autopilot interface.
  const pilot = {
    act,
    /** The current driving style (for debugging; describe() reports the mix). */
    get style(): Style {
      return style;
    },
    describe: () => ({
      name: "drift-kit-autopilot",
      seed,
      styles: STYLE_MIX,
      interrupts: INTERRUPTS,
      holdTicks: [6, 45],
      idleTicks: [4, 31],
      reverseTicks: [12, 41],
    }),
  };
  return pilot;
}
