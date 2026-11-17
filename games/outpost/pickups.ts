// Health and ammo pickups on fixed pads. Taken pickups come back after a while.

import { PADS, type Pad } from "./arena";
import type { Player } from "./player";
import { MAX_RESERVE, type Blaster } from "./weapons";

const RESPAWN = 450;
const HEALTH = 35;
const AMMO = 18;

export interface PickupState {
  pad: Pad;
  /** 0 when available, otherwise ticks until it returns. */
  wait: number;
}

export function newPickups(): PickupState[] {
  return PADS.map((pad) => ({ pad, wait: 0 }));
}

/** Collects anything the player is standing on and can use. Returns the pads collected. */
export function stepPickups(items: PickupState[], p: Player, b: Blaster, alive: boolean): PickupState[] {
  const got: PickupState[] = [];
  for (const it of items) {
    if (it.wait > 0) {
      it.wait--;
      continue;
    }
    if (!alive) continue;
    const { pad } = it;
    if (Math.hypot(p.x - pad.x, p.z - pad.z) > 0.95 || Math.abs(p.y - pad.y) > 1.3) continue;
    if (pad.kind === "health") {
      if (p.health >= 100) continue;
      p.health = Math.min(100, p.health + HEALTH);
    } else {
      if (b.reserve >= MAX_RESERVE) continue;
      b.reserve = Math.min(MAX_RESERVE, b.reserve + AMMO);
    }
    it.wait = RESPAWN;
    got.push(it);
  }
  return got;
}
