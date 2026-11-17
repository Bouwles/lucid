// The built-in games. Each is loaded on demand so the app starts fast.

import type { GameModule } from "./shared/types";

export const GAMES: Record<string, () => Promise<GameModule>> = {
  "drift-kit": () => import("./drift-kit/index"),
  bramble: () => import("./bramble/index"),
  outpost: () => import("./outpost/index"),
};

/** Games that use mouse look and want pointer lock. */
export const POINTER_LOCK = new Set(["outpost"]);
