const query = typeof window !== "undefined" ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

export function reducedMotion(): boolean {
  return query?.matches ?? false;
}

/** Duration helper for Svelte transitions: zero when the user asked for less motion. */
export function dur(ms: number): number {
  return reducedMotion() ? 0 : ms;
}
