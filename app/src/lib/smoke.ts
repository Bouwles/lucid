// LUCID_SMOKE_TEST=1: open the first installed dream, wait for 30 drawn
// frames (or 30 s), report to the shell, which writes logs/smoke.json and exits.

import { smokeResult } from "./shell";

const FRAMES_NEEDED = 30;
const TIMEOUT_MS = 30_000;

let active = false;
let done = false;
let frames = 0;
let started = 0;

/** Arm smoke mode as early as possible, so an engine that fails to start is reported too. */
export function startSmoke(): void {
  active = true;
}

/** Start the 30 s clock once the dream is opening. */
export function smokeDreamOpening(): void {
  if (!active || started) return;
  started = performance.now();
  setTimeout(() => finish(false, `only ${frames} frames drawn in ${TIMEOUT_MS / 1000} s`), TIMEOUT_MS);
}

export function smokeActive(): boolean {
  return active;
}

export function smokeFrameDrawn(): void {
  if (!active || done) return;
  frames++;
  if (frames >= FRAMES_NEEDED) {
    const s = ((performance.now() - started) / 1000).toFixed(1);
    finish(true, `${frames} dream frames drawn in ${s} s`);
  }
}

export function smokeFail(detail: string): void {
  if (active) finish(false, detail);
}

function finish(ok: boolean, detail: string): void {
  if (!active || done) return;
  done = true;
  void smokeResult(ok, detail);
}
