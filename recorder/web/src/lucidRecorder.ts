// LUCID web recorder. Writes the raw recording format in docs/formats.md
// ("Raw recordings, format_version 1") from a browser game's canvas.

export const RECORDER_VERSION = "0.1.0-web";
export const FORMAT_VERSION = 1;

export type ButtonName = "left" | "right" | "up" | "down" | "jump" | "action_a" | "action_b";

export const BUTTON_BITS: Readonly<Record<ButtonName, number>> = {
  left: 0,
  right: 1,
  up: 2,
  down: 3,
  jump: 4,
  action_a: 5,
  action_b: 6,
};

export const KEY_MAP: Readonly<Record<string, ButtonName>> = {
  ArrowLeft: "left",
  KeyA: "left",
  ArrowRight: "right",
  KeyD: "right",
  ArrowUp: "up",
  KeyW: "up",
  ArrowDown: "down",
  KeyS: "down",
  Space: "jump",
  KeyZ: "action_a",
  KeyX: "action_b",
};

export interface Action {
  /** Bit i set when button i is held (see BUTTON_BITS). 0-255. */
  buttons: number;
  /** Look delta for this step, each in [-1, 1]. [0, 0] for 2D games. */
  look: [number, number];
}

export interface ActionSource {
  /** "human", "autopilot", "replay" or a more specific autopilot name. */
  readonly name: string;
  readonly settings?: Record<string, unknown>;
  /** Raw mouse delta that maps to a look value of 1. null for 2D games. */
  readonly lookScale?: number | null;
  /** Called by startEpisode with the episode seed. */
  reset?(seed: number): void;
  /** Action for step `step`, chosen while frame `step` is on screen. */
  next(step: number): Action;
}

export type Vertical = "grounded" | "jumping" | "sliding" | "falling" | "none";

export interface StepState {
  lane: number | null;
  x: number;
  y: number;
  vertical: Vertical;
  alive: boolean;
  speed: number;
  distance: number;
  [extra: string]: unknown;
}

/** The part of the JSZip API the recorder uses. */
export interface JSZipLike {
  file(path: string, data: string | Blob): unknown;
  generateAsync(options: { type: "blob" }): Promise<Blob>;
}

export type JSZipConstructor = new () => JSZipLike;

export interface RecorderOptions {
  /** The game's canvas. Read synchronously inside tick(). */
  canvas: HTMLCanvasElement | OffscreenCanvas;
  gameId: string;
  /** [W, H] of the saved PNG frames. */
  recordingSize: [number, number];
  /** Seconds per logic step. */
  logicTimestep: number;
  gameBuildVersion: string;
  getState: () => StepState;
  actionSource: ActionSource;
  /** A JSZip constructor (or anything with the same file/generateAsync API). */
  zip: JSZipConstructor;
}

export interface SaveOptions {
  /** Drop saved episodes from memory afterwards. Later saves then hold only new episodes. */
  release?: boolean;
}

export interface EndOptions {
  corrupted?: boolean;
  reason?: string | null;
}

export interface Recorder {
  readonly sessionId: string;
  /** True between startEpisode and endEpisode. */
  readonly recording: boolean;
  /** Frames captured but not yet PNG-encoded. */
  readonly pendingFrames: number;
  startEpisode(seed: number): void;
  tick(): Action;
  endEpisode(options?: EndOptions): void;
  save(options?: SaveOptions): Promise<Blob>;
}

interface Episode {
  index: number;
  id: string;
  seed: number;
  actionSourceName: string;
  actionSourceSettings: Record<string, unknown>;
  lookScale: number | null;
  /** Each entry is a steps.jsonl line without episode_end, as a JSON prefix. */
  lines: string[];
  frames: (Blob | null)[];
  ended: boolean;
  corrupted: boolean;
  corruptionReason: string | null;
}

const ENCODER_POOL = 4;

function pad6(n: number): string {
  return String(n).padStart(6, "0");
}

function two(n: number): string {
  return String(n).padStart(2, "0");
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function finiteOr(v: number, fallback: number): number {
  return Number.isFinite(v) ? v : fallback;
}

export function createRecorder(options: RecorderOptions): Recorder {
  const {
    canvas,
    gameId,
    recordingSize,
    logicTimestep,
    gameBuildVersion,
    getState,
    actionSource,
    zip: Zip,
  } = options;
  const [W, H] = recordingSize;
  if (!Number.isInteger(W) || !Number.isInteger(H) || W <= 0 || H <= 0) {
    throw new Error("recordingSize must be two positive integers");
  }
  if (typeof Zip !== "function") {
    throw new Error("pass a JSZip constructor as `zip`");
  }

  const now = new Date();
  const stamp =
    `${now.getUTCFullYear()}${two(now.getUTCMonth() + 1)}${two(now.getUTCDate())}_` +
    `${two(now.getUTCHours())}${two(now.getUTCMinutes())}${two(now.getUTCSeconds())}`;
  const sessionId = `${gameId}_${stamp}`;
  const session = {
    format_version: FORMAT_VERSION,
    session_id: sessionId,
    game_id: gameId,
    created_utc: now.toISOString().replace(/\.\d{3}Z$/, "Z"),
    recorder_version: RECORDER_VERSION,
    game_build_version: gameBuildVersion,
    unity_version: null,
    recording_resolution: [W, H],
    logic_timestep: logicTimestep,
    frame_format: "png",
  };

  // The capture canvas is copied into an ImageBitmap synchronously, so the
  // game can render the next frame straight away; PNG encoding happens later
  // on a small pool of canvases.
  const capture = new OffscreenCanvas(W, H);
  const captureCtx = capture.getContext("2d") as OffscreenCanvasRenderingContext2D;
  captureCtx.imageSmoothingEnabled = true;
  captureCtx.imageSmoothingQuality = "high";

  const encoders = Array.from({ length: ENCODER_POOL }, () => {
    const c = new OffscreenCanvas(W, H);
    return {
      canvas: c,
      ctx: c.getContext("2d") as OffscreenCanvasRenderingContext2D,
      chain: Promise.resolve() as Promise<void>,
    };
  });
  let nextEncoder = 0;
  let pendingFrames = 0;

  let current: Episode | null = null;
  const finished: Episode[] = [];
  let nextIndex = 0;

  function captureFrame(ep: Episode, step: number): void {
    // Opaque black first, so transparent canvas areas save as black rather
    // than as whatever RGB sits under alpha 0.
    captureCtx.fillStyle = "#000";
    captureCtx.fillRect(0, 0, W, H);
    captureCtx.drawImage(canvas, 0, 0, W, H);
    const bitmap = capture.transferToImageBitmap();
    ep.frames[step] = null;
    pendingFrames++;

    const enc = encoders[nextEncoder];
    nextEncoder = (nextEncoder + 1) % encoders.length;
    enc.chain = enc.chain.then(async () => {
      try {
        enc.ctx.clearRect(0, 0, W, H);
        enc.ctx.drawImage(bitmap, 0, 0);
        ep.frames[step] = await enc.canvas.convertToBlob({ type: "image/png" });
      } catch (err) {
        ep.corrupted = true;
        ep.corruptionReason ??= `frame ${step} failed to encode: ${String(err)}`;
      } finally {
        bitmap.close();
        pendingFrames--;
      }
    });
  }

  function startEpisode(seed: number): void {
    if (!Number.isInteger(seed)) throw new Error("seed must be an integer");
    if (current && !current.ended) endEpisode();
    actionSource.reset?.(seed);
    const index = nextIndex;
    current = {
      index,
      id: `ep_${pad6(index)}`,
      seed,
      actionSourceName: actionSource.name,
      actionSourceSettings: { ...(actionSource.settings ?? {}) },
      lookScale: actionSource.lookScale ?? null,
      lines: [],
      frames: [],
      ended: false,
      corrupted: false,
      corruptionReason: null,
    };
  }

  function tick(): Action {
    const ep = current;
    if (!ep || ep.ended) throw new Error("call startEpisode(seed) before tick()");
    const step = ep.lines.length;

    captureFrame(ep, step);
    const raw = actionSource.next(step);
    const state = getState();

    let buttons = Math.trunc(finiteOr(raw.buttons, 0)) & 0xff;
    let look: [number, number] = [
      clamp(finiteOr(raw.look?.[0] ?? 0, 0), -1, 1),
      clamp(finiteOr(raw.look?.[1] ?? 0, 0), -1, 1),
    ];
    // The game-over frames are recorded with no input (docs/formats.md).
    if (state && state.alive === false) {
      buttons = 0;
      look = [0, 0];
    }

    const time = Math.round(step * logicTimestep * 1e6) / 1e6;
    const prefix = JSON.stringify({ step, time, buttons, look, state: state ?? null }).slice(0, -1);
    ep.lines.push(prefix);
    return { buttons, look };
  }

  function endEpisode(opts: EndOptions = {}): void {
    const ep = current;
    if (!ep || ep.ended) return;
    current = null;
    if (ep.lines.length === 0) return; // nothing recorded, index is reused
    ep.ended = true;
    if (opts.corrupted) {
      ep.corrupted = true;
      ep.corruptionReason = opts.reason ?? ep.corruptionReason ?? "marked corrupted by the game";
    }
    nextIndex = ep.index + 1;
    finished.push(ep);
  }

  async function save(opts: SaveOptions = {}): Promise<Blob> {
    await Promise.all(encoders.map((e) => e.chain));
    const episodes = finished.slice();
    const z = new Zip();
    z.file("session.json", JSON.stringify(session));

    for (const ep of episodes) {
      const dir = `episodes/${ep.id}/`;
      const n = ep.lines.length;
      for (let i = 0; i < n; i++) {
        const frame = ep.frames[i];
        if (frame) {
          z.file(`${dir}frames/${pad6(i)}.png`, frame);
        } else if (!ep.corrupted) {
          ep.corrupted = true;
          ep.corruptionReason = `frame ${i} is missing`;
        }
      }
      const steps = ep.lines.map((p, i) => `${p},"episode_end":${i === n - 1}}`).join("\n") + "\n";
      z.file(`${dir}steps.jsonl`, steps);
      const meta = {
        format_version: FORMAT_VERSION,
        episode_id: ep.id,
        session_id: sessionId,
        game_id: gameId,
        seed: ep.seed,
        game_build_version: gameBuildVersion,
        recorder_version: RECORDER_VERSION,
        action_source: { name: ep.actionSourceName, settings: ep.actionSourceSettings },
        recording_resolution: [W, H],
        logic_timestep: logicTimestep,
        look_scale: ep.lookScale,
        num_steps: n,
        corrupted: ep.corrupted,
        corruption_reason: ep.corrupted ? ep.corruptionReason : null,
      };
      z.file(`${dir}meta.json`, JSON.stringify(meta));
    }

    const blob = await z.generateAsync({ type: "blob" });
    if (opts.release) {
      for (const ep of episodes) {
        const i = finished.indexOf(ep);
        if (i >= 0) finished.splice(i, 1);
      }
    }
    return blob;
  }

  return {
    sessionId,
    get recording() {
      return current !== null && !current.ended;
    },
    get pendingFrames() {
      return pendingFrames;
    },
    startEpisode,
    tick,
    endEpisode,
    save,
  };
}

// ---------------------------------------------------------------- randomness

/** Small fast seeded PRNG. Returns floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Replaces Math.random with mulberry32(seed). Returns a function that restores it. */
export function installSeededMathRandom(seed: number): () => void {
  const original = Math.random;
  const seeded = mulberry32(seed);
  Math.random = seeded;
  return function uninstall(): void {
    if (Math.random === seeded) Math.random = original;
  };
}

// ------------------------------------------------------------ action sources

export interface KeyboardActionSourceOptions {
  /** Where key events are listened for. Default: window. */
  target?: EventTarget;
  /** Element that takes pointer lock on click. Needed only when lookScale is set. */
  pointerLockElement?: HTMLElement | null;
  /** Raw mouse delta (pixels per step) that maps to look 1. null for 2D games. */
  lookScale?: number | null;
}

/**
 * Human input. Arrows/WASD, Space, Z and X, with left+right and up+down
 * cancelling out. With a lookScale, pointer-lock mouse deltas are summed per
 * step, clipped to +-lookScale and divided by it. look[0] is positive to the
 * right and look[1] positive upwards.
 */
export class KeyboardActionSource implements ActionSource {
  readonly name = "human";
  readonly settings: Record<string, unknown> = {};
  readonly lookScale: number | null;

  private readonly target: EventTarget;
  private readonly lockElement: HTMLElement | null;
  private readonly held = new Set<string>();
  private dx = 0;
  private dy = 0;
  private attached = false;

  constructor(options: KeyboardActionSourceOptions = {}) {
    this.target = options.target ?? window;
    this.lockElement = options.pointerLockElement ?? null;
    this.lookScale = options.lookScale ?? null;
    if (this.lookScale !== null && !(this.lookScale > 0)) {
      throw new Error("lookScale must be a positive number or null");
    }
    this.attach();
  }

  attach(): void {
    if (this.attached) return;
    this.attached = true;
    this.target.addEventListener("keydown", this.onKeyDown as EventListener);
    this.target.addEventListener("keyup", this.onKeyUp as EventListener);
    window.addEventListener("blur", this.onBlur);
    if (this.lookScale !== null) {
      document.addEventListener("mousemove", this.onMouseMove);
      this.lockElement?.addEventListener("click", this.onLockClick);
    }
  }

  detach(): void {
    if (!this.attached) return;
    this.attached = false;
    this.target.removeEventListener("keydown", this.onKeyDown as EventListener);
    this.target.removeEventListener("keyup", this.onKeyUp as EventListener);
    window.removeEventListener("blur", this.onBlur);
    document.removeEventListener("mousemove", this.onMouseMove);
    this.lockElement?.removeEventListener("click", this.onLockClick);
    this.onBlur();
  }

  reset(): void {
    this.dx = 0;
    this.dy = 0;
  }

  next(): Action {
    const names = new Set<ButtonName>();
    for (const code of this.held) names.add(KEY_MAP[code]);
    if (names.has("left") && names.has("right")) {
      names.delete("left");
      names.delete("right");
    }
    if (names.has("up") && names.has("down")) {
      names.delete("up");
      names.delete("down");
    }
    let buttons = 0;
    for (const n of names) buttons |= 1 << BUTTON_BITS[n];

    let look: [number, number] = [0, 0];
    if (this.lookScale !== null) {
      const s = this.lookScale;
      look = [clamp(this.dx, -s, s) / s, clamp(-this.dy, -s, s) / s];
    }
    this.dx = 0;
    this.dy = 0;
    return { buttons, look };
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    if (!(e.code in KEY_MAP) || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    this.held.add(e.code);
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (!(e.code in KEY_MAP)) return;
    e.preventDefault();
    this.held.delete(e.code);
  };

  private onBlur = (): void => {
    this.held.clear();
    this.dx = 0;
    this.dy = 0;
  };

  private onMouseMove = (e: MouseEvent): void => {
    if (this.lockElement && document.pointerLockElement !== this.lockElement) return;
    if (!this.lockElement && document.pointerLockElement === null) return;
    this.dx += e.movementX;
    this.dy += e.movementY;
  };

  private onLockClick = (): void => {
    if (document.pointerLockElement !== this.lockElement) this.lockElement?.requestPointerLock();
  };
}

/** Plays back a recorded list of actions, for determinism tests. */
export class ReplayActionSource implements ActionSource {
  readonly name = "replay";
  readonly settings: Record<string, unknown>;
  readonly lookScale: number | null;
  private readonly actions: Action[];

  constructor(actions: Action[], options: { lookScale?: number | null; source?: string } = {}) {
    this.actions = actions;
    this.lookScale = options.lookScale ?? null;
    this.settings = options.source ? { source: options.source } : {};
  }

  /** Builds a replay from the text of a steps.jsonl file. */
  static fromStepsJsonl(text: string, options: { lookScale?: number | null; source?: string } = {}) {
    const actions = text
      .split("\n")
      .filter((line) => line.trim() !== "")
      .map((line) => {
        const s = JSON.parse(line) as { buttons: number; look: [number, number] };
        return { buttons: s.buttons, look: [s.look[0], s.look[1]] as [number, number] };
      });
    return new ReplayActionSource(actions, options);
  }

  next(step: number): Action {
    const a = this.actions[step];
    return a ? { buttons: a.buttons, look: [a.look[0], a.look[1]] } : { buttons: 0, look: [0, 0] };
  }
}
