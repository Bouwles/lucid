// The contract every built-in LUCID game implements. See docs/games.md.

/** Unified buttons (docs/formats.md): bit i is set while button i is held. */
export const BUTTON = {
  left: 1 << 0,
  right: 1 << 1,
  up: 1 << 2,
  down: 1 << 3,
  jump: 1 << 4,
  actionA: 1 << 5,
  actionB: 1 << 6,
} as const;

export type Vertical = "grounded" | "jumping" | "sliding" | "falling" | "none";

/** One logic tick's input. look is in [-1, 1]; [0, 0] for games without mouse look. */
export interface Action {
  buttons: number;
  look: [number, number];
}

/** Labels for evaluation probes and dataset statistics. Never model inputs. */
export interface GameState {
  lane: number | null;
  x: number;
  y: number;
  vertical: Vertical;
  alive: boolean;
  speed: number;
  distance: number;
  /** Any other numeric labels worth keeping, e.g. heading. Saved with the state. */
  extra?: Record<string, number>;
}

export interface GameOptions {
  canvas: HTMLCanvasElement;
  /** normal: what people play (effects, shake, particles). recording: calm and clean for LUCID. */
  mode: "normal" | "recording";
  seed: number;
  /** Play sound effects (normal mode only). */
  audio: boolean;
}

/** Text the app draws over the game (normal mode) or over its dream. */
export type Hud = Record<string, string>;

export interface Autopilot {
  /** The action for the next tick, chosen from the current game state. Deterministic given its seed. */
  act(game: LucidGame): Action;
  /** Settings saved in each episode's meta.json. */
  describe(): Record<string, unknown>;
}

export interface LucidGame {
  readonly id: string;
  /** Fixed logic rate in ticks per second (30 for every built-in game). */
  readonly logicHz: number;
  /** Load assets and build the scene. Resolves when ready to reset(). */
  init(opts: GameOptions): Promise<void>;
  /** Start a new episode. All randomness comes from this seed. */
  reset(seed: number): void;
  /**
   * Advance exactly one logic tick with this action. Frame N is what the player saw when choosing
   * action N; the action's effect first shows in frame N+1.
   */
  step(action: Action): void;
  /** Draw the current state. alpha in [0, 1] interpolates between the last two ticks (recording uses 1). */
  render(alpha: number): void;
  resize(width: number, height: number): void;
  /** Labels for the current tick (the state the player sees now). */
  state(): GameState;
  /** True once the episode has ended and its game-over hold (about 1 s, alive=false) has finished. */
  isOver(): boolean;
  hud(): Hud;
  createAutopilot(seed: number): Autopilot;
  dispose(): void;
}

export interface GameModule {
  createGame(): LucidGame;
}
