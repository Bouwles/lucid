// Runs a game for normal play: fixed-timestep logic, interpolated rendering at display rate.

import { InputState, isGameKey } from "./input";
import type { Action, Autopilot, LucidGame } from "./types";

export interface RunnerOptions {
  /** Called after every logic tick with the game (for HUD updates). */
  onTick?: (game: LucidGame) => void;
  /** Called when an episode ends. Return true to start a new one automatically. */
  onOver?: (game: LucidGame) => boolean;
  /** Let an autopilot play instead of the keyboard. */
  autopilot?: Autopilot | null;
  /** Pointer lock for mouse look (first-person games). */
  pointerLock?: boolean;
}

export class GameRunner {
  readonly input = new InputState();
  private raf = 0;
  private last = 0;
  private acc = 0;
  private running = false;
  private seedCounter: number;
  private detach: (() => void) | null = null;

  constructor(readonly game: LucidGame, readonly canvas: HTMLCanvasElement, private opts: RunnerOptions = {}, seed = 1) {
    this.seedCounter = seed;
  }

  start(): void {
    this.game.reset(this.seedCounter++);
    this.attach();
    this.running = true;
    this.last = performance.now();
    this.acc = 0;
    const frame = (now: number) => {
      if (!this.running) return;
      const dt = 1 / this.game.logicHz;
      this.acc += Math.min(0.25, (now - this.last) / 1000);
      this.last = now;
      while (this.acc >= dt) {
        const action: Action = this.opts.autopilot ? this.opts.autopilot.act(this.game) : this.input.take();
        this.game.step(action);
        this.acc -= dt;
        this.opts.onTick?.(this.game);
        if (this.game.isOver()) {
          const again = this.opts.onOver ? this.opts.onOver(this.game) : true;
          if (again) this.game.reset(this.seedCounter++);
          else return this.stop();
        }
      }
      this.game.render(this.acc / dt);
      this.raf = requestAnimationFrame(frame);
    };
    this.raf = requestAnimationFrame(frame);
  }

  restart(): void {
    this.game.reset(this.seedCounter++);
  }

  setAutopilot(a: Autopilot | null): void {
    this.opts.autopilot = a;
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.detach?.();
    this.detach = null;
    this.input.releaseAll();
  }

  private attach(): void {
    const down = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isGameKey(e.code)) {
        e.preventDefault();
        this.input.keyDown(e.code);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (this.input.keyUp(e.code)) e.preventDefault();
    };
    const blur = () => this.input.releaseAll();
    const move = (e: MouseEvent) => {
      if (document.pointerLockElement === this.canvas) this.input.mouseMove(e.movementX, e.movementY);
    };
    const click = () => {
      if (this.opts.pointerLock && document.pointerLockElement !== this.canvas) void this.canvas.requestPointerLock?.();
    };
    const mdown = (e: MouseEvent) => {
      if (e.button === 0 && document.pointerLockElement === this.canvas) this.input.mouseButton(true);
    };
    const mup = (e: MouseEvent) => {
      if (e.button === 0) this.input.mouseButton(false);
    };
    const lockChange = () => {
      if (document.pointerLockElement !== this.canvas) this.input.mouseButton(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    document.addEventListener("mousemove", move);
    this.canvas.addEventListener("click", click);
    this.canvas.addEventListener("mousedown", mdown);
    window.addEventListener("mouseup", mup);
    document.addEventListener("pointerlockchange", lockChange);
    this.detach = () => {
      this.canvas.removeEventListener("mousedown", mdown);
      window.removeEventListener("mouseup", mup);
      document.removeEventListener("pointerlockchange", lockChange);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      document.removeEventListener("mousemove", move);
      this.canvas.removeEventListener("click", click);
      if (document.pointerLockElement === this.canvas) document.exitPointerLock();
    };
  }
}
