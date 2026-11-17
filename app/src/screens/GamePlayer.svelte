<script lang="ts">
  // The real built-in game, running in the app: fixed-timestep logic, HUD drawn here, not in the canvas.
  import { onMount } from "svelte";
  import { GAMES, POINTER_LOCK } from "@games/registry";
  import { GameRunner } from "@games/shared/runner";
  import type { Hud, LucidGame } from "@games/shared/types";
  import * as engine from "../lib/engine";
  import { back, href } from "../lib/router.svelte";
  import type { Game } from "../lib/types";

  let { id }: { id: string } = $props();

  let info = $state<Game | null>(null);
  let status = $state<"loading" | "playing" | "error">("loading");
  let errorText = $state("");
  let hud = $state<Hud>({});
  let showHelp = $state(true);
  let paused = $state(false);
  let canvas: HTMLCanvasElement;
  let stage: HTMLDivElement;
  let runner: GameRunner | null = null;
  let game: LucidGame | null = null;
  let helpTimer: ReturnType<typeof setTimeout> | undefined;
  const pointerLock = POINTER_LOCK.has(id);

  function fit(): void {
    if (!canvas || !stage || !game) return;
    const r = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    game.resize(canvas.width, canvas.height);
  }

  function restart(): void {
    runner?.restart();
    canvas.focus();
  }

  function togglePause(): void {
    if (!runner) return;
    paused = !paused;
    if (paused) runner.stop();
    else {
      runner.start();
      canvas.focus();
    }
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === "Escape") {
      e.preventDefault();
      if (document.pointerLockElement) return; // the browser uses this Esc to release the mouse
      back(`/game/${id}`);
    } else if (e.key === "p" || e.key === "P") {
      togglePause();
    } else if (e.key === "r" || e.key === "R") {
      restart();
    }
  }

  onMount(() => {
    let disposed = false;
    const ro = new ResizeObserver(fit);
    (async () => {
      try {
        info = await engine.game(id).catch(() => null);
        const load = GAMES[id];
        if (!load) throw new Error(`There's no built-in game called "${id}".`);
        game = (await load()).createGame();
        await game.init({ canvas, mode: "normal", seed: Date.now() % 100000, audio: true });
        if (disposed) return game.dispose();
        fit();
        ro.observe(stage);
        runner = new GameRunner(game, canvas, { pointerLock, onTick: (g) => (hud = g.hud()) }, Date.now() % 100000);
        runner.start();
        status = "playing";
        canvas.focus();
        helpTimer = setTimeout(() => (showHelp = false), 5000);
      } catch (e) {
        status = "error";
        errorText = e instanceof Error ? e.message : String(e);
      }
    })();
    return () => {
      disposed = true;
      clearTimeout(helpTimer);
      ro.disconnect();
      runner?.stop();
      game?.dispose();
    };
  });
</script>

<svelte:window onkeydown={onKey} />

<div class="player" style:--accent={info?.accent ?? undefined} role="application" aria-label="{info?.name ?? 'Game'}">
  <header class="toolbar">
    <div class="group">
      <button class="tb" onclick={() => back(`/game/${id}`)}>Back</button>
      <span class="name">{info?.name ?? id}</span>
    </div>
    <div class="group">
      <button class="tb" onclick={restart} disabled={status !== "playing"}>Restart <kbd>R</kbd></button>
      <button class="tb" onclick={togglePause} disabled={status !== "playing"}>{paused ? "Resume" : "Pause"} <kbd>P</kbd></button>
      {#if info?.dream.installed}
        <a class="tb primary" href={href(`/dream/${id}`)}>Play its dream</a>
      {/if}
    </div>
  </header>

  <div class="stage" bind:this={stage}>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <canvas bind:this={canvas} tabindex="0" aria-label="Game view"></canvas>

    {#if status === "playing"}
      <div class="hud" aria-live="off">
        {#each Object.entries(hud) as [k, v] (k)}
          <div class="hud-item"><span class="k">{k}</span><span class="v">{v}</span></div>
        {/each}
      </div>
    {/if}

    {#if status === "loading"}
      <div class="center"><p class="muted">Loading the game…</p></div>
    {:else if status === "error"}
      <div class="center">
        <p>The game couldn't start.</p>
        <p class="muted small">{errorText}</p>
      </div>
    {/if}

    {#if paused}
      <div class="center veil"><p class="big">Paused</p><p class="muted">Press P to carry on.</p></div>
    {/if}

    {#if showHelp && info?.controls.length && status === "playing"}
      <div class="help" role="note">
        {#each info.controls as c (c.keys)}
          <div><kbd>{c.keys}</kbd> {c.action}</div>
        {/each}
        {#if pointerLock}<div class="muted">Click the game to use the mouse. Esc lets go of it.</div>{/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .player {
    height: 100%;
    display: grid;
    grid-template-rows: auto 1fr;
    background: var(--bg-deep);
  }

  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--line);
    background: var(--bg);
  }

  .group {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .name {
    font-family: var(--font-display);
    font-size: var(--t-md);
    font-weight: 700;
  }

  .tb {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    min-height: 34px;
    padding: 0 0.8rem;
    border-radius: 7px;
    border: 1px solid var(--line-strong);
    background: transparent;
    color: var(--text);
    font-size: var(--t-sm);
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
  }

  .tb:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .tb.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-ink);
  }

  kbd {
    font-family: ui-monospace, monospace;
    font-size: 0.75em;
    opacity: 0.7;
  }

  .stage {
    position: relative;
    overflow: hidden;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
    outline: none;
  }

  canvas:focus-visible {
    box-shadow: inset 0 0 0 2px var(--accent);
  }

  .hud {
    position: absolute;
    top: 1rem;
    left: 1rem;
    display: flex;
    gap: 0.6rem;
    pointer-events: none;
  }

  .hud-item {
    display: flex;
    align-items: baseline;
    gap: 0.45rem;
    padding: 0.4rem 0.75rem;
    border-radius: var(--r-sm);
    background: rgb(15 17 21 / 0.72);
    font-variant-numeric: tabular-nums;
  }

  .hud-item .k {
    color: var(--muted);
    font-size: var(--t-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .hud-item .v {
    font-family: var(--font-display);
    font-size: var(--t-md);
    font-weight: 700;
  }

  .center {
    position: absolute;
    inset: 0;
    display: grid;
    place-content: center;
    text-align: center;
    gap: 0.4rem;
  }

  .veil {
    background: rgb(15 17 21 / 0.55);
  }

  .big {
    font-family: var(--font-display);
    font-size: var(--t-xl);
    font-weight: 700;
  }

  .small {
    font-size: var(--t-sm);
  }

  .muted {
    color: var(--muted);
  }

  .help {
    position: absolute;
    left: 1rem;
    bottom: 1rem;
    display: grid;
    gap: 0.3rem;
    padding: 0.75rem 1rem;
    border-radius: var(--r-md);
    background: rgb(15 17 21 / 0.8);
    font-size: var(--t-sm);
    pointer-events: none;
  }

  .help kbd {
    opacity: 1;
    font-size: var(--t-xs);
    padding: 0.05rem 0.35rem;
    border: 1px solid var(--line-strong);
    border-radius: 4px;
    margin-right: 0.35rem;
  }
</style>
