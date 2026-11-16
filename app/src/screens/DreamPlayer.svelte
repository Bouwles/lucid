<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import * as engine from "../lib/engine";
  import { back } from "../lib/router.svelte";
  import { saveSettings, settings } from "../lib/settings.svelte";
  import { smokeFail, smokeFrameDrawn } from "../lib/smoke";
  import { dur } from "../lib/motion";
  import type { Control, Game } from "../lib/types";

  let { id }: { id: string } = $props();

  // Input mapping and timing match the v1 browser demo (web/main.js), so the
  // same models respond the same way.
  const BUTTON_BITS: Record<string, number> = {
    left: 0,
    right: 1,
    up: 2,
    down: 3,
    jump: 4,
    action_a: 5,
    action_b: 6,
  };
  const KEY_MAP: Record<string, string> = {
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
  const SAMPLER_STEPS = [1, 2, 3, 4, 6, 8];
  const SEND_INTERVAL_MS = 1000 / 20;
  const CHANNEL_DREAM = 0;
  const OVERLAY_MS = 3000;
  const IDLE_MS = 2500;

  const DEFAULT_CONTROLS: Control[] = [
    { keys: "Arrows / WASD", action: "Move" },
    { keys: "Space", action: "Jump" },
    { keys: "Z", action: "Action A" },
    { keys: "X", action: "Action B" },
  ];

  type Status = "connecting" | "playing" | "busy" | "closed";

  let game = $state<Game | null>(null);
  let status = $state<Status>("connecting");
  let closeReason = $state("");
  let errorText = $state<string | null>(null);
  let fps = $state<number | null>(null);
  let frameMs = $state<number | null>(null);
  let serverSteps = $state<number | null>(null);
  let chosenSteps = $state<number | null>(settings.value.sampler_steps ?? null);
  let resolution = $state<[number, number]>([64, 64]);
  let scale = $state(1);
  let controlsVisible = $state(true);
  let toolbarIdle = $state(false);
  let explainOpen = $state(false);

  let canvas = $state<HTMLCanvasElement>();
  let stage = $state<HTMLDivElement>();
  let toolbar = $state<HTMLElement>();

  let ws: WebSocket | null = null;
  let helloSeen = false;
  let sendTimer: ReturnType<typeof setInterval> | undefined;
  let lastSentButtons = -1;
  const heldKeys = new Set<string>();
  let overlayTimer: ReturnType<typeof setTimeout> | undefined;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  let firstFrameDrawn = false;
  let destroyed = false;

  const controls = $derived(game?.controls.length ? game.controls : DEFAULT_CONTROLS);
  const shownSteps = $derived(chosenSteps ?? serverSteps);

  // ------------------------------------------------------------ connection

  function connect(): void {
    if (ws && ws.readyState <= WebSocket.OPEN) return;
    status = "connecting";
    errorText = null;
    helloSeen = false;
    let busy = false;
    const sock = new WebSocket(engine.dreamSocketUrl());
    sock.binaryType = "arraybuffer";
    ws = sock;

    sock.onmessage = (event) => {
      if (sock !== ws) return;
      if (typeof event.data === "string") {
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        if (msg.t === "busy") busy = true;
        handleMessage(msg);
      } else {
        handleFrame(event.data as ArrayBuffer);
      }
    };

    sock.onclose = (event) => {
      if (sock !== ws || destroyed) return;
      ws = null;
      helloSeen = false;
      updateSending();
      fps = frameMs = null;
      status = busy ? "busy" : "closed";
      closeReason = event.reason || "";
      smokeFail(`dream socket closed (${event.code}) ${event.reason}`);
    };
  }

  function send(obj: unknown): void {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function handleMessage(msg: Record<string, unknown>): void {
    switch (msg.t) {
      case "hello":
        onHello(msg);
        break;
      case "stats":
        fps = Number(msg.fps);
        frameMs = Number(msg.frame_ms);
        serverSteps = Number(msg.sampler_steps);
        break;
      case "error":
        showError(String(msg.msg));
        break;
    }
  }

  function onHello(msg: Record<string, unknown>): void {
    const games = (msg.games as { name: string }[] | undefined) ?? [];
    if (!games.some((g) => g.name === id)) {
      showError("This dream isn't installed. Download it from the game page first.");
      smokeFail(`hello did not list ${id}`);
      return;
    }
    const res = msg.resolution as [number, number] | undefined;
    if (res) setResolution(res[0], res[1]);
    serverSteps = Number(msg.sampler_steps);
    helloSeen = true;
    status = "playing";

    if (chosenSteps !== null && chosenSteps !== serverSteps) {
      send({ t: "config", sampler_steps: chosenSteps });
    }
    send({ t: "reset", game: id });
    if (document.hidden) send({ t: "pause" });
    lastSentButtons = -1;
    updateSending();
  }

  // ------------------------------------------------------------------ input

  function currentButtons(): number {
    const held = new Set<string>();
    for (const code of heldKeys) held.add(KEY_MAP[code]);
    // Opposite directions cancel, as in v1 and in the recorder.
    if (held.has("left") && held.has("right")) {
      held.delete("left");
      held.delete("right");
    }
    if (held.has("up") && held.has("down")) {
      held.delete("up");
      held.delete("down");
    }
    let bits = 0;
    for (const name of held) bits |= 1 << BUTTON_BITS[name];
    return bits;
  }

  function canSend(): boolean {
    return helloSeen && ws !== null && ws.readyState === WebSocket.OPEN && !document.hidden;
  }

  function sendAct(): void {
    const buttons = currentButtons();
    lastSentButtons = buttons;
    send({ t: "act", buttons, look: [0, 0] });
  }

  function onInputChange(): void {
    if (canSend() && currentButtons() !== lastSentButtons) sendAct();
  }

  function updateSending(): void {
    if (canSend()) {
      if (!sendTimer) sendTimer = setInterval(sendAct, SEND_INTERVAL_MS);
    } else if (sendTimer) {
      clearInterval(sendTimer);
      sendTimer = undefined;
    }
  }

  function inToolbar(target: EventTarget | null): boolean {
    return target instanceof Node && !!toolbar?.contains(target);
  }

  function onKeyDown(e: KeyboardEvent): void {
    if (e.key === "Escape") {
      e.preventDefault();
      if (explainOpen) {
        explainOpen = false;
      } else if (inToolbar(document.activeElement)) {
        // First Esc hands the keyboard back to the dream; the next one leaves.
        (document.activeElement as HTMLElement).blur();
      } else {
        back("/");
      }
      return;
    }
    const name = KEY_MAP[e.code];
    if (!name || e.ctrlKey || e.metaKey || e.altKey) return;
    // While a toolbar control has keyboard focus, Space and arrows operate it.
    if (inToolbar(e.target)) return;
    e.preventDefault();
    heldKeys.add(e.code);
    onInputChange();
  }

  function onKeyUp(e: KeyboardEvent): void {
    if (!KEY_MAP[e.code]) return;
    heldKeys.delete(e.code);
    onInputChange();
  }

  function releaseAll(): void {
    heldKeys.clear();
    onInputChange();
  }

  function onVisibility(): void {
    if (document.hidden) {
      heldKeys.clear();
      // Let go of every button before going quiet, so the dream isn't left holding a key.
      if (helloSeen) {
        send({ t: "act", buttons: 0, look: [0, 0] });
        lastSentButtons = 0;
        send({ t: "pause" });
      }
    } else if (helloSeen) {
      send({ t: "resume" });
    }
    updateSending();
  }

  /** After a mouse click on a toolbar control, give the keys back to the dream. */
  function giveBack(e: MouseEvent): void {
    if (e.detail > 0 && e.currentTarget instanceof HTMLElement) e.currentTarget.blur();
  }

  // ----------------------------------------------------------------- frames

  let ctx: CanvasRenderingContext2D | null = null;
  let frameSeq = 0;
  let drawnSeq = 0;
  let pending: { seq: number; bmp: ImageBitmap } | null = null;
  let rafId = 0;

  // bytes: [channel, ...webp]. Decodes can finish out of order, so each frame
  // carries a sequence number and anything older than what's queued or drawn is dropped.
  function handleFrame(buffer: ArrayBuffer): void {
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 2 || bytes[0] !== CHANNEL_DREAM) return;
    const seq = ++frameSeq;
    createImageBitmap(new Blob([bytes.subarray(1)], { type: "image/webp" }))
      .then((bmp) => {
        if (destroyed || seq <= drawnSeq || (pending && pending.seq > seq)) {
          bmp.close();
          return;
        }
        pending?.bmp.close();
        pending = { seq, bmp };
        if (!rafId) rafId = requestAnimationFrame(drawPending);
      })
      .catch(() => {});
  }

  function drawPending(): void {
    rafId = 0;
    const p = pending;
    if (!p || !canvas) return;
    pending = null;
    drawnSeq = p.seq;
    if (canvas.width !== p.bmp.width || canvas.height !== p.bmp.height) {
      setResolution(p.bmp.width, p.bmp.height);
    }
    if (!ctx) ctx = canvas.getContext("2d", { alpha: false });
    if (ctx) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(p.bmp, 0, 0);
    }
    p.bmp.close();
    smokeFrameDrawn();
    if (!firstFrameDrawn) {
      firstFrameDrawn = true;
      scheduleOverlayFade();
    }
  }

  // ----------------------------------------------------------------- layout

  function setResolution(w: number, h: number): void {
    if (resolution[0] === w && resolution[1] === h) return;
    resolution = [w, h];
    if (canvas) {
      canvas.width = w;
      canvas.height = h;
      ctx = null;
    }
    layout();
  }

  // Largest whole-number scale that fits, so every model pixel becomes an
  // exact block of screen pixels. Fractional only if even 1x doesn't fit.
  function layout(): void {
    if (!stage) return;
    const availW = stage.clientWidth - 48;
    const availH = stage.clientHeight - 48;
    const fit = Math.min(availW / resolution[0], availH / resolution[1]);
    scale = fit >= 1 ? Math.floor(fit) : Math.max(fit, 0.1);
  }

  // ---------------------------------------------------------------- overlays

  function scheduleOverlayFade(): void {
    clearTimeout(overlayTimer);
    overlayTimer = setTimeout(() => (controlsVisible = false), OVERLAY_MS);
  }

  function showControls(e: MouseEvent): void {
    giveBack(e);
    controlsVisible = true;
    scheduleOverlayFade();
  }

  function wake(): void {
    toolbarIdle = false;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      if (!inToolbar(document.activeElement) && !explainOpen) toolbarIdle = true;
    }, IDLE_MS);
  }

  let errorTimer: ReturnType<typeof setTimeout> | undefined;
  function showError(text: string): void {
    errorText = text;
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => (errorText = null), 6000);
  }

  // --------------------------------------------------------------- controls

  function chooseSteps(n: number, e: MouseEvent): void {
    giveBack(e);
    chosenSteps = n;
    send({ t: "config", sampler_steps: n });
  }

  function resetDream(e: MouseEvent): void {
    giveBack(e);
    send({ t: "reset", game: id });
  }

  onMount(() => {
    void engine
      .game(id)
      .then((g) => (game = g))
      .catch(() => {});
    void saveSettings({ last_dream: id }).catch(() => {});
    connect();
    layout();
    wake();

    const ro = new ResizeObserver(layout);
    if (stage) ro.observe(stage);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", releaseAll);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      destroyed = true;
      ro.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", releaseAll);
      document.removeEventListener("visibilitychange", onVisibility);
      clearInterval(sendTimer);
      clearTimeout(overlayTimer);
      clearTimeout(idleTimer);
      clearTimeout(errorTimer);
      if (rafId) cancelAnimationFrame(rafId);
      pending?.bmp.close();
      const sock = ws;
      ws = null;
      sock?.close(1000, "left the dream");
    };
  });
</script>

<div class="player" class:idle={toolbarIdle && !explainOpen} style:--accent={game?.accent ?? undefined} onmousemove={wake} role="application" aria-label="{game?.name ?? 'Dream'} player">
  <header class="toolbar" bind:this={toolbar} onfocusin={wake}>
    <div class="group">
      <button class="tb back" onclick={() => back("/")} aria-keyshortcuts="Escape">
        <kbd>Esc</kbd> Back
      </button>
      <h1 class="name">{game?.name ?? id} <span class="muted">dream</span></h1>
    </div>

    <div class="group">
      <div class="seg" role="radiogroup" aria-label="Sampler steps">
        <span class="seg-label" aria-hidden="true">Steps</span>
        {#each SAMPLER_STEPS as n (n)}
          <button
            role="radio"
            aria-checked={shownSteps === n}
            aria-label="{n} sampler steps"
            onclick={(e) => chooseSteps(n, e)}>{n}</button
          >
        {/each}
      </div>

      <div class="seg" role="radiogroup" aria-label="Quality">
        <button role="radio" aria-checked="true" onclick={giveBack}>Fast</button>
        <span title="Needs the upsampler, coming later">
          <button role="radio" aria-checked="false" disabled aria-describedby="sharp-why">Sharp</button>
        </span>
        <span id="sharp-why" class="visually-hidden">Needs the upsampler, coming later.</span>
      </div>

      <button class="tb" onclick={resetDream}>Reset dream</button>
      <button class="tb" onclick={showControls}>Controls</button>
      <div class="explain-wrap">
        <button
          class="tb"
          aria-expanded={explainOpen}
          aria-controls="explain"
          onclick={(e) => {
            explainOpen = !explainOpen;
            giveBack(e);
          }}>What am I looking at?</button
        >
        {#if explainOpen}
          <div id="explain" class="explain" transition:fade={{ duration: dur(140) }}>
            <p>
              Nothing here is running a game engine. A neural network is looking at the last few frames and the
              keys you're pressing, and imagining the next frame, about 15 times a second.{#if id === "test-track"}{" "}This test track is a tiny practice game I use to check the whole system works; the real games are
                coming.{/if}
            </p>
            <button
              class="btn small"
              onclick={(e) => {
                explainOpen = false;
                giveBack(e);
              }}>Got it</button
            >
          </div>
        {/if}
      </div>
    </div>
  </header>

  <div class="stage" bind:this={stage}>
    <canvas
      bind:this={canvas}
      width={resolution[0]}
      height={resolution[1]}
      style:width="{Math.floor(resolution[0] * scale)}px"
      style:height="{Math.floor(resolution[1] * scale)}px"
      aria-label="Dream frames"
    ></canvas>

    <div class="stats num" aria-label="Performance">
      <span><span class="k">FPS</span> {fps != null ? fps.toFixed(1) : "–"}</span>
      <span><span class="k">Frame</span> {frameMs != null ? `${frameMs.toFixed(1)} ms` : "–"}</span>
      <span><span class="k">Steps</span> {serverSteps ?? "–"}</span>
    </div>

    {#if status === "connecting"}
      <div class="notice" aria-live="polite"><p>Waking the dream up…</p></div>
    {:else if status === "busy"}
      <div class="notice" role="alert">
        <p>The engine is already running a dream somewhere else.</p>
        <button class="btn" onclick={connect}>Try again</button>
      </div>
    {:else if status === "closed"}
      <div class="notice" role="alert">
        <p>The dream connection closed.{closeReason ? ` ${closeReason}` : ""}</p>
        <button class="btn primary" onclick={connect}>Reconnect</button>
      </div>
    {/if}

    {#if controlsVisible && status === "playing"}
      <ul class="keys" aria-label="Controls" out:fade={{ duration: dur(600) }}>
        {#each controls as c (c.keys)}
          <li><kbd>{c.keys}</kbd> {c.action}</li>
        {/each}
        <li><kbd>Esc</kbd> Back</li>
      </ul>
    {/if}

    {#if errorText}
      <p class="toast" role="alert" transition:fade={{ duration: dur(160) }}>{errorText}</p>
    {/if}
  </div>
</div>

<style>
  .player {
    position: relative;
    height: 100%;
    display: grid;
    grid-template-rows: auto 1fr;
    background: var(--bg-deep);
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem 1.5rem;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--line);
    background: var(--bg);
    transition: opacity 400ms var(--ease);
    position: relative;
    z-index: 2;
  }

  /* While playing, the toolbar steps back until the mouse moves. */
  .idle .toolbar {
    opacity: 0.3;
  }

  .group {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .name {
    font-size: var(--t-md);
    font-weight: 700;
    font-variation-settings: "wdth" 85;
    letter-spacing: -0.01em;
  }

  .name .muted {
    font-weight: 500;
  }

  .tb {
    min-height: 34px;
    padding: 0 0.8rem;
    border-radius: 7px;
    border: 1px solid var(--line-strong);
    background: transparent;
    font-size: var(--t-sm);
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background var(--dur) var(--ease);
  }

  .tb:hover {
    background: var(--bg-hover);
  }

  .back {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding-left: 0.4rem;
  }

  .stats {
    position: absolute;
    top: 0.75rem;
    right: 1rem;
    display: flex;
    gap: 1rem;
    padding: 0.35rem 0.7rem;
    border-radius: var(--r-sm);
    background: rgb(15 17 21 / 0.8);
    font-size: var(--t-xs);
    z-index: 1;
  }

  .stats .k {
    color: var(--muted);
    margin-right: 0.15rem;
  }

  .seg {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--line-strong);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
  }

  .seg-label {
    font-size: var(--t-xs);
    color: var(--muted);
    padding: 0 0.4rem 0 0.35rem;
  }

  .seg button {
    min-width: 30px;
    height: 28px;
    padding: 0 0.55rem;
    border: 0;
    border-radius: 5px;
    background: transparent;
    font-size: var(--t-sm);
    font-weight: 600;
    cursor: pointer;
  }

  .seg button:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .seg button[aria-checked="true"] {
    background: var(--accent);
    color: var(--accent-ink);
  }

  .seg button:disabled {
    color: var(--faint);
    cursor: not-allowed;
  }

  .explain-wrap {
    position: relative;
  }

  .explain {
    position: absolute;
    right: 0;
    top: calc(100% + 10px);
    width: min(420px, 80vw);
    padding: 1.25rem;
    border-radius: var(--r-md);
    background: var(--bg-raised);
    border: 1px solid var(--line-strong);
    box-shadow: 0 18px 40px rgb(0 0 0 / 0.45);
    display: grid;
    gap: 1rem;
    justify-items: start;
    line-height: 1.6;
  }

  .stage {
    position: relative;
    display: grid;
    place-items: center;
    overflow: hidden;
    min-height: 0;
  }

  canvas {
    display: block;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    background: #000;
  }

  .notice {
    position: absolute;
    inset: 0;
    display: grid;
    place-content: center;
    justify-items: center;
    gap: 1rem;
    text-align: center;
    color: var(--muted);
    background: color-mix(in srgb, var(--bg-deep) 70%, transparent);
  }

  .keys {
    position: absolute;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem 1.25rem;
    margin: 0;
    padding: 0.75rem 1.1rem;
    list-style: none;
    font-size: var(--t-sm);
    border-radius: var(--r-md);
    background: rgb(15 17 21 / 0.88);
    border: 1px solid var(--line);
    max-width: calc(100% - 2rem);
  }

  .keys li {
    white-space: nowrap;
    color: var(--muted);
  }

  kbd {
    font-family: var(--font-text);
    font-size: var(--t-xs);
    font-weight: 700;
    color: var(--text);
    padding: 0.1rem 0.4rem;
    border: 1px solid var(--line-strong);
    border-bottom-width: 2px;
    border-radius: 4px;
    background: var(--bg-raised);
  }

  .toast {
    position: absolute;
    top: 1rem;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.6rem 1rem;
    border-radius: var(--r-sm);
    background: var(--bg-raised);
    border: 1px solid var(--danger);
    color: var(--text);
    font-size: var(--t-sm);
  }
</style>
