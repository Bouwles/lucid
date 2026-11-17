<script lang="ts">
  import { onMount } from "svelte";
  import * as engine from "../lib/engine";
  import { formatMb } from "../lib/format";
  import { href, back } from "../lib/router.svelte";
  import { launchGame, onGameExited } from "../lib/shell";
  import { ACTIVE_JOB_STATES, type Game, type Job } from "../lib/types";
  import GameMedia from "../components/GameMedia.svelte";
  import JobProgress from "../components/JobProgress.svelte";

  let { id }: { id: string } = $props();

  let game = $state<Game | null>(null);
  let error = $state<string | null>(null);
  let job = $state<Job | null>(null);
  let jobError = $state<string | null>(null);
  let playNote = $state<string | null>(null);
  let poll: ReturnType<typeof setInterval> | undefined;

  async function load(): Promise<void> {
    try {
      game = await engine.game(id);
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function track(j: Job): void {
    job = j;
    clearInterval(poll);
    poll = setInterval(async () => {
      if (!job) return;
      try {
        job = await engine.job(job.id);
      } catch (e) {
        jobError = e instanceof Error ? e.message : String(e);
        return;
      }
      if (job.state === "done") {
        stopPolling();
        job = null;
        await load();
      } else if (job.state === "failed") {
        stopPolling();
        jobError = job.error ?? "The download failed.";
        job = null;
      } else if (job.state === "cancelled") {
        stopPolling();
        job = null;
      }
    }, 500);
  }

  function stopPolling(): void {
    clearInterval(poll);
    poll = undefined;
  }

  async function downloadDream(): Promise<void> {
    jobError = null;
    try {
      track(await engine.startJob("download_model", { game: id }));
    } catch (e) {
      jobError = e instanceof Error ? e.message : String(e);
    }
  }

  async function downloadGame(): Promise<void> {
    jobError = null;
    try {
      track(await engine.startJob("download_game", { game: id }));
    } catch (e) {
      jobError = e instanceof Error ? e.message : String(e);
    }
  }

  async function control(action: "pause" | "resume" | "cancel"): Promise<void> {
    if (!job) return;
    try {
      job = await engine.controlJob(job.id, action);
    } catch (e) {
      jobError = e instanceof Error ? e.message : String(e);
    }
  }

  async function play(): Promise<void> {
    if (!game) return;
    playNote = null;
    try {
      await launchGame(game.id, game.game.path ?? "");
      playNote = `${game.name} is running in its own window.`;
    } catch (e) {
      playNote = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(() => {
    void load();
    // Pick up a download that's already going (or paused) for this game.
    void engine
      .jobs()
      .then((list) => {
        const active = list.find((j) => j.params?.game === id && ACTIVE_JOB_STATES.includes(j.state));
        if (active) track(active);
      })
      .catch(() => {});
    let unlisten: (() => void) | undefined;
    void onGameExited((e) => {
      if (e.id === id) playNote = `${game?.name ?? "The game"} closed.`;
    }).then((u) => (unlisten = u));
    return () => {
      stopPolling();
      unlisten?.();
    };
  });
</script>

<div class="page">
  <div class="page-inner">
    <button class="backlink" onclick={() => back("/")}>Back</button>

    {#if error}
      <h1 class="page-title">Game not found</h1>
      <p class="lede">The engine said: {error}</p>
      <a class="btn" href={href("/")} style="margin-top: 1.5rem">Go home</a>
    {:else if !game}
      <p class="muted" aria-live="polite">Loading…</p>
    {:else}
      <article class="layout" style:--game-accent={game.accent ?? "var(--accent)"} style:--accent={game.accent ?? undefined}>
        <div class="media-col">
          <div class="media">
            <GameMedia {game} autoplay fit="contain" />
          </div>
          {#if game.controls.length}
            <section aria-labelledby="controls-title">
              <h2 id="controls-title" class="sub">Controls</h2>
              <dl class="controls">
                {#each game.controls as c (c.keys)}
                  <dt><kbd>{c.keys}</kbd></dt>
                  <dd>{c.action}</dd>
                {/each}
              </dl>
            </section>
          {/if}
        </div>

        <div class="text-col">
          <div class="head">
            <p class="genre">{game.genre}</p>
            {#if game.status === "coming_soon"}
              <span class="badge soon">Coming soon</span>
            {/if}
          </div>
          <h1 class="title">{game.name}</h1>
          <p class="tagline">{game.tagline}</p>
          <p class="description">{game.description}</p>

          <div class="actions">
            <div class="action">
              {#if job}
                <JobProgress
                  {job}
                  onpause={() => control("pause")}
                  onresume={() => control("resume")}
                  oncancel={() => control("cancel")}
                />
              {:else if game.dream.installed}
                <a class="btn primary big" href={href(`/dream/${game.id}`)}>Play the dream</a>
                <p class="note">LUCID imagines this game frame by frame. No game engine runs.</p>
              {:else if game.dream.downloadable}
                <button class="btn primary big" onclick={downloadDream}>
                  Download dream{game.dream.size_mb ? ` (${formatMb(game.dream.size_mb)})` : ""}
                </button>
                <p class="note">The dream model is small and only downloads once.</p>
              {:else}
                <button class="btn big" disabled aria-describedby="dream-note">Play the dream</button>
                <p class="note" id="dream-note">The dream for this game is still being trained.</p>
              {/if}
              {#if jobError}
                <p class="error" role="alert">{jobError}</p>
              {/if}
            </div>

            <div class="action">
              {#if game.game.builtin}
                <a class="btn big" href={href(`/play/${game.id}`)}>Play the real game</a>
                <p class="note">The actual game, running right here in the app.</p>
              {:else if game.game.installed}
                <button class="btn big" onclick={play}>Play</button>
              {:else if game.game.downloadable}
                <button class="btn big" onclick={downloadGame} disabled={!!job}>
                  Download game{game.game.size_mb ? ` (${formatMb(game.game.size_mb)})` : ""}
                </button>
              {:else}
                <button class="btn big" disabled aria-describedby="play-note">Play</button>
                <p class="note" id="play-note">Game build coming soon.</p>
              {/if}
              {#if playNote}
                <p class="note" aria-live="polite">{playNote}</p>
              {/if}
            </div>

            <div class="action">
              <button class="btn big" disabled aria-describedby="split-note">Split screen: real vs dream</button>
              <p class="note" id="split-note">Coming in a later version.</p>
            </div>
          </div>
        </div>
      </article>
    {/if}
  </div>
</div>

<style>
  .backlink {
    background: none;
    border: 0;
    padding: 0.25rem 0;
    margin-bottom: 1.5rem;
    color: var(--muted);
    cursor: pointer;
    font-weight: 500;
  }

  .backlink::before {
    content: "‹ ";
  }

  .backlink:hover {
    color: var(--text);
  }

  .layout {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) minmax(320px, 1fr);
    gap: clamp(2rem, 4vw, 4rem);
    align-items: start;
  }

  .media {
    aspect-ratio: 1;
    border-radius: var(--r-lg);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line);
  }

  .sub {
    font-size: var(--t-md);
    margin: 2rem 0 0.75rem;
  }

  .controls {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.6rem 1.25rem;
    margin: 0;
    align-items: baseline;
  }

  .controls dd {
    margin: 0;
    color: var(--muted);
  }

  kbd {
    font-family: var(--font-text);
    font-size: var(--t-sm);
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border: 1px solid var(--line-strong);
    border-bottom-width: 2px;
    border-radius: 5px;
    background: var(--bg-raised);
  }

  .head {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }

  .genre {
    color: var(--game-accent);
    font-weight: 600;
  }

  .soon {
    color: var(--game-accent);
  }

  .title {
    font-size: clamp(3rem, 7vw, 6.5rem);
    font-weight: 800;
    font-variation-settings: "wdth" 75;
    line-height: 0.88;
    letter-spacing: -0.035em;
    margin-bottom: 1rem;
  }

  .tagline {
    font-size: var(--t-md);
    margin-bottom: 1rem;
  }

  .description {
    color: var(--muted);
    max-width: 58ch;
  }

  .actions {
    display: grid;
    gap: 1.5rem;
    margin-top: 2.5rem;
  }

  .action {
    display: grid;
    justify-items: start;
    gap: 0.5rem;
  }

  .note {
    font-size: var(--t-sm);
    color: var(--muted);
  }

  .error {
    font-size: var(--t-sm);
    color: var(--danger);
  }
</style>
