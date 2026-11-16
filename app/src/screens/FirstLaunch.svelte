<script lang="ts">
  import { onMount } from "svelte";
  import * as engine from "../lib/engine";
  import { go } from "../lib/router.svelte";
  import { saveSettings } from "../lib/settings.svelte";
  import type { Game, Hardware } from "../lib/types";
  import GameMedia from "../components/GameMedia.svelte";
  import Mark from "../components/Mark.svelte";

  let hw = $state<Hardware | null>(null);
  let hwError = $state<string | null>(null);
  let clipGame = $state<Game | null>(null);

  onMount(() => {
    // The first /hardware call measures dream speed, so it can take a few seconds.
    engine
      .hardware()
      .then((h) => (hw = h))
      .catch((e) => (hwError = e instanceof Error ? e.message : String(e)));
    engine
      .games()
      .then((list) => (clipGame = list.find((g) => g.id === "test-track" && (g.media.clip || g.media.cover)) ?? null))
      .catch(() => {});
  });

  async function leave(path: string): Promise<void> {
    try {
      await saveSettings({ seen_intro: true });
    } catch {
      /* not fatal: the intro just shows again next time */
    }
    go(path, true);
  }

  function training(h: Hardware): string {
    if (!h.can_train) return "Not practical on this machine";
    return h.training_pack_installed ? "Ready" : "Needs the training pack";
  }
</script>

<div class="intro">
  <div class="text">
    <div class="brand"><Mark size={40} /><span>LUCID</span></div>
    <p class="hello">LUCID dreams games. It watches the last few frames and the keys you press, then imagines the next frame.</p>

    <section class="check" aria-live="polite" aria-labelledby="check-title">
      <h1 id="check-title" class="visually-hidden">Hardware check</h1>
      {#if hw}
        <p class="summary">{hw.summary}</p>
        <dl class="details">
          <div><dt>System</dt><dd>{hw.os} ({hw.arch})</dd></div>
          <div><dt>Chip</dt><dd>{hw.gpu ?? "Unknown"}{hw.memory_gb ? `, ${hw.memory_gb} GB` : ""}</dd></div>
          <div>
            <dt>Dream speed</dt>
            <dd class="num">
              {hw.dream_fps != null ? `about ${Math.round(hw.dream_fps)} FPS` : "unknown"}{hw.dream_fps_measured
                ? ", measured"
                : ", estimated"}
            </dd>
          </div>
          <div><dt>Teaching</dt><dd>{training(hw)}</dd></div>
        </dl>
      {:else if hwError}
        <p class="summary">I couldn't check this computer's hardware.</p>
        <p class="muted">{hwError}</p>
      {:else}
        <p class="summary checking">Checking what this computer can do…</p>
        <p class="muted">This measures how fast dreams run here. It takes a few seconds, once.</p>
      {/if}
    </section>

    <div class="actions">
      <button class="btn primary big" onclick={() => leave("/")}>Play a game</button>
      <button class="btn big" onclick={() => leave("/teach")}>Teach LUCID a game</button>
    </div>
  </div>

  {#if clipGame}
    <figure class="clip">
      <div class="clip-media"><GameMedia game={clipGame} autoplay fit="contain" /></div>
      <figcaption>The test track, a tiny practice game for checking the system.</figcaption>
    </figure>
  {/if}
</div>

<style>
  .intro {
    height: 100%;
    overflow-y: auto;
    display: grid;
    grid-template-columns: minmax(360px, 6fr) minmax(260px, 5fr);
    align-items: center;
    gap: clamp(2rem, 5vw, 5rem);
    padding: clamp(2rem, 6vw, 6rem);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 2.25rem;
    font-variation-settings: "wdth" 75;
    line-height: 1;
    margin-bottom: 2.5rem;
  }

  .hello {
    color: var(--muted);
    font-size: var(--t-md);
    max-width: 44ch;
    margin-bottom: 2.5rem;
  }

  .summary {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: clamp(1.9rem, 3.6vw, 3rem);
    line-height: 1.08;
    letter-spacing: -0.02em;
    max-width: 22ch;
    margin-bottom: 1.5rem;
    text-wrap: balance;
  }

  .checking {
    color: var(--muted);
  }

  .details {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, max-content));
    gap: 0.75rem 2.5rem;
    margin: 0;
    font-size: var(--t-sm);
  }

  .details dt {
    color: var(--muted);
  }

  .details dd {
    margin: 0;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 3rem;
  }

  .clip {
    margin: 0;
  }

  .clip-media {
    aspect-ratio: 1;
    border-radius: var(--r-lg);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line);
  }

  figcaption {
    margin-top: 0.75rem;
    font-size: var(--t-sm);
    color: var(--muted);
  }

  @media (max-width: 1000px) {
    .intro {
      grid-template-columns: 1fr;
    }

    .clip {
      max-width: 360px;
    }
  }
</style>
