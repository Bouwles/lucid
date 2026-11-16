<script lang="ts">
  import { onMount } from "svelte";
  import * as engine from "../lib/engine";
  import { href } from "../lib/router.svelte";
  import { settings } from "../lib/settings.svelte";
  import type { Game } from "../lib/types";
  import GameMedia from "../components/GameMedia.svelte";

  let games = $state<Game[] | null>(null);
  let error = $state<string | null>(null);
  let hovered = $state<string | null>(null);

  const featured = $derived(games?.find((g) => g.status === "available") ?? null);
  const rest = $derived(games?.filter((g) => g !== featured) ?? []);
  const continueGame = $derived(
    games?.find((g) => g.id === settings.value.last_dream && g.dream.installed) ?? null,
  );

  async function load(): Promise<void> {
    error = null;
    try {
      games = await engine.games();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(load);
</script>

<div class="page">
  <div class="page-inner">
    {#if error}
      <h1 class="page-title">Games</h1>
      <p class="lede">I couldn't load the game list from the engine: {error}</p>
      <button class="btn" onclick={load} style="margin-top: 1.5rem">Try again</button>
    {:else if games === null}
      <p class="muted" aria-live="polite">Loading games…</p>
    {:else}
      {#if continueGame}
        <a class="continue" href={href(`/dream/${continueGame.id}`)}>
          <span class="continue-thumb"><GameMedia game={continueGame} /></span>
          <span class="continue-text">
            <span class="muted">Continue where you left off</span>
            <strong>{continueGame.name} dream</strong>
          </span>
          <span class="btn small">Continue</span>
        </a>
      {/if}

      {#if featured}
        <section
          class="feature"
          aria-labelledby="feature-title"
          style:--game-accent={featured.accent ?? "var(--accent)"}
          style:--accent={featured.accent ?? undefined}
        >
          <a
            class="feature-media"
            href={href(`/game/${featured.id}`)}
            aria-label="{featured.name} details"
            onmouseenter={() => (hovered = featured.id)}
            onmouseleave={() => (hovered = null)}
            onfocus={() => (hovered = featured.id)}
            onblur={() => (hovered = null)}
          >
            <GameMedia game={featured} playing={hovered === featured.id} />
          </a>
          <div class="feature-text">
            <p class="genre">{featured.genre}</p>
            <h1 id="feature-title" class="feature-title">{featured.name}</h1>
            <p class="tagline">{featured.tagline}</p>
            <div class="feature-actions">
              {#if featured.dream.installed}
                <a class="btn primary big" href={href(`/dream/${featured.id}`)}>Play the dream</a>
              {/if}
              <a class="btn big" href={href(`/game/${featured.id}`)}>Details</a>
            </div>
          </div>
        </section>
      {:else}
        <h1 class="page-title">Games</h1>
      {/if}

      {#if rest.length}
        <section class="section" aria-labelledby="games-title">
          <h2 id="games-title" class="section-title">{featured ? "More games" : "All games"}</h2>
          <ul class="grid">
            {#each rest as g (g.id)}
              <li>
                <a
                  class="card"
                  class:soon={g.status === "coming_soon"}
                  href={href(`/game/${g.id}`)}
                  style:--game-accent={g.accent ?? "var(--accent)"}
                  onmouseenter={() => (hovered = g.id)}
                  onmouseleave={() => (hovered = null)}
                  onfocus={() => (hovered = g.id)}
                  onblur={() => (hovered = null)}
                >
                  <span class="card-media"><GameMedia game={g} playing={hovered === g.id} /></span>
                  <span class="card-body">
                    <span class="card-top">
                      <span class="card-name">{g.name}</span>
                      {#if g.status === "coming_soon"}
                        <span class="badge soon-badge">Coming soon</span>
                      {/if}
                    </span>
                    <span class="card-genre">{g.genre}</span>
                    <span class="card-tagline">{g.tagline}</span>
                  </span>
                </a>
              </li>
            {/each}
          </ul>
        </section>
      {/if}

      <section class="section" aria-labelledby="projects-title">
        <h2 id="projects-title" class="section-title">Your Teach projects</h2>
        <div class="empty">
          <p>
            You don't have any projects yet. Teaching LUCID your own game is coming in a later version.
          </p>
          <a class="btn small" href={href("/teach")}>See how it will work</a>
        </div>
      </section>
    {/if}
  </div>
</div>

<style>
  .continue {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0.75rem 0.6rem 0.6rem;
    margin-bottom: 2.5rem;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    text-decoration: none;
    max-width: 520px;
    transition: border-color var(--dur) var(--ease);
  }

  .continue:hover {
    border-color: var(--line-strong);
  }

  .continue-thumb {
    width: 52px;
    height: 52px;
    border-radius: var(--r-sm);
    overflow: hidden;
    flex: none;
  }

  .continue-text {
    display: grid;
    flex: 1;
    font-size: var(--t-sm);
  }

  .continue-text strong {
    font-size: var(--t-base);
  }

  .feature {
    display: grid;
    grid-template-columns: minmax(260px, 5fr) 6fr;
    gap: clamp(1.5rem, 4vw, 3.5rem);
    align-items: end;
  }

  .feature-media {
    display: block;
    aspect-ratio: 1;
    border-radius: var(--r-lg);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line);
    transition: box-shadow var(--dur) var(--ease);
  }

  .feature-media:hover,
  .feature-media:focus-visible {
    box-shadow: 0 0 0 2px var(--game-accent);
  }

  .feature-media:focus-visible {
    outline-offset: 4px;
    border-radius: var(--r-lg);
  }

  .genre {
    color: var(--game-accent);
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  /* The one loud thing on the page: the game's name, set huge and condensed. */
  .feature-title {
    font-size: clamp(3.5rem, 9vw, 8.5rem);
    font-weight: 800;
    font-variation-settings: "wdth" 75;
    line-height: 0.86;
    letter-spacing: -0.035em;
    margin-bottom: 1.25rem;
  }

  .tagline {
    font-size: var(--t-md);
    color: var(--muted);
    max-width: 40ch;
  }

  .feature-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 2rem;
  }

  .grid {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1.25rem;
  }

  .card {
    display: flex;
    flex-direction: column;
    height: 100%;
    text-decoration: none;
    border-radius: var(--r-md);
    background: var(--bg-raised);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line);
    transition: box-shadow var(--dur) var(--ease);
  }

  .card:hover {
    box-shadow: 0 0 0 1px var(--line-strong);
  }

  .card:focus-visible {
    border-radius: var(--r-md);
  }

  .card-media {
    display: block;
    aspect-ratio: 16 / 10;
  }

  .card-body {
    display: grid;
    gap: 0.3rem;
    padding: 1rem 1.1rem 1.25rem;
  }

  .card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .card-name {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: var(--t-lg);
    line-height: 1.1;
  }

  .soon-badge {
    color: var(--game-accent);
  }

  .card-genre {
    font-size: var(--t-sm);
    font-weight: 600;
    color: var(--muted);
  }

  .card-tagline {
    font-size: var(--t-sm);
    color: var(--muted);
  }

  .empty {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem 1.5rem;
    padding: 1.5rem;
    border: 1px dashed var(--line-strong);
    border-radius: var(--r-md);
    color: var(--muted);
  }

  .empty p {
    max-width: 56ch;
  }

  @media (max-width: 1100px) {
    .feature {
      grid-template-columns: minmax(220px, 1fr) 1fr;
    }
  }
</style>
