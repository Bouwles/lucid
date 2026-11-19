<script lang="ts">
  import { mediaUrl } from "../lib/engine";
  import { isVideo } from "../lib/format";
  import { reducedMotion } from "../lib/motion";
  import type { Game } from "../lib/types";

  // `playing` is driven by the parent (hover or keyboard focus on the card).
  // `autoplay` keeps the clip running, for big media on the game page.
  let {
    game,
    playing = false,
    autoplay = false,
    fit = "cover",
  }: { game: Game; playing?: boolean; autoplay?: boolean; fit?: "cover" | "contain" } = $props();

  const cover = $derived(mediaUrl(game.media.cover));
  const clip = $derived(mediaUrl(game.media.clip));
  const clipIsVideo = $derived(isVideo(clip));
  const live = $derived((playing || autoplay) && !reducedMotion());

  // Small sources (64 px dream clips) get nearest-neighbour scaling so every
  // model pixel stays a crisp block. Large screenshots scale smoothly.
  let small = $state(true);
  let video = $state<HTMLVideoElement>();

  function measure(e: Event): void {
    const el = e.currentTarget as HTMLImageElement | HTMLVideoElement;
    const w = el instanceof HTMLImageElement ? el.naturalWidth : el.videoWidth;
    if (w) small = w < 480;
  }

  $effect(() => {
    if (!video) return;
    if (live) void video.play().catch(() => {});
    else video.pause();
  });

  // For a GIF clip, swap the <img> source while live. With no cover, a GIF
  // still shows (its first frame when reduced motion is on, as best we can).
  const imgSrc = $derived(clip && !clipIsVideo && (live || !cover) ? clip : cover);
</script>

<div class="media" class:pixel={small} class:contain={fit === "contain"} style:--game-accent={game.accent ?? "var(--accent)"}>
  {#if clipIsVideo && clip}
    <video
      bind:this={video}
      src={clip}
      poster={cover ?? undefined}
      muted
      loop
      playsinline
      preload="metadata"
      onloadedmetadata={measure}
      aria-hidden="true"
    ></video>
  {:else if imgSrc}
    <img src={imgSrc} alt="" onload={measure} draggable="false" />
  {:else}
    <div class="plate" aria-hidden="true">
      <span>{game.name}</span>
    </div>
  {/if}
</div>

<style>
  .media {
    position: relative;
    width: 100%;
    height: 100%;
    background: var(--bg-deep);
    overflow: hidden;
  }

  img,
  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .contain img,
  .contain video {
    object-fit: contain;
  }

  .pixel img,
  .pixel video {
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }

  .plate {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: flex-end;
    padding: 1.25rem;
    background:
      repeating-linear-gradient(
        90deg,
        color-mix(in srgb, var(--game-accent) 9%, transparent) 0 1px,
        transparent 1px 24px
      ),
      var(--bg-deep);
  }

  .plate span {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: clamp(2rem, 6vw, 4.5rem);
    font-variation-settings: "wdth" 75;
    line-height: 0.9;
    color: color-mix(in srgb, var(--game-accent) 30%, var(--bg-deep));
  }
</style>
