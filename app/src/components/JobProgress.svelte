<script lang="ts">
  import { formatEta } from "../lib/format";
  import type { Job } from "../lib/types";

  let {
    job,
    onpause,
    onresume,
    oncancel,
  }: { job: Job; onpause: () => void; onresume: () => void; oncancel: () => void } = $props();

  const pct = $derived(Math.round(Math.min(1, Math.max(0, job.progress)) * 100));
</script>

<div class="job" aria-live="polite">
  <div class="row">
    <p class="message">{job.message || (job.state === "queued" ? "Waiting to start" : "Working")}</p>
    <p class="num pct">{pct}%</p>
  </div>
  <div
    class="bar"
    role="progressbar"
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow={pct}
    aria-label="Download progress"
  >
    <div class="fill" class:paused={job.state === "paused"} style:width="{pct}%"></div>
  </div>
  <div class="row">
    <p class="muted small">
      {#if job.state === "paused"}Paused{:else if job.state === "queued"}Queued{:else}{formatEta(job.eta_s)}{/if}
    </p>
    <div class="actions">
      {#if job.state === "paused"}
        <button class="btn small" onclick={onresume}>Resume</button>
      {:else}
        <button class="btn small" onclick={onpause} disabled={job.state === "queued"}>Pause</button>
      {/if}
      <button class="btn small danger" onclick={oncancel}>Cancel</button>
    </div>
  </div>
</div>

<style>
  .job {
    display: grid;
    gap: 0.6rem;
    max-width: 460px;
  }

  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .message {
    font-weight: 600;
  }

  .small {
    font-size: var(--t-sm);
  }

  .bar {
    height: 6px;
    border-radius: 3px;
    background: var(--line);
    overflow: hidden;
  }

  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 400ms linear;
  }

  .fill.paused {
    background: var(--muted);
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }
</style>
