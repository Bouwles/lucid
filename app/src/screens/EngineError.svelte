<script lang="ts">
  import { onMount } from "svelte";
  import { engineLogsPath, inTauri, openPath, retryEngine, revealLabel } from "../lib/shell";
  import Mark from "../components/Mark.svelte";

  let { message, detail, onretry }: { message: string; detail: string; onretry: () => void } = $props();

  let logPath = $state("");
  let retrying = $state(false);

  onMount(() => {
    engineLogsPath()
      .then((p) => (logPath = p))
      .catch(() => {});
  });

  async function retry(): Promise<void> {
    retrying = true;
    try {
      await retryEngine();
    } finally {
      retrying = false;
      onretry();
    }
  }
</script>

<div class="wrap">
  <div class="box" role="alert">
    <Mark size={36} />
    <h1>LUCID's engine isn't running</h1>
    <p class="msg">{message}</p>
    {#if detail}
      <pre>{detail}</pre>
    {/if}
    {#if logPath}
      <p class="muted log">
        The full log is in <code>{logPath}</code>
        {#if inTauri}
          <button class="link" onclick={() => openPath(logPath)}>{revealLabel}</button>
        {/if}
      </p>
    {/if}
    <div class="actions">
      <button class="btn primary" onclick={retry} disabled={retrying}>{retrying ? "Restarting…" : "Retry"}</button>
    </div>
  </div>
</div>

<style>
  .wrap {
    height: 100%;
    overflow-y: auto;
    display: grid;
    place-items: center;
    padding: 3rem;
  }

  .box {
    max-width: 680px;
    width: 100%;
    display: grid;
    gap: 1rem;
  }

  h1 {
    font-size: var(--t-xl);
    margin-top: 0.5rem;
  }

  .msg {
    font-size: var(--t-md);
  }

  pre {
    margin: 0;
    padding: 1rem;
    max-height: 240px;
    overflow: auto;
    border-radius: var(--r-sm);
    background: var(--bg-deep);
    border: 1px solid var(--line);
    font-size: var(--t-xs);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--muted);
  }

  code {
    font-size: var(--t-sm);
    word-break: break-all;
    color: var(--text);
  }

  .link {
    background: none;
    border: 0;
    padding: 0;
    margin-left: 0.5rem;
    color: var(--accent);
    font-weight: 600;
    cursor: pointer;
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  .actions {
    margin-top: 0.75rem;
  }
</style>
