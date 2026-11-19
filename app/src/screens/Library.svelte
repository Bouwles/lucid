<script lang="ts">
  import { onMount } from "svelte";
  import * as engine from "../lib/engine";
  import { formatBytes } from "../lib/format";
  import { inTauri, openPath, revealLabel } from "../lib/shell";
  import type { Library, LibraryItem } from "../lib/types";

  let lib = $state<Library | null>(null);
  let error = $state<string | null>(null);
  let confirming = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let actionError = $state<string | null>(null);

  const groups: { kind: LibraryItem["kind"]; title: string; empty: string }[] = [
    { kind: "models", title: "Dream models", empty: "No dream models yet." },
    { kind: "games", title: "Games", empty: "No games downloaded yet." },
    { kind: "projects", title: "Teach projects", empty: "No projects yet." },
  ];

  const key = (i: LibraryItem) => `${i.kind}/${i.id}`;

  // Show catalogue names ("Drift Kit dream") instead of raw ids where we can.
  let names = $state<Record<string, string>>({});

  function label(item: LibraryItem): string {
    if (item.kind === "models") return names[item.name] ? `${names[item.name]} dream` : item.name;
    if (item.kind === "games") return names[item.id] ?? item.name;
    return item.name;
  }

  async function load(): Promise<void> {
    try {
      lib = await engine.library();
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function remove(item: LibraryItem): Promise<void> {
    busy = key(item);
    actionError = null;
    try {
      await engine.deleteLibraryItem(item.kind, item.id);
      confirming = null;
      await load();
    } catch (e) {
      actionError = `Couldn't delete ${item.name}: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      busy = null;
    }
  }

  onMount(() => {
    void load();
    engine
      .games()
      .then((list) => (names = Object.fromEntries(list.map((g) => [g.id, g.name]))))
      .catch(() => {});
  });
</script>

<div class="page">
  <div class="page-inner">
    <h1 class="page-title">Library</h1>
    <p class="lede">Everything LUCID keeps on this computer, and how much space it takes.</p>

    {#if error}
      <p class="error" role="alert">I couldn't read the library: {error}</p>
      <button class="btn" onclick={load}>Try again</button>
    {:else if !lib}
      <p class="muted" aria-live="polite">Loading…</p>
    {:else}
      <div class="totals">
        <div>
          <p class="big num">{formatBytes(lib.totalBytes)}</p>
          <p class="muted">downloaded or trained</p>
        </div>
        <div>
          <p class="big num">{formatBytes(lib.freeBytes)}</p>
          <p class="muted">free on this disk</p>
        </div>
        {#if lib.dataDir && inTauri}
          <button class="btn small" onclick={() => lib?.dataDir && openPath(lib.dataDir)}>{revealLabel}</button>
        {/if}
      </div>

      {#if actionError}
        <p class="error" role="alert">{actionError}</p>
      {/if}

      {#each groups as g (g.kind)}
        {@const items = lib.items.filter((i) => i.kind === g.kind)}
        <section class="section" aria-labelledby="lib-{g.kind}">
          <h2 id="lib-{g.kind}" class="section-title">{g.title}</h2>
          {#if items.length === 0}
            <p class="muted">{g.empty}</p>
          {:else}
            <ul class="rows">
              {#each items as item (key(item))}
                <li class="row">
                  <div class="what">
                    <span class="name">{label(item)}</span>
                    {#if item.bundled}<span class="badge muted">Comes with LUCID</span>{/if}
                  </div>
                  <span class="size num">{formatBytes(item.sizeBytes)}</span>
                  <div class="acts">
                    {#if item.path && inTauri}
                      <button class="btn small" onclick={() => item.path && openPath(item.path)}>{revealLabel}</button>
                    {/if}
                    {#if confirming === key(item)}
                      <span class="confirm">Delete {label(item)}?</span>
                      <button class="btn small danger" onclick={() => remove(item)} disabled={busy === key(item)}
                        >Delete</button
                      >
                      <button class="btn small" onclick={() => (confirming = null)}>Keep</button>
                    {:else}
                      <button
                        class="btn small danger"
                        disabled={item.bundled}
                        title={item.bundled ? "This one comes with LUCID and can't be deleted" : undefined}
                        onclick={() => (confirming = key(item))}>Delete</button
                      >
                    {/if}
                  </div>
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      {/each}
    {/if}
  </div>
</div>

<style>
  .totals {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 1.5rem 3.5rem;
    margin-top: 2.5rem;
  }

  .big {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: var(--t-xl);
    line-height: 1;
    margin-bottom: 0.3rem;
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    border-top: 1px solid var(--line);
  }

  .row {
    display: grid;
    grid-template-columns: 1fr 110px auto;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--line);
  }

  .what {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0;
  }

  .name {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .size {
    color: var(--muted);
    text-align: right;
  }

  .acts {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .confirm {
    font-size: var(--t-sm);
  }

  .error {
    color: var(--danger);
    margin: 1rem 0;
  }
</style>
