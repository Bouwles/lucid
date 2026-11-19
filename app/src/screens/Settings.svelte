<script lang="ts">
  import { onMount } from "svelte";
  import * as engine from "../lib/engine";
  import { formatBytes } from "../lib/format";
  import { href } from "../lib/router.svelte";
  import { saveSettings, settings } from "../lib/settings.svelte";
  import { appVersion } from "../lib/shell";
  import type { Hardware, Library } from "../lib/types";

  const STEP_CHOICES = [3, 4, 6];

  let hw = $state<Hardware | null>(null);
  let lib = $state<Library | null>(null);
  let version = $state("");
  let saveError = $state<string | null>(null);

  const consent = $derived(Boolean(settings.value.recording_consent));
  const steps = $derived(settings.value.sampler_steps ?? null);

  async function save(patch: Record<string, unknown>): Promise<void> {
    saveError = null;
    try {
      await saveSettings(patch);
    } catch (e) {
      saveError = `Couldn't save that setting: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  onMount(() => {
    engine.hardware().then((h) => (hw = h)).catch(() => {});
    engine.library().then((l) => (lib = l)).catch(() => {});
    appVersion().then((v) => (version = v)).catch(() => {});
  });
</script>

<div class="page">
  <div class="page-inner narrow">
    <h1 class="page-title">Settings</h1>
    {#if saveError}<p class="error" role="alert">{saveError}</p>{/if}

    <section class="section" aria-labelledby="s-hw">
      <h2 id="s-hw" class="section-title">This computer</h2>
      {#if hw}
        <p class="summary">{hw.summary}</p>
        <dl class="facts">
          <dt>System</dt><dd>{hw.os} ({hw.arch})</dd>
          <dt>Chip</dt><dd>{hw.gpu ?? "Unknown"}</dd>
          <dt>Memory</dt><dd>{hw.memory_gb != null ? `${hw.memory_gb} GB` : "Unknown"}</dd>
          <dt>Compute</dt><dd>{hw.device}</dd>
          <dt>Dream speed</dt>
          <dd class="num">
            {hw.dream_fps != null ? `about ${Math.round(hw.dream_fps)} FPS` : "not measured yet"}{hw.dream_fps != null
              ? hw.dream_fps_measured
                ? " (measured)"
                : " (estimated)"
              : ""}
          </dd>
          <dt>Inference</dt><dd>{hw.onnx_providers.join(", ") || "None found"}</dd>
          <dt>Training pack</dt><dd>{hw.training_pack_installed ? "Installed" : "Not installed"}</dd>
        </dl>
      {:else}
        <p class="muted">Checking…</p>
      {/if}
    </section>

    <section class="section" aria-labelledby="s-storage">
      <h2 id="s-storage" class="section-title">Storage</h2>
      {#if lib}
        <p>
          Downloads and trained models use <strong class="num">{formatBytes(lib.totalBytes)}</strong>, with
          <strong class="num">{formatBytes(lib.freeBytes)}</strong> free on this disk.
        </p>
      {/if}
      <a class="btn small spaced" href={href("/library")}>Manage in Library</a>
    </section>

    <section class="section" aria-labelledby="s-privacy">
      <h2 id="s-privacy" class="section-title">Privacy</h2>
      <div class="setting">
        <div>
          <p class="label" id="consent-label">Save my play sessions as training data</p>
          <p class="muted help" id="consent-help">
            When this is on, playing a built-in game records the frames and your key presses so they can help train
            better dreams. Recordings stay on this computer. LUCID never uploads anything and has no telemetry.
          </p>
        </div>
        <button
          class="switch"
          role="switch"
          aria-checked={consent}
          aria-labelledby="consent-label"
          aria-describedby="consent-help"
          onclick={() => save({ recording_consent: !consent })}
        >
          <span class="knob"></span>
        </button>
      </div>
    </section>

    <section class="section" aria-labelledby="s-perf">
      <h2 id="s-perf" class="section-title">Performance</h2>
      <div class="setting">
        <div>
          <label class="label" for="steps">Default sampler steps</label>
          <p class="muted help">
            How many denoising steps the dream takes per frame. Fewer is faster, more is cleaner. You can also change
            this while playing.
          </p>
        </div>
        <select
          id="steps"
          value={steps === null ? "" : String(steps)}
          onchange={(e) => {
            const v = (e.currentTarget as HTMLSelectElement).value;
            save({ sampler_steps: v === "" ? null : Number(v) });
          }}
        >
          <option value="">Model default</option>
          {#each STEP_CHOICES as n (n)}
            <option value={String(n)}>{n}</option>
          {/each}
        </select>
      </div>
    </section>

    <section class="section" aria-labelledby="s-about">
      <h2 id="s-about" class="section-title">About</h2>
      <dl class="facts">
        <dt>App</dt><dd class="num">{version || "…"}</dd>
        <dt>Engine</dt><dd class="num">{engine.current().version}</dd>
      </dl>
      <h3 class="sub">Credits and licences</h3>
      <p class="muted help">
        The built-in games will list every asset pack they use, with its licence, in games/CREDITS.md. The interface
        uses Bricolage Grotesque and Instrument Sans, both under the SIL Open Font License.
      </p>
      <p class="made">Made by Paul Nercessian</p>
    </section>
  </div>
</div>

<style>
  .narrow {
    max-width: 760px;
  }

  .summary {
    font-size: var(--t-md);
    margin-bottom: 1.25rem;
  }

  .facts {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 0.5rem 1.5rem;
    margin: 0;
    font-size: var(--t-sm);
  }

  .facts dt {
    color: var(--muted);
  }

  .facts dd {
    margin: 0;
  }

  .spaced {
    margin-top: 1rem;
  }

  .setting {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 2rem;
  }

  .label {
    font-weight: 600;
    display: block;
    margin-bottom: 0.3rem;
  }

  .help {
    font-size: var(--t-sm);
    max-width: 58ch;
  }

  .switch {
    flex: none;
    width: 52px;
    height: 30px;
    border-radius: 999px;
    border: 1px solid var(--line-strong);
    background: var(--bg-raised);
    padding: 3px;
    cursor: pointer;
    transition: background var(--dur) var(--ease);
  }

  .knob {
    display: block;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--muted);
    transition:
      transform var(--dur) var(--ease),
      background var(--dur) var(--ease);
  }

  .switch[aria-checked="true"] {
    background: var(--accent);
    border-color: var(--accent);
  }

  .switch[aria-checked="true"] .knob {
    transform: translateX(22px);
    background: var(--accent-ink);
  }

  .switch:focus-visible {
    border-radius: 999px;
  }

  select {
    flex: none;
    min-height: 40px;
    padding: 0 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--line-strong);
    background: var(--bg-raised);
  }

  .sub {
    font-size: var(--t-md);
    margin: 1.75rem 0 0.5rem;
  }

  .made {
    margin-top: 2rem;
    font-family: var(--font-display);
    font-weight: 700;
    font-size: var(--t-md);
  }

  .error {
    color: var(--danger);
  }
</style>
