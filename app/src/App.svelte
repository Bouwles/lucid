<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import * as engine from "./lib/engine";
  import { router, go } from "./lib/router.svelte";
  import { loadSettings } from "./lib/settings.svelte";
  import { onEngineExited, smokeMode } from "./lib/shell";
  import { smokeDreamOpening, smokeFail, startSmoke } from "./lib/smoke";
  import { dur } from "./lib/motion";
  import Rail from "./components/Rail.svelte";
  import EngineError from "./screens/EngineError.svelte";
  import FirstLaunch from "./screens/FirstLaunch.svelte";
  import Home from "./screens/Home.svelte";
  import GamePage from "./screens/GamePage.svelte";
  import DreamPlayer from "./screens/DreamPlayer.svelte";
  import GamePlayer from "./screens/GamePlayer.svelte";
  import Teach from "./screens/Teach.svelte";
  import Library from "./screens/Library.svelte";
  import Settings from "./screens/Settings.svelte";

  type Boot =
    | { state: "booting" }
    | { state: "ready"; version: string }
    | { state: "error"; message: string; detail: string };

  let boot = $state<Boot>({ state: "booting" });
  // Only show the "starting" screen if the engine takes a moment, to avoid a flash.
  let showStarting = $state(false);

  const fullWindow = $derived(router.route.name === "dream" || router.route.name === "play" || router.route.name === "intro");

  function errorText(e: unknown): string {
    return e instanceof Error ? e.message : String(e);
  }

  async function start(): Promise<void> {
    boot = { state: "booting" };
    showStarting = false;
    const timer = setTimeout(() => (showStarting = true), 400);
    try {
      const smoke = await smokeMode();
      if (smoke) startSmoke();
      const info = await engine.connect();
      const s = await loadSettings();
      boot = { state: "ready", version: info.version };
      if (smoke) {
        smokeDreamOpening();
        const list = await engine.games();
        const first = list.find((g) => g.dream.installed);
        if (first) go(`/dream/${first.id}`, true);
        else smokeFail("no installed dream to open");
        return;
      }
      if (!s.seen_intro && router.route.name === "home") go("/intro", true);
    } catch (e) {
      const text = errorText(e);
      const [first, ...rest] = text.split("\n");
      boot = { state: "error", message: first, detail: rest.join("\n") };
      smokeFail(`engine failed to start: ${first}`);
    } finally {
      clearTimeout(timer);
    }
  }

  onMount(() => {
    void start();
    let unlisten: (() => void) | undefined;
    void onEngineExited((e) => {
      boot = {
        state: "error",
        message: `The engine stopped unexpectedly (exit code ${e.code ?? "unknown"}).`,
        detail: e.stderr_tail,
      };
      smokeFail(`engine exited with code ${e.code}`);
    }).then((u) => (unlisten = u));
    return () => unlisten?.();
  });
</script>

{#if boot.state === "error"}
  <EngineError message={boot.message} detail={boot.detail} onretry={start} />
{:else if boot.state === "booting"}
  {#if showStarting}
    <div class="starting" in:fade={{ duration: dur(200) }}>
      <p>Starting the engine…</p>
    </div>
  {/if}
{:else}
  <div class="shell" class:full={fullWindow}>
    {#if !fullWindow}
      <Rail version={boot.version} />
    {/if}
    <main id="main">
      {#key router.key}
        <div class="view" in:fade={{ duration: dur(160) }}>
          {#if router.route.name === "home"}
            <Home />
          {:else if router.route.name === "intro"}
            <FirstLaunch />
          {:else if router.route.name === "game"}
            <GamePage id={router.route.id} />
          {:else if router.route.name === "dream"}
            <DreamPlayer id={router.route.id} />
          {:else if router.route.name === "play"}
            <GamePlayer id={router.route.id} />
          {:else if router.route.name === "teach"}
            <Teach />
          {:else if router.route.name === "library"}
            <Library />
          {:else if router.route.name === "settings"}
            <Settings />
          {/if}
        </div>
      {/key}
    </main>
  </div>
{/if}

<style>
  .shell {
    display: grid;
    grid-template-columns: 216px 1fr;
    height: 100%;
  }

  .shell.full {
    grid-template-columns: 1fr;
  }

  main,
  .view {
    height: 100%;
    min-width: 0;
    overflow: hidden;
  }

  .starting {
    height: 100%;
    display: grid;
    place-items: center;
    color: var(--muted);
  }
</style>
