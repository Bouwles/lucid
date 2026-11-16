<script lang="ts">
  import { router, href } from "../lib/router.svelte";
  import Mark from "./Mark.svelte";

  let { version }: { version: string } = $props();

  const items = [
    { path: "/", label: "Home", match: ["home", "game"] },
    { path: "/teach", label: "Teach LUCID", match: ["teach"] },
    { path: "/library", label: "Library", match: ["library"] },
    { path: "/settings", label: "Settings", match: ["settings"] },
  ];
</script>

<nav class="rail" aria-label="Main">
  <a class="brand" href={href("/")} aria-label="LUCID home">
    <Mark size={30} />
    <span class="word">LUCID</span>
  </a>
  <ul>
    {#each items as item (item.path)}
      <li>
        <a
          href={href(item.path)}
          aria-current={item.match.includes(router.route.name) ? "page" : undefined}>{item.label}</a
        >
      </li>
    {/each}
  </ul>
  <p class="foot">Engine {version}</p>
</nav>

<style>
  .rail {
    display: flex;
    flex-direction: column;
    gap: 2.25rem;
    padding: 1.75rem 1rem 1.25rem;
    border-right: 1px solid var(--line);
    background: var(--bg);
    height: 100%;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0 0.5rem;
    text-decoration: none;
  }

  .word {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 1.6rem;
    font-variation-settings: "wdth" 75;
    letter-spacing: 0.01em;
    line-height: 1;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 2px;
  }

  ul a {
    display: block;
    padding: 0.55rem 0.75rem;
    border-radius: 8px;
    text-decoration: none;
    color: var(--muted);
    font-weight: 500;
    transition:
      color var(--dur) var(--ease),
      background var(--dur) var(--ease);
  }

  ul a:hover {
    color: var(--text);
    background: var(--bg-raised);
  }

  ul a[aria-current="page"] {
    color: var(--text);
    background: var(--bg-raised);
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .foot {
    margin-top: auto;
    padding: 0 0.75rem;
    font-size: var(--t-xs);
    color: var(--faint);
  }
</style>
