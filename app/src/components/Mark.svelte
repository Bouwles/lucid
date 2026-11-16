<script lang="ts">
  import mark from "../assets/mark.svg?raw";

  let { size = 28 }: { size?: number } = $props();
</script>

<span class="mark" style:width="{size}px" style:height="{size}px">{@html mark}</span>

<style>
  .mark {
    display: inline-block;
    color: var(--accent);
    flex: none;
  }

  .mark :global(svg) {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
