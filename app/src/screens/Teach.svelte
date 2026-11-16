<script lang="ts">
  const footage = [
    { type: "Simple 2D (fixed or side-scrolling camera)", min: "30 minutes", rec: "2 hours", example: "Platformers, runners" },
    { type: "3D with a following camera", min: "1 hour", rec: "4 hours", example: "Racing games, third-person" },
    { type: "3D with free camera (first-person)", min: "2 hours", rec: "6+ hours", example: "Shooters" },
  ];
</script>

<div class="page">
  <div class="page-inner">
    <h1 class="page-title">Teach LUCID your game</h1>
    <p class="lede">
      This is coming in a later version. Nothing here works yet, but here's how it will work, so you can start
      thinking about footage.
    </p>

    <section class="section" aria-labelledby="paths-title">
      <h2 id="paths-title" class="section-title">Two ways in</h2>
      <p class="muted intro">
        A world model needs to know which keys were pressed in every frame. A normal gameplay video doesn't have that,
        so the two paths give different results.
      </p>
      <div class="paths">
        <article class="path">
          <h3>Record with LUCID</h3>
          <p class="tag">Recommended</p>
          <p>
            You play your game while LUCID records that one window and the keys you press. The dream then responds
            properly to your controls. Recording only happens while you've started it, with a visible indicator and a
            hotkey to stop.
          </p>
        </article>
        <article class="path">
          <h3>Upload a video</h3>
          <p class="tag">Approximate controls</p>
          <p>
            You drop in gameplay videos you already have. LUCID has to work out the key presses from the video itself,
            so the dream looks right but the controls are rougher. A short recorded session alongside the videos helps
            a lot.
          </p>
        </article>
      </div>
    </section>

    <section class="section" aria-labelledby="footage-title">
      <h2 id="footage-title" class="section-title">How much footage it needs</h2>
      <table class="data">
        <thead>
          <tr>
            <th scope="col">Game type</th>
            <th scope="col">Minimum</th>
            <th scope="col">Recommended</th>
            <th scope="col">For example</th>
          </tr>
        </thead>
        <tbody>
          {#each footage as row (row.type)}
            <tr>
              <th scope="row">{row.type}</th>
              <td class="num">{row.min}</td>
              <td class="num">{row.rec}</td>
              <td class="muted">{row.example}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <p class="muted foot">
        These are starting guidelines, not promises. I'll calibrate them from experiments on the built-in games. With
        less than the minimum you can still try, and the result is labelled preview quality.
      </p>
    </section>

    <section class="section" aria-labelledby="video-title">
      <h2 id="video-title" class="section-title">What makes good footage</h2>
      <ul class="tips">
        <li>480p or higher, at 30 FPS or more.</li>
        <li>One game per project, with the same graphics settings throughout.</li>
        <li>Continuous play: no edits, menus, loading screens, webcam overlays or stream alerts.</li>
        <li>Varied play, including mistakes and deaths. LUCID only dreams what it has seen.</li>
      </ul>
      <p class="muted foot">Everything stays on your computer. LUCID never uploads footage or models.</p>
    </section>
  </div>
</div>

<style>
  .intro {
    max-width: 62ch;
    margin-bottom: 1.5rem;
  }

  .paths {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.25rem;
  }

  .path {
    padding: 1.5rem;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
  }

  .path h3 {
    font-size: var(--t-lg);
  }

  .tag {
    font-size: var(--t-sm);
    font-weight: 600;
    color: var(--accent);
    margin: 0.35rem 0 0.9rem;
  }

  .path p:last-child {
    color: var(--muted);
  }

  table.data th[scope="row"] {
    color: var(--text);
    font-weight: 500;
  }

  .foot {
    margin-top: 1rem;
    max-width: 66ch;
    font-size: var(--t-sm);
  }

  .tips {
    margin: 0;
    padding-left: 1.2rem;
    display: grid;
    gap: 0.4rem;
    max-width: 66ch;
  }
</style>
