import { fileURLToPath } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// The built-in games live in ../games and are compiled into the app.
const gamesDir = fileURLToPath(new URL("../games", import.meta.url));

// Port 1420 is what tauri.conf.json's devUrl points at.
export default defineConfig({
  plugins: [svelte()],
  resolve: { alias: { "@games": gamesDir }, dedupe: ["three"] },
  assetsInclude: ["**/*.glb", "**/*.ogg"],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "localhost",
    watch: { ignored: ["**/src-tauri/**"] },
    fs: { allow: [".", gamesDir] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: "es2022",
    rollupOptions: { input: { main: "index.html", record: "record.html", gamedev: "gamedev.html", remote: "remote.html" } },
  },
});
