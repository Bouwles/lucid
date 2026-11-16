import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// Port 1420 is what tauri.conf.json's devUrl points at.
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "localhost",
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: "es2022",
  },
});
