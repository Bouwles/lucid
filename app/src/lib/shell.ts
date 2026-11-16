// Thin wrappers over the Tauri commands in src-tauri/src/lib.rs, with
// fallbacks so `npm run dev` also works in a plain browser.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { EngineInfo } from "./types";

export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export async function engineInfo(): Promise<EngineInfo> {
  if (inTauri) return invoke<EngineInfo>("engine_info");
  const port = Number(import.meta.env.VITE_ENGINE_PORT);
  const token = import.meta.env.VITE_ENGINE_TOKEN as string | undefined;
  if (!port || !token) {
    throw new Error(
      "Running in a browser without an engine. Start the engine, then run npm run dev with VITE_ENGINE_PORT and VITE_ENGINE_TOKEN set from its ready line.",
    );
  }
  return { port, token, version: "dev" };
}

export async function retryEngine(): Promise<void> {
  if (inTauri) await invoke("retry_engine");
}

export async function engineLogsPath(): Promise<string> {
  if (inTauri) return invoke<string>("engine_logs_path");
  return "";
}

export async function smokeMode(): Promise<boolean> {
  if (inTauri) return invoke<boolean>("smoke_mode");
  return new URLSearchParams(location.search).has("smoke");
}

export async function smokeResult(ok: boolean, detail: string): Promise<void> {
  if (inTauri) {
    await invoke("smoke_result", { ok, detail });
  } else {
    console.info(`[smoke] ok=${ok} ${detail}`);
  }
}

export async function openPath(path: string): Promise<void> {
  if (inTauri) await invoke("open_path", { path });
  else console.info(`[open_path] ${path}`);
}

export async function launchGame(id: string, path: string): Promise<void> {
  if (!inTauri) throw new Error("Games can only be launched from the desktop app.");
  await invoke("launch_game", { id, path });
}

export async function appVersion(): Promise<string> {
  if (!inTauri) return "0.2.0 (browser)";
  const { getVersion } = await import("@tauri-apps/api/app");
  return getVersion();
}

export interface EngineExited {
  code: number | null;
  stderr_tail: string;
}

export async function onEngineExited(cb: (e: EngineExited) => void): Promise<UnlistenFn> {
  if (!inTauri) return () => {};
  return listen<EngineExited>("engine-exited", (e) => cb(e.payload));
}

export async function onGameExited(
  cb: (e: { id: string; code: number | null }) => void,
): Promise<UnlistenFn> {
  if (!inTauri) return () => {};
  return listen<{ id: string; code: number | null }>("game-exited", (e) => cb(e.payload));
}

const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
export const revealLabel = /Mac/.test(ua)
  ? "Show in Finder"
  : /Windows/.test(ua)
    ? "Show in Explorer"
    : "Show in folder";
