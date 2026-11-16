import * as engine from "./engine";
import type { Settings } from "./types";

export const settings = $state<{ value: Settings }>({ value: {} });

export async function loadSettings(): Promise<Settings> {
  settings.value = await engine.getSettings();
  return settings.value;
}

export async function saveSettings(patch: Partial<Settings>): Promise<void> {
  settings.value = { ...settings.value, ...patch };
  settings.value = await engine.putSettings(patch);
}
