// Shapes from docs/engine-api.md. Keep in step with the engine.

export interface EngineInfo {
  port: number;
  token: string;
  version: string;
}

export interface Hardware {
  os: string;
  arch: string;
  device: string;
  gpu: string | null;
  memory_gb: number | null;
  onnx_providers: string[];
  dream_fps: number | null;
  dream_fps_measured: boolean;
  can_train: boolean;
  training_pack_installed: boolean;
  summary: string;
}

export interface Control {
  keys: string;
  action: string;
}

export interface Game {
  id: string;
  name: string;
  genre: string;
  tagline: string;
  description: string;
  accent: string | null;
  status: "available" | "coming_soon";
  controls: Control[];
  game: { downloadable: boolean; installed: boolean; size_mb: number | null; path?: string | null };
  dream: {
    installed: boolean;
    bundled: boolean;
    size_mb: number | null;
    quality: string[];
    downloadable: boolean;
  };
  media: { cover: string | null; clip: string | null };
}

export type JobState = "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: string;
  params: Record<string, unknown>;
  state: JobState;
  progress: number;
  message: string;
  eta_s: number | null;
  created_utc: string;
  updated_utc: string;
  error: string | null;
}

export interface Settings {
  seen_intro?: boolean;
  last_dream?: string | null;
  recording_consent?: boolean;
  sampler_steps?: number | null;
  [key: string]: unknown;
}

/** One row in the library, normalised from whatever the engine sends. */
export interface LibraryItem {
  kind: "models" | "games" | "projects";
  id: string;
  name: string;
  sizeBytes: number | null;
  bundled: boolean;
  path: string | null;
}

export interface Library {
  items: LibraryItem[];
  totalBytes: number | null;
  freeBytes: number | null;
  dataDir: string | null;
}

export const ACTIVE_JOB_STATES: JobState[] = ["queued", "running", "paused"];
