// Client for the LUCID engine (docs/engine-api.md): token-authenticated HTTP
// plus the dream WebSocket, all on 127.0.0.1.

import { engineInfo } from "./shell";
import type {
  EngineInfo,
  Game,
  Hardware,
  Job,
  Library,
  LibraryItem,
  Settings,
} from "./types";

let info: EngineInfo | null = null;

/** Ask the shell for the engine's port and token. Throws if the engine failed. */
export async function connect(): Promise<EngineInfo> {
  info = await engineInfo();
  // Browser dev mode only knows the port and token; ask the engine for its version.
  if (info.version === "dev") {
    try {
      info = { ...info, version: (await health()).version };
    } catch {
      /* keep "dev" */
    }
  }
  return info;
}

export function current(): EngineInfo {
  if (!info) throw new Error("Engine not connected yet");
  return info;
}

function base(): string {
  return `http://127.0.0.1:${current().port}`;
}

export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${current().token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const res = await fetch(base() + path, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    throw new EngineError(detail || `HTTP ${res.status}`, res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** URL for catalogue media (`/media/...`), carrying the token as a query parameter. */
export function mediaUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^(data|blob):/.test(path)) return path;
  const url = new URL(path, base());
  url.searchParams.set("token", current().token);
  return url.toString();
}

export function dreamSocketUrl(): string {
  const { port, token } = current();
  return `ws://127.0.0.1:${port}/dream?token=${encodeURIComponent(token)}`;
}

export const health = () => request<{ status: string; version: string; uptime_s: number }>("/health");
export const hardware = () => request<Hardware>("/hardware");
export const games = () => request<Game[] | { games: Game[] }>("/games").then(unwrapList<Game>("games"));
export const game = (id: string) => request<Game>(`/games/${encodeURIComponent(id)}`);
export const jobs = () => request<Job[] | { jobs: Job[] }>("/jobs").then(unwrapList<Job>("jobs"));
export const job = (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`);
export const startJob = (kind: string, params: Record<string, unknown>) =>
  request<Job>("/jobs", { method: "POST", body: JSON.stringify({ kind, params }) });
export const controlJob = (id: string, action: "pause" | "resume" | "cancel") =>
  request<Job>(`/jobs/${encodeURIComponent(id)}/${action}`, { method: "POST" });
export const getSettings = () => request<Settings>("/settings");
export const putSettings = (patch: Partial<Settings>) =>
  request<Settings>("/settings", { method: "PUT", body: JSON.stringify(patch) });
export const deleteLibraryItem = (kind: LibraryItem["kind"], id: string) =>
  request<unknown>(`/library/${kind}/${encodeURIComponent(id)}`, { method: "DELETE" });

function unwrapList<T>(key: string) {
  return (v: T[] | Record<string, T[]>): T[] => (Array.isArray(v) ? v : (v[key] ?? []));
}

// The contract doesn't pin the /library shape down, so accept the likely
// spellings (bytes or MB, list or keyed object) and normalise them here.
type Raw = Record<string, unknown>;

function bytesFrom(o: Raw, stem: string): number | null {
  const b = o[`${stem}_bytes`] ?? (stem === "size" ? o.bytes : undefined);
  if (typeof b === "number") return b;
  const mb = o[`${stem}_mb`];
  if (typeof mb === "number") return mb * 1024 * 1024;
  const gb = o[`${stem}_gb`];
  if (typeof gb === "number") return gb * 1024 ** 3;
  return null;
}

export async function library(): Promise<Library> {
  const raw = await request<Raw>("/library");
  const items: LibraryItem[] = [];
  for (const kind of ["models", "games", "projects"] as const) {
    const list = raw[kind];
    const rows: Raw[] = Array.isArray(list)
      ? (list as Raw[])
      : list && typeof list === "object"
        ? Object.entries(list as Record<string, Raw>).map(([id, v]) => ({ id, ...v }))
        : [];
    for (const r of rows) {
      const id = String(r.id ?? r.name ?? "");
      items.push({
        kind,
        id,
        name: String(r.name ?? r.title ?? r.game ?? id),
        sizeBytes: bytesFrom(r, "size"),
        bundled: Boolean(r.bundled ?? r.read_only ?? false),
        path: typeof r.path === "string" ? r.path : null,
      });
    }
  }
  const disk = (raw.disk && typeof raw.disk === "object" ? raw.disk : raw) as Raw;
  return {
    items,
    totalBytes: bytesFrom(raw, "total") ?? bytesFrom(raw, "used"),
    freeBytes: bytesFrom(disk, "free"),
    dataDir: typeof raw.data_dir === "string" ? raw.data_dir : null,
  };
}
