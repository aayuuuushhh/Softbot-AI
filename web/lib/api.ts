// Thin typed client for the Uddhar FastAPI backend.
//
// The API base is resolved in the browser at runtime, not baked in at build
// time: NEXT_PUBLIC_API_URL if it was set when building, otherwise the same
// host the dashboard was served from, on port 8765. One Docker image then works
// on any host.

import type {
  AgentRun,
  AllocationPlan,
  Dispatch,
  DispatchStatus,
  EventDoc,
  GraphGeo,
  InventoryItem,
  Reach,
  ZoneNeeds,
  DamageGeo,
} from "./types";

export function apiBase(): string {
  const built = process.env.NEXT_PUBLIC_API_URL;
  if (built) return built.replace(/\/$/, "");
  if (typeof window === "undefined") return "http://localhost:8765";
  return `${window.location.protocol}//${window.location.hostname}:8765`;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => req<{ status: string; mongodb: boolean; cuda: boolean; device: string;
    sat_backend: string; ground_backend: string }>("/health"),
  events: () => req<EventDoc[]>("/api/events"),
  event: (id: string) => req<EventDoc>(`/api/events/${id}`),
  damage: (id: string) => req<DamageGeo>(`/api/events/${id}/damage`),
  graph: (id: string) => req<GraphGeo>(`/api/events/${id}/graph`),
  reachability: (id: string) => req<Reach[]>(`/api/events/${id}/reachability`),
  needs: (id: string) => req<ZoneNeeds[]>(`/api/events/${id}/needs`),
  dispatches: (id: string) => req<Dispatch[]>(`/api/events/${id}/dispatches`),
  inventory: (id: string) => req<InventoryItem[]>(`/api/inventory?event_id=${id}`),
  agentRuns: (id: string) => req<AgentRun[]>(`/api/events/${id}/agent-runs?limit=1`),
  agentRun: (runId: string) => req<AgentRun>(`/api/agent-runs/${runId}`),
  allocate: (id: string) => req<AllocationPlan>(`/api/events/${id}/allocate`, { method: "POST" }),
  approveAll: (id: string) =>
    req<{ reserved: string[]; refused: { id: string; reason: string }[] }>(
      `/api/events/${id}/dispatches/approve`, { method: "POST" }),
  setDispatch: (dispatchId: string, status: DispatchStatus) =>
    req<Dispatch>(`/api/dispatches/${dispatchId}/status?status=${status}`, { method: "PATCH" }),
  setEdge: (edgeId: string, status: string) =>
    req(`/api/edges/${edgeId}?status=${status}`, { method: "PATCH" }),
  setAir: (id: string, verified: boolean) =>
    req(`/api/events/${id}/air-transport?verified=${verified}`, { method: "PATCH" }),
  analyze: (id: string, pre: File, post: File) => {
    const form = new FormData();
    form.append("pre", pre);
    form.append("post", post);
    return req<Record<string, unknown>>(`/api/events/${id}/analyze`, { method: "POST", body: form });
  },
  uploadGround: (id: string, file: File, lat: number, lon: number, reporter?: string) => {
    const form = new FormData();
    form.append("event_id", id);
    form.append("lat", String(lat));
    form.append("lon", String(lon));
    if (reporter) form.append("reporter", reporter);
    form.append("image", file);
    return req<Record<string, unknown>>("/api/upload/ground", { method: "POST", body: form });
  },
  reportUrl: (id: string) => `${apiBase()}/api/events/${id}/report.pdf`,
  summaryUrl: (id: string) => `${apiBase()}/api/events/${id}/summary.txt`,
  overlayUrl: (id: string) => `${apiBase()}/api/events/${id}/overlay.png`,
};
