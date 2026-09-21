// Wire types mirroring core/schemas.py and the route payloads.

export type Severity = "none" | "minor" | "major" | "destroyed";
export type DispatchStatus = "proposed" | "reserved" | "in_transit" | "delivered" | "cancelled";
export type Stage = "pending" | "running" | "done" | "failed";

export interface EventDoc {
  id: string;
  name: string;
  disaster_type: string;
  region: string;
  bbox: [number, number, number, number] | null;
  air_transport_verified: boolean;
  cloud_fraction: number;
  stages: Record<"overhead" | "ground" | "fusion" | "graph" | "allocation", Stage> & {
    message: string | null;
  };
}

export interface Derivation {
  value: number;
  formula: string;
  assumptions: string[];
  sources: string[];
}

export interface ZoneNeeds {
  zone_id: string;
  name: string;
  severity: Severity;
  affected_people?: number;
  water_litres?: number;
  sku_demand?: Record<string, number>;
  derivations?: Record<string, Derivation | Record<string, Derivation>>;
}

export interface Citation {
  claim: string;
  source_title: string;
  source_url: string;
  published?: string | null;
}

export interface Dispatch {
  id: string;
  event_id: string;
  zone_id: string;
  from_node_id: string;
  items: { kind: string; sku: string; quantity: number }[];
  transport_mode: "road" | "air";
  route: string[];
  route_distance_km?: number | null;
  priority: number;
  rationale: string;
  citations: Citation[];
  status: DispatchStatus;
}

export interface InventoryItem {
  id: string;
  node_id: string;
  kind: string;
  sku: string;
  quantity: number;
  reserved: number;
  unit: string;
}

export interface Reach {
  zone_id: string | null;
  node_id: string;
  name: string;
  status: "road" | "air" | "unreachable";
  needs_air_verification?: boolean;
  best: null | { mode: string; path: string[]; path_names: string[]; distance_km: number };
}

export interface Parameter {
  name: string;
  value: number;
  unit: string;
  kind: "standard" | "assumption";
  description: string;
  adjusted_by_agent: boolean;
  note?: string | null;
  citations: Citation[];
}

export interface AgentRun {
  _id: string;
  mode: "agent" | "baseline";
  degraded: boolean;
  violations: string[];
  notes: string[];
  summary: string;
  created_at: string;
  research: {
    status: string;
    error?: string | null;
    queries: string[];
    precedents: (Citation & { id: number; parameter: string | null; value?: string | null })[];
  };
  accepted: { parameter: string; value: number; baseline: number; rationale: string;
    precedent_ids: number[] }[];
  rejected: { parameter: string; value: number; reason: string }[];
  parameters: Record<string, Parameter>;
  unmet: { zone_id: string; name: string; sku: string; shortfall: number; reason: string }[];
  zone_rank: { zone_id: string; name: string; rank: number; access: string }[];
}

export interface AllocationPlan {
  run_id: string;
  mode: string;
  degraded: boolean;
  notes: string[];
  summary: string;
  research_status: string;
}

type FC<P> = { type: "FeatureCollection"; features: { type: "Feature"; geometry: GeoJSON.Geometry; properties: P }[] };

export interface GraphGeo {
  nodes: FC<{ id: string; name: string; kind: string; zone_id: string | null; status: string }>;
  edges: FC<{ id: string; status: string; distance_km: number; source: string; from: string; to: string;
    blocked_reason: string | null }>;
}

export interface DamageGeo {
  detections: FC<{ severity: Severity; confidence: number; color: string }>;
  zones: FC<{ id: string; name: string; severity: Severity; damage_score: number; decided_by: string;
    population: number }>;
  severity_breakdown: Record<Severity, number>;
}
