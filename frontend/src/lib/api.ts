export interface DamageCounts {
  none: number;
  minor: number;
  major: number;
  destroyed: number;
}

export interface BuildingCounts {
  none: number;
  minor: number;
  major: number;
  destroyed: number;
}

export interface Zone {
  rank: number;
  bbox: number[];
  damage_counts: DamageCounts;
  building_counts: BuildingCounts;
  priority_score: number;
  /** Mean predicted-class probability over the zone's building pixels.
   *  Only the `pytorch` inference mode produces probabilities; null otherwise. */
  confidence?: number | null;
  centroid_lat?: number | null;
  centroid_lng?: number | null;
}

export interface AnalysisSummary {
  total_building_pixels: number;
  total_buildings: number;
  destroyed_pct: number;
  major_pct: number;
  minor_pct: number;
}

export interface AnalysisResult {
  zones: Zone[];
  summary: AnalysisSummary;
  mask_base64?: string | null;
  pair_id?: string | null;
  inference_mode: string;
  geo_available: boolean;
  /** "wgs84" = real map coords; "image" = pixel-space zone map. */
  geo_mode?: "wgs84" | "image";
  /** Pixel size of the damage mask / source image [width, height]. */
  image_size?: [number, number] | null;
  geo_message?: string | null;
}

export interface DemoPair {
  id: string;
  disaster_type: string;
  pre_image: string;
  post_image: string;
}

/** A line of the responder roster the allocation plan divides up. */
export interface ResourceUnit {
  name: string;
  quantity: number;
  unit: string;
}

export interface ZoneRisk {
  zone_rank: number;
  /** critical | high | moderate | low */
  risk_level: string;
  risk_score: number;
  primary_hazards: string[];
  population_at_risk: string;
  access_notes: string;
  rationale: string;
}

export interface ResourceAssignment {
  zone_rank: number;
  resource: string;
  quantity: number;
  unit: string;
  /** immediate | urgent | scheduled */
  urgency: string;
  justification: string;
}

export interface BriefResponse {
  brief: string;
  /** gemini | gemini-fallback | stub */
  source: string;
  risk_analysis: ZoneRisk[];
  resource_allocation: ResourceAssignment[];
  reserve_notes: string;
}

/*
  How many ranked zones the dashboard presents. Must match MAX_PLAN_ZONES in
  backend/app/services/narrator.py: the brief, the risk read and the allocation
  plan all cover exactly these zones, so the overlay and the map have to draw
  the same ones. Showing sixteen markers beside a five-zone risk panel made the
  two read as different analyses of different places.
*/
export const FOCUS_ZONE_COUNT = 5;

/** The roster the dashboard starts with. Editable in Mission Control. */
export const DEFAULT_RESOURCES: ResourceUnit[] = [
  { name: "Search & rescue teams", quantity: 8, unit: "teams" },
  { name: "Medical units", quantity: 6, unit: "units" },
  { name: "Heavy equipment", quantity: 4, unit: "machines" },
  { name: "Relief supply trucks", quantity: 10, unit: "trucks" },
  { name: "Shelter capacity", quantity: 1200, unit: "people" },
];

export interface HealthResponse {
  status: string;
  inference_mode: string;
  demo_pairs: number;
}

/*
  Important:
  Your backend is FastAPI on port 8000.
  Your frontend is Next.js on port 3000.

  If image URLs accidentally point to localhost:3000, the images fail.
  So every API/demo image request must go to the backend base URL.
*/
const RAW_API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://127.0.0.1:8000";

export const API_BASE = RAW_API_BASE.replace(/\/$/, "");

/*
  Without a timeout a wedged backend leaves the UI spinning forever. Analyze is
  the slow one: `docker` inference runs the TF baseline and takes ~2 min/pair.
*/
const TIMEOUT_ANALYZE_MS = 300_000;
const TIMEOUT_BRIEF_MS = 90_000;
const TIMEOUT_DEFAULT_MS = 30_000;
/*
  Per-attempt timeout for connect probes (must stay below the quiet-poll budget).
  A request to a sleeping host is *held* by the platform proxy while the
  container boots, so aborting early just throws away a probe that was about to
  be answered and sends another one at an already-busy box. Wait it out.
*/
const TIMEOUT_CONNECT_ATTEMPT_MS = 25_000;

/*
  AbortSignal.timeout is missing on older browsers (Safari < 16, Chrome < 103).
  Calling it unconditionally made every fetch throw a TypeError immediately,
  so the dashboard latched "Frontend Only" forever on those browsers no matter
  how healthy the backend was. Fall back to a plain AbortController + setTimeout.
*/
function timeoutSignal(ms: number): AbortSignal {
  if (typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(
    () => controller.abort(new DOMException("signal timed out", "TimeoutError")),
    ms,
  );
  return controller.signal;
}

function mergeAbortSignals(
  ...signals: Array<AbortSignal | undefined>
): AbortSignal | undefined {
  const active = signals.filter((s): s is AbortSignal => Boolean(s));
  if (active.length === 0) return undefined;
  if (active.length === 1) return active[0];
  if (typeof AbortSignal.any === "function") {
    return AbortSignal.any(active);
  }
  const controller = new AbortController();
  for (const signal of active) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      return controller.signal;
    }
    signal.addEventListener("abort", () => controller.abort(signal.reason), {
      once: true,
    });
  }
  return controller.signal;
}

async function readError(res: Response, fallback: string): Promise<string> {
  // Never surface raw backend bodies (paths, stack fragments) in the UI.
  const statusHints: Record<number, string> = {
    400: "The request was rejected. Check that both images are valid PNG/JPEG under 15 MB.",
    401: "This analysis server requires an API key. Contact the operator if you need access.",
    404: "The requested demo pair or image was not found.",
    413: "The upload is too large for this server.",
    429: "Too many requests. Wait a moment and try again.",
    503: "The analysis service is temporarily unavailable. Try again shortly.",
  };
  if (statusHints[res.status]) {
    return statusHints[res.status];
  }
  try {
    const text = await res.text();
    if (!text) return fallback;
    try {
      const json = JSON.parse(text) as { detail?: unknown };
      if (typeof json.detail === "string" && json.detail.length < 200) {
        // Allow short, intentional FastAPI detail strings only.
        return json.detail;
      }
    } catch {
      /* not JSON */
    }
  } catch {
    /* ignore */
  }
  return fallback;
}

export async function fetchHealth(options?: {
  signal?: AbortSignal;
  timeoutMs?: number;
}): Promise<HealthResponse> {
  const timeoutMs = options?.timeoutMs ?? TIMEOUT_DEFAULT_MS;
  const res = await fetch(`${API_BASE}/health`, {
    cache: "no-store",
    signal: mergeAbortSignals(options?.signal, timeoutSignal(timeoutMs)),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Backend unavailable"));
  }

  return res.json();
}

export async function fetchDemoPairs(options?: {
  signal?: AbortSignal;
  timeoutMs?: number;
}): Promise<DemoPair[]> {
  const timeoutMs = options?.timeoutMs ?? TIMEOUT_DEFAULT_MS;
  const res = await fetch(`${API_BASE}/demo/pairs`, {
    cache: "no-store",
    signal: mergeAbortSignals(options?.signal, timeoutSignal(timeoutMs)),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Failed to load demo pairs"));
  }

  return res.json();
}

/*
  The deployed backend sits on a free tier that sleeps when idle, and a cold
  start takes longer than any single request timeout. One failed probe therefore
  means "still waking", not "offline" — so keep probing until the budget runs
  out, and only then call it offline.

  The budget is wall-clock, not a retry count: a sleeping host leaves requests
  hanging until they time out, while a host that is simply down refuses them
  instantly. A fixed number of retries would spend minutes in the first case and
  a couple of seconds in the second — far too little to outlast a boot.

  The budget must outlast a Render free-tier spin-up. At 90s it did not: the
  probe gave up, the dashboard latched FRONTEND ONLY, and the backend then
  finished booting — so an upload analyzed fine (it waits up to 300s) while the
  header still claimed the backend was offline. Three minutes covers the boot.
*/
const CONNECT_BUDGET_MS = 180_000;
const RETRY_MIN_MS = 1_500;
const RETRY_MAX_MS = 5_000;

export class ConnectionCancelledError extends Error {
  constructor() {
    super("Connection cancelled");
    this.name = "ConnectionCancelledError";
  }
}

export async function connectToBackend(options?: {
  signal?: AbortSignal;
  budgetMs?: number;
  attemptTimeoutMs?: number;
}): Promise<{ health: HealthResponse; pairs: DemoPair[] }> {
  const {
    signal,
    budgetMs = CONNECT_BUDGET_MS,
    attemptTimeoutMs = TIMEOUT_CONNECT_ATTEMPT_MS,
  } = options ?? {};
  const deadline = Date.now() + budgetMs;

  let lastError: unknown = new Error("Backend unavailable");
  let backoff = RETRY_MIN_MS;

  for (;;) {
    if (signal?.aborted) throw new ConnectionCancelledError();

    try {
      // Health alone defines online/offline. Demo pairs are best-effort so a
      // pairs failure cannot force FRONTEND ONLY while analyze still works.
      const health = await fetchHealth({ signal, timeoutMs: attemptTimeoutMs });
      let pairs: DemoPair[] = [];
      try {
        pairs = await fetchDemoPairs({ signal, timeoutMs: attemptTimeoutMs });
      } catch (pairsErr) {
        if (signal?.aborted) throw new ConnectionCancelledError();
        if (
          pairsErr instanceof DOMException &&
          pairsErr.name === "AbortError" &&
          signal?.aborted
        ) {
          throw new ConnectionCancelledError();
        }
        pairs = [];
      }
      return { health, pairs };
    } catch (err) {
      if (signal?.aborted || err instanceof ConnectionCancelledError) {
        throw new ConnectionCancelledError();
      }
      if (err instanceof DOMException && err.name === "AbortError" && signal?.aborted) {
        throw new ConnectionCancelledError();
      }
      lastError = err;
    }

    if (signal?.aborted) throw new ConnectionCancelledError();
    if (Date.now() + backoff >= deadline) throw lastError;

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, backoff);
      if (!signal) return;
      const onAbort = () => {
        clearTimeout(timer);
        reject(new ConnectionCancelledError());
      };
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    });
    backoff = Math.min(backoff * 1.5, RETRY_MAX_MS);
  }
}

export async function analyzeDemoPair(pairId: string): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("demo_pair_id", pairId);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
    signal: timeoutSignal(TIMEOUT_ANALYZE_MS),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Failed to analyze demo pair"));
  }

  return res.json();
}

export async function analyzeUpload(
  pre: File,
  post: File,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("pre_image", pre);
  form.append("post_image", post);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
    signal: timeoutSignal(TIMEOUT_ANALYZE_MS),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Failed to analyze uploaded images"));
  }

  return res.json();
}

export async function fetchBrief(
  analysis: AnalysisResult,
  context?: string,
  resources?: ResourceUnit[],
): Promise<BriefResponse> {
  const res = await fetch(`${API_BASE}/brief`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ analysis, context, resources }),
    signal: timeoutSignal(TIMEOUT_BRIEF_MS),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Failed to generate brief"));
  }

  return res.json();
}

export async function fetchReportPdf(
  analysis: AnalysisResult,
  brief: string,
): Promise<Blob> {
  const res = await fetch(`${API_BASE}/report/pdf`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ analysis, brief }),
    signal: timeoutSignal(TIMEOUT_DEFAULT_MS),
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Failed to generate PDF report"));
  }

  return res.blob();
}

export function demoImageUrl(filenameOrPath: string): string {
  if (!filenameOrPath) return "";

  const value = filenameOrPath.trim();

  /*
    Absolute URLs are only accepted when they already point at this API.
    Reject arbitrary http(s) hosts to avoid open redirects from a poisoned
    /demo/pairs payload.
  */
  if (value.startsWith("http://") || value.startsWith("https://")) {
    try {
      const url = new URL(value);
      const base = new URL(API_BASE);
      if (url.origin === base.origin) {
        return value;
      }
    } catch {
      /* fall through */
    }
    return "";
  }

  /*
    Case 2:
    Backend returned:
    /demo/images/demo_pre_disaster.png
  */
  if (value.startsWith("/demo/images/")) {
    return `${API_BASE}${value}`;
  }

  /*
    Case 3:
    Backend returned another root-relative path.
    This still needs to go to FastAPI, not Next.js.
  */
  if (value.startsWith("/")) {
    return `${API_BASE}${value}`;
  }

  /*
    Case 4:
    Backend returned only:
    demo_pre_disaster.png
  */
  return `${API_BASE}/demo/images/${encodeURIComponent(value)}`;
}