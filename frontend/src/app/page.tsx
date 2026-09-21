// frontend/src/app/page.tsx

"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import BriefPanel from "@/components/BriefPanel";
import DamageCanvas from "@/components/DamageCanvas";
import ZoneMap from "@/components/ZoneMap";
import RiskPanel from "@/components/RiskPanel";
import AllocationPanel from "@/components/AllocationPanel";
import RosterEditor from "@/components/RosterEditor";
import {
  analyzeDemoPair,
  analyzeUpload,
  connectToBackend,
  ConnectionCancelledError,
  demoImageUrl,
  fetchBrief,
  fetchReportPdf,
  DEFAULT_RESOURCES,
  type AnalysisResult,
  type DemoPair,
  type ResourceAssignment,
  type ResourceUnit,
  type ZoneRisk,
} from "@/lib/api";
import { FALLBACK_DEMO_PAIRS } from "@/lib/demoPairs";

const MIN_ZOOM = 1;
const MAX_ZOOM = 3;

function getFriendlyError(error: unknown) {
  // AbortSignal.timeout rejects with a TimeoutError DOMException, whose message
  // ("signal timed out") means nothing to a coordinator staring at the screen.
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return "The analysis server took too long to respond. Large image pairs can take a couple of minutes — wait a moment, then try again.";
  }

  const message = error instanceof Error ? error.message : String(error);

  if (
    message.toLowerCase().includes("failed to fetch") ||
    message.toLowerCase().includes("networkerror")
  ) {
    return "Could not reach the analysis server. It may still be waking up — this can take up to a minute on first use.";
  }

  return message;
}

function FilePicker({
  label,
  file,
  onSelect,
}: {
  label: string;
  file: File | null;
  onSelect: (f: File | null) => void;
}) {
  const id = useId();

  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="block font-label text-[9px] uppercase tracking-[0.18em] text-diq-muted"
      >
        {label}
      </label>

      <label
        htmlFor={id}
        className="flex cursor-pointer items-center gap-2 rounded-lg border border-diq-line bg-slate-50 px-2.5 py-2 text-xs transition hover:border-diq-blue hover:bg-slate-50"
      >
        <span className="shrink-0 rounded-md bg-slate-100 px-2 py-1 text-[11px] font-semibold text-diq-ink">
          Browse
        </span>

        <span
          className={`min-w-0 truncate ${
            file ? "text-diq-ink" : "text-diq-muted"
          }`}
        >
          {file ? file.name : "No file selected"}
        </span>
      </label>

      <input
        id={id}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

const MAX_CLIENT_UPLOAD_BYTES = 15 * 1024 * 1024;
const ALLOWED_CLIENT_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
  "image/tiff",
]);

function validateUploadFile(file: File): string | null {
  if (file.size > MAX_CLIENT_UPLOAD_BYTES) {
    return "Each image must be 15 MB or smaller.";
  }
  if (file.type && !ALLOWED_CLIENT_TYPES.has(file.type)) {
    return "Use PNG, JPEG, WebP, or TIFF images.";
  }
  return null;
}

function UploadDropBox({
  preFile,
  postFile,
  onPreSelect,
  onPostSelect,
}: {
  preFile: File | null;
  postFile: File | null;
  onPreSelect: (f: File | null) => void;
  onPostSelect: (f: File | null) => void;
}) {
  const assignDropped = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const list = Array.from(files).slice(0, 2);
    list.forEach((file, index) => {
      const err = validateUploadFile(file);
      if (err) {
        window.alert(err);
        return;
      }
      if (index === 0) onPreSelect(file);
      if (index === 1) onPostSelect(file);
    });
  };

  return (
    <div
      className="rounded-2xl border border-dashed border-diq-line bg-slate-50 p-3"
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        assignDropped(e.dataTransfer.files);
      }}
    >
      <div className="flex flex-col items-center justify-center text-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-diq-line bg-slate-50 text-xl text-diq-muted">
          ⇧
        </div>

        <p className="mt-3 max-w-[190px] text-sm font-semibold leading-5 text-slate-700">
          Drag & drop before/after images here
        </p>

        <p className="mt-1 text-[11px] text-diq-muted">or select files below</p>

        <div className="my-3 h-px w-full bg-diq-line/40" />
      </div>

      <div className="space-y-2.5">
        <FilePicker
          label="Before image"
          file={preFile}
          onSelect={(f) => {
            if (f) {
              const err = validateUploadFile(f);
              if (err) {
                window.alert(err);
                return;
              }
            }
            onPreSelect(f);
          }}
        />

        <FilePicker
          label="After image"
          file={postFile}
          onSelect={(f) => {
            if (f) {
              const err = validateUploadFile(f);
              if (err) {
                window.alert(err);
                return;
              }
            }
            onPostSelect(f);
          }}
        />
      </div>

      <p className="mt-3 text-center text-[10px] text-diq-muted">
        PNG, JPG, WebP, TIFF up to 15MB each
      </p>
    </div>
  );
}

function EmptyImageState() {
  return (
    <div className="flex h-full min-h-[720px] items-center justify-center bg-slate-50">
      <div className="px-6 text-center">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-diq-line bg-slate-50 text-xl text-diq-muted">
          ◇
        </div>

        <p className="mt-4 font-label text-xs uppercase tracking-[0.22em] text-diq-muted">
          Awaiting Imagery
        </p>

        <p className="mt-2 text-sm text-slate-600">
          Load a demo pair or upload before/after images.
        </p>
      </div>
    </div>
  );
}

type StepState = "waiting" | "processing" | "done";

/*
  Only stages the frontend can actually observe. This panel used to list six
  fake steps ("Aligning Images", "Extracting Buildings", …) all driven by one
  boolean — pure theater, and exactly the kind of overclaiming the rest of the
  app has been scrubbing out. The backend runs as two calls we can genuinely
  track: /analyze (inference + zone scoring) and /brief (AI narration).
*/
function PipelineStatus({
  loading,
  analysis,
  briefLoading,
  brief,
  seconds,
}: {
  loading: boolean;
  analysis: AnalysisResult | null;
  briefLoading: boolean;
  brief: string | null;
  seconds: number | null;
}) {
  /*
    `loading` spans the whole analyze-then-brief run, so it alone cannot tell
    step 1 from step 2: once the analysis result lands, step 1 is done even
    though the run (now fetching the brief) is still in flight. runAnalysis
    clears `analysis` up front, so a stale result can't fake a completed step.
  */
  const analysisState: StepState = analysis
    ? "done"
    : loading
      ? "processing"
      : "waiting";
  const briefState: StepState = briefLoading
    ? "processing"
    : brief
      ? "done"
      : "waiting";

  const steps: { label: string; detail: string; state: StepState }[] = [
    {
      label: "Damage Analysis",
      detail: "Inference, zone scoring & georeferencing",
      state: analysisState,
    },
    {
      label: "AI Situation Brief",
      detail: "Plain-language summary of the ranked zones",
      state: briefState,
    },
  ];

  return (
    <div className="rounded-2xl border border-diq-line bg-slate-50 p-4">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-100 text-[11px] font-bold text-slate-600">
          2
        </span>

        <p className="font-label text-xs uppercase tracking-[0.18em] text-slate-600">
          Processing Pipeline
        </p>
      </div>

      <div className="mt-4 space-y-3">
        {steps.map((step, index) => (
          <div key={step.label} className="relative flex items-start gap-3">
            {index !== steps.length - 1 && (
              <span
                className={`absolute left-[11px] top-6 h-8 w-px ${
                  step.state === "done"
                    ? "bg-green-400/80"
                    : step.state === "processing"
                      ? "bg-blue-50"
                      : "bg-diq-line/70"
                }`}
              />
            )}

            <span
              className={`z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[11px] font-bold ${
                step.state === "done"
                  ? "border-green-300 bg-green-50 text-green-800"
                  : step.state === "processing"
                    ? "border-diq-blue bg-orange-50 text-diq-muted"
                    : "border-diq-line bg-white text-diq-muted"
              }`}
            >
              {step.state === "done" ? "✓" : index + 1}
            </span>

            <div>
              <p
                className={`text-sm font-semibold ${
                  step.state !== "waiting" ? "text-diq-ink" : "text-diq-muted"
                }`}
              >
                {step.label}
              </p>

              <p
                className={`text-xs ${
                  step.state === "done"
                    ? "text-green-700"
                    : step.state === "processing"
                      ? "text-diq-blue"
                      : "text-slate-600"
                }`}
              >
                {step.state === "done"
                  ? "Completed"
                  : step.state === "processing"
                    ? "Processing…"
                    : step.detail}
              </p>
            </div>
          </div>
        ))}
      </div>

      {analysisState === "done" && (
        <div className="mt-5 rounded-xl border border-green-200 bg-green-50 px-4 py-4 text-center">
          <p className="text-sm font-bold text-green-800">
            ✓ Analysis Complete
          </p>
          {seconds != null && (
            <p className="mt-1 text-xs text-diq-muted">
              {seconds.toFixed(1)} seconds
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/*
  A cold start can run past any figure we quote, and a spinner that just says
  "waking" with nothing moving reads as hung — so count the wait out loud. The
  elapsed seconds are the difference between "it's working on it" and "it's
  broken"; this component is only mounted while connecting, so mount time is
  the start of the wait.
*/
function BackendConnectingNotice() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const id = setInterval(
      () => setElapsed(Math.round((Date.now() - startedAt) / 1000)),
      1_000,
    );
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl border border-diq-line bg-blue-50 px-3 py-2.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-1 h-2.5 w-2.5 shrink-0 animate-pulse rounded-full bg-blue-400" />

        <div>
          <p className="text-xs font-black uppercase tracking-[0.12em] text-blue-800">
            Waking backend · {elapsed}s
          </p>
          <p className="mt-1 text-[11px] leading-5 text-blue-800">
            The free-tier server sleeps when idle and a cold start takes up to
            two minutes. This page keeps retrying on its own — demo pairs appear
            the moment it answers, with no refresh needed.
          </p>
        </div>
      </div>
    </div>
  );
}

function BackendOfflineNotice({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber-50 text-[10px] font-black text-amber-800">
          !
        </span>

        <div>
          <p className="text-xs font-black uppercase tracking-[0.12em] text-amber-800">
            Backend unreachable
          </p>
          <p className="mt-1 text-[11px] leading-5 text-amber-800">
            Could not reach the analysis server. It may still be starting up.
          </p>

          <button
            type="button"
            onClick={onRetry}
            className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] font-bold text-amber-800 transition hover:border-amber-300 hover:bg-amber-50"
          >
            ↻ Retry connection
          </button>
        </div>
      </div>
    </div>
  );
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-3">
      <p className="text-sm font-bold text-red-800">Action blocked</p>
      <p className="mt-1 text-xs leading-5 text-red-800/70">{message}</p>
    </div>
  );
}

function MapControls({
  onZoomIn,
  onZoomOut,
  canZoomIn,
  canZoomOut,
}: {
  onZoomIn: () => void;
  onZoomOut: () => void;
  canZoomIn: boolean;
  canZoomOut: boolean;
}) {
  return (
    <div className="absolute bottom-4 right-4 z-20 flex flex-col overflow-hidden rounded-lg border border-diq-line bg-slate-50 shadow-sm shadow-slate-900/5">
      <button
        type="button"
        onClick={onZoomIn}
        disabled={!canZoomIn}
        className="flex h-9 w-9 items-center justify-center border-b border-diq-line text-lg font-bold text-diq-ink transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Zoom in"
      >
        +
      </button>

      <button
        type="button"
        onClick={onZoomOut}
        disabled={!canZoomOut}
        className="flex h-9 w-9 items-center justify-center text-lg font-bold text-diq-ink transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Zoom out"
      >
        −
      </button>
    </div>
  );
}

function FullscreenButton({
  targetRef,
}: {
  targetRef: React.RefObject<HTMLElement | null>;
}) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(document.fullscreenElement === targetRef.current);
    };

    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, [targetRef]);

  const toggleFullscreen = () => {
    const el = targetRef.current;
    if (!el) return;

    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void el.requestFullscreen?.();
    }
  };

  return (
    <button
      type="button"
      onClick={toggleFullscreen}
      className="absolute bottom-4 right-16 z-20 flex h-9 w-9 items-center justify-center rounded-lg border border-diq-line bg-slate-50 text-sm text-diq-ink shadow-sm shadow-slate-900/5 transition hover:bg-slate-100"
      aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
    >
      {isFullscreen ? "⤡" : "⛶"}
    </button>
  );
}

export default function HomePage() {
  /*
    Seed with the bundled pair list so the demo is clickable from the first
    paint. Gating the dropdown on a successful health probe meant a cold or
    restarting backend left nothing to demo at all; the live /demo/pairs
    response replaces this as soon as it arrives.
  */
  const [pairs, setPairs] = useState<DemoPair[]>(FALLBACK_DEMO_PAIRS);
  const [selectedPair, setSelectedPair] = useState<string>(
    FALLBACK_DEMO_PAIRS[0]?.id ?? "",
  );
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [postUrl, setPostUrl] = useState<string>("");
  const [preUrl, setPreUrl] = useState<string>("");
  const [brief, setBrief] = useState<string | null>(null);
  const [briefSource, setBriefSource] = useState<string | null>(null);
  const [risks, setRisks] = useState<ZoneRisk[]>([]);
  const [allocation, setAllocation] = useState<ResourceAssignment[]>([]);
  const [reserveNotes, setReserveNotes] = useState("");
  const [roster, setRoster] = useState<ResourceUnit[]>(DEFAULT_RESOURCES);
  /*
    The roster the plan on screen was actually built from. Editing the roster
    changes the next run, not the plan already rendered, so the allocation
    panel has to measure "committed" against the numbers the model divided —
    otherwise dropping the team count to 2 made a live plan read "8 of 2
    committed".
  */
  const [planRoster, setPlanRoster] = useState<ResourceUnit[]>(DEFAULT_RESOURCES);
  const [loading, setLoading] = useState(false);
  const [briefLoading, setBriefLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOffline, setBackendOffline] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [analysisSeconds, setAnalysisSeconds] = useState<number | null>(null);
  /** After a successful API call, ignore failed quiet probes until this time. */
  const onlineUntilRef = useRef(0);

  /*
    Everything the narrator produced, cleared as one unit. Three call sites used
    to clear the brief by hand and forget the risk read and the allocation, so
    switching demo pairs blanked the brief but left the previous city's dispatch
    plan on screen next to an empty map.
  */
  const clearFindings = useCallback(() => {
    setBrief(null);
    setBriefSource(null);
    setRisks([]);
    setAllocation([]);
    setReserveNotes("");
  }, []);

  const markOnline = useCallback(() => {
    onlineUntilRef.current = Date.now() + 120_000;
    setBackendOffline(false);
  }, []);

  const beforePanelRef = useRef<HTMLDivElement>(null);
  const afterPanelRef = useRef<HTMLDivElement>(null);
  const [afterZoom, setAfterZoom] = useState(1);

  const zoomIn = useCallback(
    () => setAfterZoom((z) => Math.min(Math.round((z + 0.25) * 100) / 100, MAX_ZOOM)),
    [],
  );
  const zoomOut = useCallback(
    () => setAfterZoom((z) => Math.max(Math.round((z - 0.25) * 100) / 100, MIN_ZOOM)),
    [],
  );

  useEffect(() => {
    setAfterZoom(MIN_ZOOM);
  }, [postUrl]);

  // Object URLs created from uploaded Files stay alive until explicitly revoked.
  // Track the pre/post blob independently so an uploaded image can be previewed
  // (and zoomed/expanded) the moment it is selected, then released when it is
  // replaced — whether by another upload or a demo pair's plain HTTP URLs.
  const preBlobRef = useRef<string | null>(null);
  const postBlobRef = useRef<string | null>(null);

  const setPreImage = useCallback((url: string, isBlob: boolean) => {
    if (preBlobRef.current) URL.revokeObjectURL(preBlobRef.current);
    preBlobRef.current = isBlob ? url : null;
    setPreUrl(url);
  }, []);

  const setPostImage = useCallback((url: string, isBlob: boolean) => {
    if (postBlobRef.current) URL.revokeObjectURL(postBlobRef.current);
    postBlobRef.current = isBlob ? url : null;
    setPostUrl(url);
  }, []);

  const setImageUrls = useCallback(
    (pre: string, post: string, isBlob: boolean) => {
      setPreImage(pre, isBlob);
      setPostImage(post, isBlob);
    },
    [setPreImage, setPostImage],
  );

  useEffect(() => {
    return () => {
      if (preBlobRef.current) URL.revokeObjectURL(preBlobRef.current);
      if (postBlobRef.current) URL.revokeObjectURL(postBlobRef.current);
    };
  }, []);

  /** True while a background poll is running; user-driven retries ignore it. */
  const quietProbeInFlightRef = useRef(false);

  const connect = useCallback(
    (options?: { signal?: AbortSignal; budgetMs?: number; quiet?: boolean }) => {
      const { signal, budgetMs, quiet } = options ?? {};

      /*
        A probe can run for its whole budget, which outlasts the poll interval
        below. Without this guard the polls pile up: several probe loops end up
        retrying at once against a host that is still booting, which is exactly
        when it can least afford the extra load. Only the polls are suppressed —
        an explicit "Retry connection" must always fire.
      */
      if (quiet) {
        if (quietProbeInFlightRef.current) return Promise.resolve();
        quietProbeInFlightRef.current = true;
      }

      // A quiet probe is the background poll below: it must not flip the UI
      // back to "Waking backend" on every tick once we already said offline.
      if (!quiet) setConnecting(true);

      return connectToBackend({ signal, budgetMs })
        .then(({ pairs: p }) => {
          if (signal?.aborted) return;
          // The live list wins when it has content; an empty/failed pairs
          // fetch must not clobber the bundled fallback the demo relies on.
          if (p.length > 0) {
            setPairs(p);
            setSelectedPair((cur) => cur || p[0].id);
          }
          markOnline();
        })
        .catch((err) => {
          if (signal?.aborted) return;
          if (err instanceof ConnectionCancelledError) return;
          if (err instanceof Error && err.message === "Connection cancelled") return;
          // Sticky online: a single quiet probe must not flip FRONTEND ONLY
          // right after analyze/brief succeeded.
          if (quiet && Date.now() < onlineUntilRef.current) return;
          // Keep the bundled pairs and selection: a failed probe means the
          // backend is unreachable *right now*, not that the demo set is gone.
          setBackendOffline(true);
        })
        .finally(() => {
          if (quiet) quietProbeInFlightRef.current = false;
          else if (!signal?.aborted) setConnecting(false);
        });
    },
    [markOnline]
  );

  useEffect(() => {
    const controller = new AbortController();
    // Budget comes from api.ts: it must outlast a free-tier cold start.
    void connect({ signal: controller.signal });
    return () => controller.abort();
  }, [connect]);

  /*
    Once the budget is spent we show "offline", but a backend can still come up
    later (a slow boot, a restarted dev server). Keep probing quietly so the
    dashboard heals itself instead of demanding a refresh.
  */
  useEffect(() => {
    if (!backendOffline || connecting) return;

    const controller = new AbortController();
    const id = setInterval(() => {
      void connect({
        signal: controller.signal,
        // Must exceed the per-attempt timeout (25s) so cold starts can retry.
        budgetMs: 60_000,
        quiet: true,
      });
    }, 10_000);

    return () => {
      controller.abort();
      clearInterval(id);
    };
  }, [backendOffline, connecting, connect]);

  const loadDemoPair = useCallback(() => {
    setError(null);
    setAnalysis(null);
    clearFindings();
    setBriefLoading(false);

    /*
      Deliberately NOT gated on backendOffline: the pair list is bundled and
      the image URLs are known, and a request to a waking Render host is held
      by its proxy until boot completes. Blocking the click here turned a
      30-second wake into "the demo doesn't work"; letting it through means
      the imagery appears the moment the backend answers.
    */
    if (!selectedPair) {
      setError("Select a demo pair first.");
      return;
    }

    const pair = pairs.find((p) => p.id === selectedPair);

    if (!pair) {
      setError(`Selected demo pair was not found: ${selectedPair}`);
      return;
    }

    setPreFile(null);
    setPostFile(null);
    setImageUrls(demoImageUrl(pair.pre_image), demoImageUrl(pair.post_image), false);
  }, [selectedPair, pairs, setImageUrls, clearFindings]);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    setAnalysis(null);
    clearFindings();
    setAnalysisSeconds(null);

    try {
      let result: AnalysisResult;

      const startedAt = performance.now();

      if (preFile && postFile) {
        // preUrl/postUrl were already set to preview blobs on file selection.
        result = await analyzeUpload(preFile, postFile);
      } else if (selectedPair) {
        result = await analyzeDemoPair(selectedPair);
        const pair = pairs.find((p) => p.id === selectedPair);

        if (pair) {
          setImageUrls(demoImageUrl(pair.pre_image), demoImageUrl(pair.post_image), false);
        }
      } else if (backendOffline) {
        throw new Error(
          "The analysis server is not reachable yet. It may still be waking up — try again in a moment."
        );
      } else {
        throw new Error("Select a demo pair or upload before/after images.");
      }

      setAnalysisSeconds((performance.now() - startedAt) / 1000);
      setAnalysis(result);
      markOnline();
      setBriefLoading(true);

      // Refresh health/pairs in the background after a successful analyze.
      void connect({ budgetMs: 45_000, quiet: true });

      /*
        No hardcoded regional context: this used to send "Pakistan disaster
        response context" with every analysis, so the AI brief narrated
        Pakistani response priorities over Mexican earthquake imagery. The
        analysis JSON already carries the pair id and disaster type — let the
        narrator work from what is actually on screen.
      */
      const briefResp = await fetchBrief(result, undefined, roster);
      setBrief(briefResp.brief);
      setBriefSource(briefResp.source);
      setRisks(briefResp.risk_analysis ?? []);
      setAllocation(briefResp.resource_allocation ?? []);
      setReserveNotes(briefResp.reserve_notes ?? "");
      // Freeze the roster this plan divided, so later roster edits do not
      // rewrite the denominators on a plan that is already on screen.
      setPlanRoster(roster);
      markOnline();
    } catch (e) {
      setError(getFriendlyError(e));
    } finally {
      setLoading(false);
      setBriefLoading(false);
    }
  }, [
    preFile,
    postFile,
    selectedPair,
    pairs,
    backendOffline,
    roster,
    setImageUrls,
    connect,
    markOnline,
    clearFindings,
  ]);

  const handleDownloadReport = useCallback(async () => {
    if (!analysis || !brief) return;

    setReportLoading(true);
    setError(null);

    try {
      const blob = await fetchReportPdf(analysis, brief);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `uddhar-report-${analysis.pair_id ?? "upload"}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      markOnline();
    } catch (e) {
      setError(getFriendlyError(e));
    } finally {
      setReportLoading(false);
    }
  }, [analysis, brief, markOnline]);

  const canAnalyze = Boolean((preFile && postFile) || selectedPair);

  /*
    A disabled button explains nothing: hovering it just shows a "blocked" cursor,
    which reads as the app refusing to work rather than as the backend still
    booting. Demo pairs come from the backend, so during a cold start there is
    nothing to select and Load Demo Pair is necessarily dead — say so, on the
    button and in its tooltip, instead of leaving the user to guess.
  */
  const loadPairBlockedReason: string | null = loading
    ? "An analysis is running. Wait for it to finish."
    : pairs.length === 0
      ? connecting
        ? "Waking the backend. Demo pairs load by themselves — no refresh needed."
        : "The backend is unreachable, so no demo pairs could be loaded."
      : !selectedPair
        ? "Choose a demo pair from the list first."
        : null;

  return (
    <main className="min-h-screen bg-diq-bg text-diq-ink diq-grid-bg">
      <div className="mx-auto max-w-[1920px] px-4 py-4">
        <header className="mb-4 border-b border-diq-line pb-4">
          <div>
            <div>
              <div>
                <h1 className="text-5xl font-black tracking-tight text-diq-ink md:text-6xl">
                  UDDHAR
                </h1>

                <p className="mt-1.5 whitespace-nowrap text-[13px] font-black uppercase tracking-[0.18em] text-slate-600">
                  See damage. Prioritize relief. Save lives.
                </p>
              </div>
            </div>

          </div>
        </header>

        <section className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
          <aside className="overflow-hidden rounded-xl border border-diq-line bg-diq-panel shadow-sm shadow-slate-900/5">
            <div className="border-b border-diq-line bg-slate-50 px-4 py-3">
              <h2 className="font-label text-xs uppercase tracking-[0.18em] text-slate-700">
                Mission Control
              </h2>
            </div>

            <div className="space-y-3.5 p-4">
              <div>
                <div className="mb-3 flex items-center gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-100 text-[11px] font-bold text-slate-600">
                    1
                  </span>

                  <p className="font-label text-xs uppercase tracking-[0.18em] text-slate-600">
                    Upload Imagery
                  </p>
                </div>

                <UploadDropBox
                  preFile={preFile}
                  postFile={postFile}
                  onPreSelect={(file) => {
                    setPreFile(file);
                    setPreImage(file ? URL.createObjectURL(file) : "", Boolean(file));
                    if (file) {
                      setSelectedPair("");
                      setAnalysis(null);
                      clearFindings();
                    }
                  }}
                  onPostSelect={(file) => {
                    setPostFile(file);
                    setPostImage(file ? URL.createObjectURL(file) : "", Boolean(file));
                    if (file) {
                      setSelectedPair("");
                      setAnalysis(null);
                      clearFindings();
                    }
                  }}
                />
              </div>

              <div className="space-y-3">
                <label htmlFor="demo-pair" className="sr-only">
                  Demo satellite image pair
                </label>
                <select
                  id="demo-pair"
                  aria-label="Demo satellite image pair"
                  value={selectedPair}
                  onChange={(e) => {
                    setSelectedPair(e.target.value);
                    setPreFile(null);
                    setPostFile(null);
                    setPreImage("", false);
                    setPostImage("", false);
                    setAnalysis(null);
                    clearFindings();
                  }}
                  disabled={pairs.length === 0}
                  className="w-full rounded-xl border border-diq-line bg-slate-50 px-3 py-3 text-sm text-diq-ink transition hover:border-diq-line disabled:cursor-not-allowed disabled:text-diq-muted"
                >
                  {pairs.length === 0 && (
                    <option value="">
                      {connecting
                        ? "Waking backend — demo pairs loading…"
                        : "Backend offline — demo pairs unavailable"}
                    </option>
                  )}

                  {pairs.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.disaster_type}: {p.id}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  onClick={loadDemoPair}
                  disabled={loadPairBlockedReason !== null}
                  title={loadPairBlockedReason ?? "Load the selected before/after pair"}
                  className="w-full rounded-lg border border-diq-line bg-white px-4 py-3 text-sm font-semibold text-diq-ink transition hover:border-diq-blue hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {pairs.length === 0 && connecting
                    ? "▣ Waiting for demo pairs…"
                    : "▣ Load Demo Pair"}
                </button>
                {loadPairBlockedReason && (
                  <p className="text-[11px] leading-5 text-diq-muted">
                    {loadPairBlockedReason}
                  </p>
                )}

                <button
                  type="button"
                  onClick={runAnalysis}
                  disabled={loading || !canAnalyze}
                  className="w-full rounded-lg bg-diq-blue px-4 py-3 text-sm font-semibold text-white shadow-sm shadow-slate-900/10 transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {loading ? "Analyzing Damage…" : "⌘ Analyze Damage"}
                </button>
              </div>

              {connecting && <BackendConnectingNotice />}

              {backendOffline && !connecting && (
                <BackendOfflineNotice onRetry={() => void connect()} />
              )}

              {error && <ErrorNotice message={error} />}

              <PipelineStatus
                loading={loading}
                analysis={analysis}
                briefLoading={briefLoading}
                brief={brief}
                seconds={analysisSeconds}
              />
            </div>

            <RosterEditor roster={roster} onChange={setRoster} disabled={loading} />
          </aside>

          <section className="space-y-4">
            <div className="overflow-hidden rounded-xl border border-diq-line bg-diq-panel shadow-sm shadow-slate-900/5">
              <div className="grid min-h-[720px] gap-0 lg:grid-cols-2">
                <div
                  ref={beforePanelRef}
                  className="relative overflow-hidden border-b border-diq-line bg-white lg:border-b-0 lg:border-r"
                >
                  <div className="absolute left-4 top-4 z-20 rounded bg-blue-950/90 px-4 py-2 text-xs font-black uppercase tracking-[0.12em] text-white shadow-sm shadow-slate-900/5">
                    Before Disaster
                  </div>

                  {preUrl ? (
                    /*
                      object-contain, not object-cover: the After panel contains
                      its image (DamageCanvas maps zone boxes onto that exact
                      letterboxed rect, so it cannot cover). Covering here would
                      crop and rescale Before independently, and the two panels
                      would show the same ground at different scales — the one
                      thing a before/after comparison must never do. Same fit on
                      both sides keeps every pixel aligned across the split.
                    */
                    <img
                      src={preUrl}
                      alt="Before disaster satellite imagery"
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : (
                    <EmptyImageState />
                  )}

                  {/*
                    No source credit on the Before panel. It used to name
                    Maxar/xBD, which was true of the xBD demo set and is not
                    true of the imagery shipped in data/demo today — a credit
                    naming the wrong provider is worse than none. Put the real
                    attribution back here once the new demo scenes' source is
                    settled.
                  */}

                  <FullscreenButton targetRef={beforePanelRef} />
                </div>

                <div
                  ref={afterPanelRef}
                  className="relative overflow-hidden bg-white"
                >
                  <div className="absolute left-4 top-4 z-20 max-w-[52%] rounded bg-blue-950/90 px-4 py-2 text-xs font-black uppercase tracking-[0.12em] text-white shadow-sm shadow-slate-900/5">
                    AI Damage Overlay After
                  </div>

                  {postUrl ? (
                    <div
                      className="h-full w-full origin-center transition-transform duration-200 ease-out"
                      style={{ transform: `scale(${afterZoom})` }}
                    >
                      <DamageCanvas postImageUrl={postUrl} analysis={analysis} />
                    </div>
                  ) : (
                    <EmptyImageState />
                  )}

                  <MapControls
                    onZoomIn={zoomIn}
                    onZoomOut={zoomOut}
                    canZoomIn={afterZoom < MAX_ZOOM}
                    canZoomOut={afterZoom > MIN_ZOOM}
                  />
                  <FullscreenButton targetRef={afterPanelRef} />
                </div>
              </div>
            </div>

            <ZoneMap analysis={analysis} postImageUrl={postUrl || undefined} />

            <BriefPanel
              brief={brief}
              source={briefSource}
              loading={briefLoading}
              onDownloadReport={handleDownloadReport}
              reportLoading={reportLoading}
            />

            <div className="grid gap-4 2xl:grid-cols-2">
              <RiskPanel risks={risks} loading={briefLoading} />

              <AllocationPanel
                allocation={allocation}
                reserveNotes={reserveNotes}
                roster={planRoster}
                loading={briefLoading}
              />
            </div>
          </section>

        </section>

      </div>
    </main>
  );
}