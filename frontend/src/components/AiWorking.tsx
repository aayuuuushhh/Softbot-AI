// frontend/src/components/AiWorking.tsx

/*
  Shared "the model is thinking" furniture for the three panels Gemini fills:
  the brief, the zone risk read and the dispatch plan. They are produced by a
  single /brief call, so they must look busy at the same time and in the same
  way — three different spinners for one request reads as three separate
  things going wrong.

  A skeleton rather than a spinner: the panels take a few seconds and their
  shape is known in advance (FOCUS_ZONE_COUNT zones, the roster on screen), so
  showing that shape tells the coordinator what is coming and stops the page
  jumping when it lands.
*/

export function Shimmer({ className = "" }: { className?: string }) {
  return (
    <span
      className={`block animate-pulse rounded bg-slate-200/90 ${className}`}
      aria-hidden="true"
    />
  );
}

export function AiWorkingLine({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative flex h-2.5 w-2.5 shrink-0">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-diq-blue opacity-70" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-diq-blue" />
      </span>

      <p className="font-label text-[11px] uppercase tracking-[0.14em] text-diq-muted">
        {label}
      </p>
    </div>
  );
}
