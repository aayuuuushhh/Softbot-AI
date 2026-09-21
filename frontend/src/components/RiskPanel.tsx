// frontend/src/components/RiskPanel.tsx

import { FOCUS_ZONE_COUNT, type ZoneRisk } from "@/lib/api";
import { AiWorkingLine, Shimmer } from "./AiWorking";

/*
  Risk level is a judgement the deterministic pipeline cannot make: scoring.py
  ranks zones by damage counts, and the LLM reads those counts against collapse
  risk, likely entrapment and access. The two are shown side by side on purpose
  so a coordinator can see where the model's read diverges from the raw rank.
*/
const LEVEL_STYLES: Record<string, { chip: string; rule: string }> = {
  critical: { chip: "bg-red-50 text-red-800 ring-red-200", rule: "bg-red-700" },
  high: {
    chip: "bg-orange-50 text-orange-800 ring-orange-200",
    rule: "bg-orange-700",
  },
  moderate: {
    chip: "bg-yellow-50 text-yellow-800 ring-yellow-200",
    rule: "bg-yellow-600",
  },
  low: {
    chip: "bg-green-50 text-green-800 ring-green-200",
    rule: "bg-green-600",
  },
};

const FALLBACK = {
  chip: "bg-slate-100 text-slate-700 ring-slate-200",
  rule: "bg-slate-400",
};

function levelStyle(level: string) {
  return LEVEL_STYLES[level.toLowerCase()] ?? FALLBACK;
}

export default function RiskPanel({
  risks,
  loading,
}: {
  risks: ZoneRisk[];
  loading: boolean;
}) {
  return (
    <section className="overflow-hidden rounded-xl border border-diq-line bg-white shadow-sm shadow-slate-900/5">
      <header className="border-b border-diq-line bg-slate-50 px-5 py-3.5">
        <h3 className="text-base font-semibold text-diq-ink">
          Zone risk assessment
        </h3>
        <p className="mt-0.5 text-xs text-diq-muted">
          How dangerous each zone is, read from its damage counts.
        </p>
      </header>

      {loading && (
        <div className="px-5 py-5">
          <AiWorkingLine label="Assessing zone risk" />

          {/*
            One placeholder per zone the plan will cover, so the panel is the
            right height before the answer lands.
          */}
          <ul className="mt-4 space-y-4">
            {Array.from({ length: FOCUS_ZONE_COUNT }, (_, i) => (
              <li key={i} className="relative pl-4">
                <Shimmer className="absolute inset-y-0 left-0 w-1 rounded-none" />

                <div className="flex items-baseline justify-between gap-3">
                  <Shimmer className="h-4 w-16" />
                  <Shimmer className="h-4 w-24" />
                </div>

                <Shimmer className="mt-2 h-3 w-full" />
                <Shimmer className="mt-1.5 h-3 w-3/5" />
              </li>
            ))}
          </ul>
        </div>
      )}

      {!loading && risks.length === 0 && (
        <p className="px-5 py-8 text-sm text-diq-muted">
          Run damage analysis to assess zone risk.
        </p>
      )}

      {!loading && risks.length > 0 && (
        <ul className="divide-y divide-diq-line">
          {risks.map((risk) => {
            const style = levelStyle(risk.risk_level);

            return (
              <li key={risk.zone_rank} className="relative py-4 pl-5 pr-5">
                <span
                  className={`absolute inset-y-0 left-0 w-1 ${style.rule}`}
                  aria-hidden="true"
                />

                <div className="flex items-baseline justify-between gap-3">
                  <p className="text-sm font-semibold text-diq-ink">
                    Zone {risk.zone_rank}
                  </p>

                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${style.chip}`}
                    >
                      {risk.risk_level}
                    </span>
                    <span className="tabular text-sm font-semibold text-diq-ink">
                      {risk.risk_score.toFixed(1)}
                    </span>
                  </div>
                </div>

                <p className="mt-1.5 text-sm leading-6 text-slate-700">
                  {risk.rationale}
                </p>

                {risk.primary_hazards.length > 0 && (
                  <ul className="mt-2 flex flex-wrap gap-1.5">
                    {risk.primary_hazards.map((hazard) => (
                      <li
                        key={hazard}
                        className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700"
                      >
                        {hazard}
                      </li>
                    ))}
                  </ul>
                )}

                {(risk.population_at_risk || risk.access_notes) && (
                  <dl className="mt-2.5 space-y-1 text-xs leading-5">
                    {risk.population_at_risk && (
                      <div className="flex gap-2">
                        <dt className="shrink-0 text-diq-muted">At risk</dt>
                        <dd className="text-slate-700">
                          {risk.population_at_risk}
                        </dd>
                      </div>
                    )}
                    {risk.access_notes && (
                      <div className="flex gap-2">
                        <dt className="shrink-0 text-diq-muted">Access</dt>
                        <dd className="text-slate-700">{risk.access_notes}</dd>
                      </div>
                    )}
                  </dl>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
