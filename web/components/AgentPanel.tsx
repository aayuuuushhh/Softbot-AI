"use client";

import type { AgentRun } from "@/lib/types";

export default function AgentPanel({ run }: { run: AgentRun | null }) {
  if (!run) return <p className="text-sm text-slate-500">No allocation run yet.</p>;
  const params = Object.values(run.parameters ?? {});
  return (
    <div className="space-y-4 text-sm">
      <div>
        <div className="flex items-center gap-2">
          <span className={`rounded px-2 py-0.5 text-xs font-semibold ${
            run.mode === "agent" ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"}`}>
            {run.mode === "agent" ? "web-grounded" : "baseline parameters"}
          </span>
          {run.degraded && <span className="rounded bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-800">degraded</span>}
          <span className="text-xs text-slate-400">{new Date(run.created_at).toLocaleString()}</span>
        </div>
        <p className="mt-2 text-slate-700">{run.summary}</p>
        {run.notes.map((n) => (
          <p key={n} className="mt-1 rounded bg-amber-50 px-2 py-1 text-xs text-amber-800">{n}</p>
        ))}
        {run.violations.map((v) => (
          <p key={v} className="mt-1 rounded bg-red-50 px-2 py-1 text-xs text-red-800">{v}</p>
        ))}
      </div>

      <section>
        <h4 className="mb-1 font-semibold text-slate-800">Web research · {run.research.status}</h4>
        {run.research.error && <p className="text-xs text-slate-500">{run.research.error}</p>}
        {run.research.queries.length > 0 && (
          <p className="text-xs text-slate-500">Searched: {run.research.queries.join(" · ")}</p>
        )}
        <ol className="mt-1 space-y-1 text-xs">
          {run.research.precedents.map((p) => (
            <li key={p.id}>
              <span className="font-mono text-slate-400">[{p.id}]</span> {p.claim}{" "}
              {p.parameter && <span className="rounded bg-slate-100 px-1 text-slate-600">{p.parameter}</span>}{" "}
              <a href={p.source_url} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                {p.source_title}
              </a>
            </li>
          ))}
        </ol>
      </section>

      {(run.accepted.length > 0 || run.rejected.length > 0) && (
        <section>
          <h4 className="mb-1 font-semibold text-slate-800">Parameter changes</h4>
          <ul className="space-y-1 text-xs">
            {run.accepted.map((a) => (
              <li key={a.parameter} className="text-emerald-800">
                ✓ {a.parameter}: {a.baseline} → {a.value} (precedents {a.precedent_ids.join(", ")}) — {a.rationale}
              </li>
            ))}
            {run.rejected.map((r, i) => (
              <li key={`${r.parameter}-${i}`} className="text-slate-500">
                ✗ {r.parameter} → {r.value}: rejected, {r.reason}
              </li>
            ))}
          </ul>
        </section>
      )}

      {run.unmet.length > 0 && (
        <section>
          <h4 className="mb-1 font-semibold text-red-800">Unmet need</h4>
          <ul className="space-y-0.5 text-xs text-slate-700">
            {run.unmet.map((u, i) => (
              <li key={i}>{u.name}: {u.shortfall} {u.sku.replace(/_/g, " ")} — {u.reason}</li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h4 className="mb-1 font-semibold text-slate-800">Planning parameters used</h4>
        <table className="w-full text-xs">
          <tbody>
            {params.map((p) => (
              <tr key={p.name} className="border-t border-slate-100 align-top">
                <td className="py-1 pr-2 font-mono text-slate-700">{p.name}</td>
                <td className="pr-2 font-mono tabular-nums">{p.value} <span className="text-slate-400">{p.unit}</span></td>
                <td className={p.adjusted_by_agent ? "text-emerald-700" : p.kind === "assumption" ? "text-amber-700" : "text-slate-500"}>
                  {p.adjusted_by_agent ? "agent-adjusted" : p.kind}
                  {p.citations[0] && (
                    <a href={p.citations[0].source_url} target="_blank" rel="noreferrer" className="ml-1 text-blue-700 hover:underline">
                      source
                    </a>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
