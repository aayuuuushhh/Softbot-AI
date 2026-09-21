"use client";

import { useState } from "react";
import type { Dispatch, DispatchStatus, GraphGeo } from "@/lib/types";
import { NEXT_STATUS, STATUS_BADGE } from "@/lib/style";

type Props = {
  dispatches: Dispatch[];
  graph: GraphGeo | null;
  zoneNames: Record<string, string>;
  onTransition: (id: string, to: DispatchStatus) => Promise<void>;
  busy: boolean;
};

export default function DispatchPanel({ dispatches, graph, zoneNames, onTransition, busy }: Props) {
  const [open, setOpen] = useState<string | null>(null);
  const nodeName = Object.fromEntries((graph?.nodes.features ?? []).map((f) => [f.properties.id, f.properties.name]));
  const live = dispatches.filter((d) => d.status !== "cancelled");

  if (!live.length) {
    return <p className="text-sm text-slate-500">No dispatch orders yet. Run allocation to plan them.</p>;
  }
  return (
    <ul className="space-y-2">
      {live.map((d) => (
        <li key={d.id} className="rounded-md border border-slate-200 p-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded bg-slate-900 px-1.5 py-0.5 text-xs font-bold text-white">P{d.priority}</span>
                <span className="font-medium">{zoneNames[d.zone_id] ?? d.zone_id}</span>
                <span className={`rounded px-1.5 py-0.5 text-xs ${STATUS_BADGE[d.status]}`}>{d.status.replace("_", " ")}</span>
                {d.transport_mode === "air" && (
                  <span className="rounded bg-violet-600 px-1.5 py-0.5 text-xs text-white">helicopter</span>
                )}
              </div>
              <div className="mt-1 text-sm text-slate-700">
                {d.items.map((i) => `${i.quantity} ${i.sku.replace(/_/g, " ")}`).join(" · ")}
              </div>
              <div className="mt-0.5 text-xs text-slate-500">
                from {nodeName[d.from_node_id] ?? "?"} · {d.route.map((id) => nodeName[id] ?? "?").join(" → ")}
                {d.route_distance_km != null && ` · ${d.route_distance_km} km`}
              </div>
            </div>
            <div className="flex shrink-0 gap-1">
              {NEXT_STATUS[d.status].map((n) => (
                <button
                  key={n.to}
                  type="button"
                  disabled={busy}
                  onClick={() => onTransition(d.id, n.to)}
                  className={`rounded px-2 py-1 text-xs font-medium disabled:opacity-50 ${
                    n.to === "cancelled" ? "border border-slate-300 text-slate-600 hover:bg-slate-100"
                      : "bg-blue-600 text-white hover:bg-blue-700"}`}
                >
                  {n.label}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            onClick={() => setOpen(open === d.id ? null : d.id)}
            className="mt-2 text-xs text-blue-700 hover:underline"
          >
            {open === d.id ? "Hide" : "Why?"} · {d.citations.length} source{d.citations.length === 1 ? "" : "s"}
          </button>
          {open === d.id && (
            <div className="mt-2 space-y-2 rounded bg-slate-50 p-2 text-xs">
              <p className="text-slate-700">{d.rationale}</p>
              <ol className="list-decimal space-y-1 pl-4">
                {d.citations.map((c) => (
                  <li key={c.source_url + c.claim}>
                    <span className="text-slate-700">{c.claim}</span>{" "}
                    <a href={c.source_url} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                      {c.source_title}
                    </a>
                    {c.published && <span className="text-slate-400"> ({c.published})</span>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}
