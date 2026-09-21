"use client";

import type { GraphGeo, InventoryItem } from "@/lib/types";
import { fmt } from "@/lib/style";

export default function InventoryPanel({ items, graph }: { items: InventoryItem[]; graph: GraphGeo | null }) {
  const nodeName = Object.fromEntries((graph?.nodes.features ?? []).map((f) => [f.properties.id, f.properties.name]));
  const byNode = new Map<string, InventoryItem[]>();
  for (const it of items) byNode.set(it.node_id, [...(byNode.get(it.node_id) ?? []), it]);

  if (!items.length) return <p className="text-sm text-slate-500">No stock recorded for this event.</p>;
  return (
    <div className="space-y-4">
      {[...byNode.entries()].map(([node, lines]) => (
        <div key={node}>
          <h4 className="mb-1 text-sm font-semibold text-slate-800">{nodeName[node] ?? node}</h4>
          <ul className="space-y-1.5">
            {lines.map((l) => {
              const reservedPct = l.quantity ? (100 * l.reserved) / l.quantity : 0;
              return (
                <li key={l.id} className="text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-700">{l.sku.replace(/_/g, " ")}</span>
                    <span className="font-mono tabular-nums text-slate-600">
                      {fmt(l.quantity - l.reserved)} free / {fmt(l.quantity)} {l.unit}
                    </span>
                  </div>
                  <div className="mt-0.5 h-1.5 rounded bg-emerald-100" title={`${l.reserved} reserved`}>
                    <div className="h-1.5 rounded bg-blue-500" style={{ width: `${reservedPct}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
      <p className="text-xs text-slate-500">Blue = reserved for approved dispatches. Stock can never be reserved twice.</p>
    </div>
  );
}
