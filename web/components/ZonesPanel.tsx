"use client";

import type { Derivation as D, Reach, ZoneNeeds } from "@/lib/types";
import { SEVERITY_BADGE, fmt } from "@/lib/style";
import Derivation from "./Derivation";

const ORDER = { destroyed: 0, major: 1, minor: 2, none: 3 } as const;

type Props = {
  needs: ZoneNeeds[];
  reach: Reach[];
  selected: string | null;
  onSelect: (zoneId: string) => void;
  decidedBy: Record<string, string>;
};

export default function ZonesPanel({ needs, reach, selected, onSelect, decidedBy }: Props) {
  const access = Object.fromEntries(reach.filter((r) => r.zone_id).map((r) => [r.zone_id, r]));
  const rows = [...needs].sort(
    (a, b) => ORDER[a.severity] - ORDER[b.severity] || (b.affected_people ?? 0) - (a.affected_people ?? 0),
  );
  const sku = (z: ZoneNeeds, k: string) =>
    (z.derivations?.sku_demand as Record<string, D> | undefined)?.[k];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-white text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="py-2 pr-2">Zone</th>
            <th className="pr-2">Access</th>
            <th className="pr-2 text-right">Affected</th>
            <th className="pr-2 text-right">Doctors</th>
            <th className="pr-2 text-right">Rescue</th>
            <th className="pr-2 text-right">Rice</th>
            <th className="pr-2 text-right">Water/day</th>
            <th className="text-right">Kits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((z) => {
            const r = access[z.zone_id];
            return (
              <tr
                key={z.zone_id}
                onClick={() => onSelect(z.zone_id)}
                className={`cursor-pointer border-t border-slate-100 ${selected === z.zone_id ? "bg-blue-50" : "hover:bg-slate-50"}`}
              >
                <td className="py-2 pr-2">
                  <div className="font-medium text-slate-900">{z.name}</div>
                  <div className="flex items-center gap-1 text-xs">
                    <span className={`rounded px-1.5 py-0.5 ${SEVERITY_BADGE[z.severity]}`}>{z.severity}</span>
                    {decidedBy[z.zone_id] && decidedBy[z.zone_id] !== "none" && (
                      <span className="text-slate-400">by {decidedBy[z.zone_id]}</span>
                    )}
                  </div>
                </td>
                <td className="pr-2 text-xs">
                  {r?.status === "unreachable" ? (
                    <span className="font-semibold text-red-700" title="No open road; air transport not verified">cut off</span>
                  ) : r?.status === "air" ? (
                    <span className="text-violet-700">air only</span>
                  ) : r ? (
                    <span className="text-emerald-700" title={r.best?.path_names.join(" → ")}>road {r.best?.distance_km} km</span>
                  ) : "–"}
                </td>
                <td className="pr-2 text-right" onClick={(e) => e.stopPropagation()}>
                  <Derivation d={z.derivations?.affected_people as D | undefined} />
                </td>
                <td className="pr-2 text-right" onClick={(e) => e.stopPropagation()}><Derivation d={sku(z, "doctor")} /></td>
                <td className="pr-2 text-right" onClick={(e) => e.stopPropagation()}><Derivation d={sku(z, "rescue_personnel")} /></td>
                <td className="pr-2 text-right" onClick={(e) => e.stopPropagation()}><Derivation d={sku(z, "rice_25kg")} /></td>
                <td className="pr-2 text-right" onClick={(e) => e.stopPropagation()}>
                  <Derivation d={z.derivations?.water_litres_per_day as D | undefined} suffix=" L" />
                </td>
                <td className="text-right" onClick={(e) => e.stopPropagation()}><Derivation d={sku(z, "trauma_kit")} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-slate-500">
        Click any number for its arithmetic. Total affected: {fmt(needs.reduce((s, z) => s + (z.affected_people ?? 0), 0))}.
        Amber lines are planning assumptions, not measurements.
      </p>
    </div>
  );
}
