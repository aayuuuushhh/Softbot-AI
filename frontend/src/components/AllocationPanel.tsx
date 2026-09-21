// frontend/src/components/AllocationPanel.tsx

import type { ResourceAssignment, ResourceUnit } from "@/lib/api";

/*
  A dispatch board, not a chart. Coordinators read this to answer one question —
  what goes where, and in what order — so the table is grouped by resource and
  the committed-versus-held figures are stated explicitly. A stacked bar would
  look better and answer neither question.
*/
const URGENCY_STYLES: Record<string, string> = {
  immediate: "bg-red-50 text-red-800 ring-red-200",
  urgent: "bg-orange-50 text-orange-800 ring-orange-200",
  scheduled: "bg-slate-100 text-slate-700 ring-slate-200",
};

function urgencyStyle(urgency: string) {
  return URGENCY_STYLES[urgency.toLowerCase()] ?? URGENCY_STYLES.scheduled;
}

/*
  "1 teams" on a dispatch board reads as a bug. Units are free text from the
  roster, so this only undoes a trailing plural "s" and leaves mass nouns
  ("people", "personnel") alone rather than guessing at English morphology.
*/
const MASS_NOUNS = new Set(["people", "personnel", "capacity", "aircraft"]);

function quantityLabel(quantity: number, unit: string) {
  const trimmed = unit.trim();
  if (!trimmed) return String(quantity);
  if (
    quantity === 1 &&
    trimmed.endsWith("s") &&
    !MASS_NOUNS.has(trimmed.toLowerCase())
  ) {
    return `${quantity} ${trimmed.slice(0, -1)}`;
  }
  return `${quantity} ${trimmed}`;
}

/*
  Thousands separators on the header totals: "1200 people" scans as a typo on a
  board that also carries single-digit counts.
*/
function formatCount(n: number) {
  return n.toLocaleString("en-US");
}

export default function AllocationPanel({
  allocation,
  reserveNotes,
  roster,
  loading,
}: {
  allocation: ResourceAssignment[];
  reserveNotes: string;
  roster: ResourceUnit[];
  loading: boolean;
}) {
  /*
    Group by resource so each roster line reads as one dispatch decision. Every
    roster line is kept, including the ones the model sent nowhere: a coordinator
    reading this board has to see what the operation actually holds, and a
    resource that silently vanished from the panel reads as "we don't have any"
    rather than "all four machines are still in reserve".
  */
  const groups = roster.map((resource) => {
    const rows = allocation.filter(
      (a) =>
        a.resource.trim().toLowerCase() === resource.name.trim().toLowerCase(),
    );
    const committed = rows.reduce((sum, row) => sum + row.quantity, 0);
    return { resource, rows, committed };
  });

  // Anything the model named that is not on the roster still has to be shown,
  // or the plan on screen would not be the plan the model produced.
  const known = new Set(roster.map((r) => r.name.trim().toLowerCase()));
  const extras = allocation.filter(
    (a) => !known.has(a.resource.trim().toLowerCase()),
  );

  return (
    <section className="overflow-hidden rounded-xl border border-diq-line bg-white shadow-sm shadow-slate-900/5">
      <header className="border-b border-diq-line bg-slate-50 px-5 py-3.5">
        <h3 className="text-base font-semibold text-diq-ink">
          Resource allocation
        </h3>
        <p className="mt-0.5 text-xs text-diq-muted">
          Where to send each unit, worst-hit zones first.
        </p>

        {roster.length > 0 && (
          <p className="tabular mt-2 text-xs text-slate-700">
            <span className="font-semibold text-diq-ink">On hand:</span>{" "}
            {roster
              .map((r) => `${r.name} ${formatCount(r.quantity)}`)
              .join(" · ")}
          </p>
        )}
      </header>

      {loading && (
        <p className="px-5 py-8 text-sm text-diq-muted">Planning allocation…</p>
      )}

      {!loading && allocation.length === 0 && (
        <p className="px-5 py-8 text-sm text-diq-muted">
          Run damage analysis to build an allocation plan.
        </p>
      )}

      {!loading && allocation.length > 0 && (
        <div className="divide-y divide-diq-line">
          {groups.map(({ resource, rows, committed }) => (
            <div key={resource.name} className="px-5 py-4">
              <div className="flex items-baseline justify-between gap-3">
                <p className="text-sm font-semibold text-diq-ink">
                  {resource.name}
                </p>
                <p className="tabular text-xs text-diq-muted">
                  {formatCount(committed)} of {formatCount(resource.quantity)}{" "}
                  {resource.unit} committed
                </p>
              </div>

              {rows.length === 0 && (
                <p className="mt-1.5 text-xs text-diq-muted">
                  Not dispatched — held in reserve.
                </p>
              )}

              {rows.length > 0 && (
                <table className="mt-2.5 w-full text-sm">
                  <tbody>
                    {rows.map((row, index) => (
                      <tr
                        key={`${row.zone_rank}-${index}`}
                        className="align-baseline"
                      >
                        <td className="w-16 py-1 pr-3 text-diq-muted">
                          Zone {row.zone_rank}
                        </td>

                        <td className="w-20 py-1 pr-3 font-semibold text-diq-ink">
                          {quantityLabel(row.quantity, row.unit)}
                        </td>

                        <td className="w-24 py-1 pr-3">
                          <span
                            className={`rounded px-1.5 py-0.5 text-xs capitalize ring-1 ring-inset ${urgencyStyle(row.urgency)}`}
                          >
                            {row.urgency}
                          </span>
                        </td>

                        <td className="py-1 text-xs leading-5 text-slate-700">
                          {row.justification}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}

          {extras.length > 0 && (
            <div className="px-5 py-4">
              <p className="text-sm font-semibold text-diq-ink">Other assets</p>
              <ul className="mt-2 space-y-1 text-sm text-slate-700">
                {extras.map((row, index) => (
                  <li key={`${row.resource}-${index}`}>
                    Zone {row.zone_rank} —{" "}
                    {quantityLabel(row.quantity, row.unit)} {row.resource}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {reserveNotes && (
            <div className="bg-slate-50 px-5 py-3.5">
              <p className="text-xs font-semibold text-diq-ink">
                Held in reserve
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-700">
                {reserveNotes}
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
