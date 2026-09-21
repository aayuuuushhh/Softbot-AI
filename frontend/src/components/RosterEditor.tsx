// frontend/src/components/RosterEditor.tsx

import type { ResourceUnit } from "@/lib/api";

/*
  The allocation plan is only useful if it divides the units this operation
  actually has. Editing happens before analysis and is sent with the brief
  request, so changing a number here changes the plan the model produces.
*/
export default function RosterEditor({
  roster,
  onChange,
  disabled,
}: {
  roster: ResourceUnit[];
  onChange: (roster: ResourceUnit[]) => void;
  disabled: boolean;
}) {
  function setQuantity(index: number, raw: string) {
    const parsed = Number.parseInt(raw, 10);
    const quantity = Number.isNaN(parsed) ? 0 : Math.max(0, Math.min(parsed, 10_000));
    onChange(roster.map((r, i) => (i === index ? { ...r, quantity } : r)));
  }

  return (
    <div className="border-t border-diq-line px-4 py-4">
      <h3 className="font-label text-xs uppercase tracking-[0.12em] text-slate-700">
        <span className="mr-2 inline-flex h-5 w-5 items-center justify-center rounded bg-slate-100 text-[11px] text-diq-muted">
          3
        </span>
        Available resources
      </h3>

      <p className="mt-2 text-xs leading-5 text-diq-muted">
        What you have to deploy. The allocation plan divides exactly this.
      </p>

      <ul className="mt-3 space-y-2">
        {roster.map((resource, index) => (
          <li key={resource.name} className="flex items-center gap-2">
            <label
              htmlFor={`roster-${index}`}
              className="flex-1 text-xs leading-4 text-slate-700"
            >
              {resource.name}
              <span className="block text-[11px] text-diq-muted">{resource.unit}</span>
            </label>

            <input
              id={`roster-${index}`}
              type="number"
              min={0}
              max={10000}
              inputMode="numeric"
              value={resource.quantity}
              disabled={disabled}
              onChange={(e) => setQuantity(index, e.target.value)}
              className="tabular w-20 rounded border border-diq-line bg-white px-2 py-1.5 text-right text-sm text-diq-ink disabled:cursor-not-allowed disabled:opacity-50"
            />
          </li>
        ))}
      </ul>
    </div>
  );
}
