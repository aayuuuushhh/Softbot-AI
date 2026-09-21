"use client";

// A number a coordinator cannot interrogate is a number they will not act on.
// Every need figure opens its arithmetic, assumptions and sources on click.

import { useState } from "react";
import type { Derivation as D } from "@/lib/types";
import { fmt } from "@/lib/style";

export default function Derivation({ d, suffix = "" }: { d?: D; suffix?: string }) {
  const [open, setOpen] = useState(false);
  if (!d) return <span className="text-slate-400">–</span>;
  return (
    <span className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="font-mono tabular-nums underline decoration-dotted decoration-slate-400 underline-offset-2 hover:text-blue-700"
        title="Show how this number was derived"
      >
        {fmt(d.value)}
        {suffix}
      </button>
      {open && (
        <span className="absolute right-0 z-30 mt-1 block w-80 rounded-md border border-slate-200 bg-white p-3 text-left text-xs shadow-lg">
          <span className="block font-mono text-slate-900">= {d.formula}</span>
          {d.assumptions.length > 0 && (
            <span className="mt-2 block text-slate-600">
              {d.assumptions.map((a) => (
                <span key={a} className={`block ${a.includes("assumption") ? "text-amber-700" : ""}`}>• {a}</span>
              ))}
            </span>
          )}
          {d.sources.length > 0 && (
            <span className="mt-2 block">
              {d.sources.map((s) => (
                <a key={s} href={s} target="_blank" rel="noreferrer" className="block truncate text-blue-700 hover:underline">
                  {s}
                </a>
              ))}
            </span>
          )}
          <button type="button" onClick={() => setOpen(false)} className="mt-2 text-slate-500 hover:text-slate-800">
            close
          </button>
        </span>
      )}
    </span>
  );
}
