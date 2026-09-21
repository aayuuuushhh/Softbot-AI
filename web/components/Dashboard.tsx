"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type {
  AgentRun, DamageGeo, Dispatch, DispatchStatus, EventDoc, GraphGeo, InventoryItem, Reach, ZoneNeeds,
} from "@/lib/types";
import AgentPanel from "./AgentPanel";
import DispatchPanel from "./DispatchPanel";
import InventoryPanel from "./InventoryPanel";
import UploadPanel from "./UploadPanel";
import ZonesPanel from "./ZonesPanel";

const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-sm text-slate-400">Loading map…</div>,
});

const TABS = ["Zones", "Dispatches", "Inventory", "Agent", "Upload"] as const;
type Tab = (typeof TABS)[number];
const STAGES = ["overhead", "ground", "fusion", "graph", "allocation"] as const;
const STAGE_DOT = { pending: "bg-slate-300", running: "bg-blue-500 animate-pulse", done: "bg-emerald-500", failed: "bg-red-500" };

export default function Dashboard() {
  const [events, setEvents] = useState<EventDoc[]>([]);
  const [eventId, setEventId] = useState<string | null>(null);
  const [event, setEvent] = useState<EventDoc | null>(null);
  const [damage, setDamage] = useState<DamageGeo | null>(null);
  const [graph, setGraph] = useState<GraphGeo | null>(null);
  const [reach, setReach] = useState<Reach[]>([]);
  const [needs, setNeeds] = useState<ZoneNeeds[]>([]);
  const [dispatches, setDispatches] = useState<Dispatch[]>([]);
  const [inventory, setInventory] = useState<InventoryItem[]>([]);
  const [run, setRun] = useState<AgentRun | null>(null);
  const [tab, setTab] = useState<Tab>("Zones");
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    api.events()
      .then((evs) => {
        setEvents(evs);
        setEventId((cur) => cur ?? evs[0]?.id ?? null);
      })
      .catch(() => setOffline(true));
  }, []);

  const refresh = useCallback(async () => {
    if (!eventId) return;
    try {
      const [ev, dmg, g, r, n, d, inv, runs] = await Promise.all([
        api.event(eventId), api.damage(eventId), api.graph(eventId), api.reachability(eventId),
        api.needs(eventId), api.dispatches(eventId), api.inventory(eventId), api.agentRuns(eventId),
      ]);
      setEvent(ev); setDamage(dmg); setGraph(g); setReach(r); setNeeds(n); setDispatches(d); setInventory(inv);
      setRun(runs[0] ? await api.agentRun(runs[0]._id) : null);
      setOffline(false);
    } catch (e) {
      setOffline(true);
      setFlash({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    }
  }, [eventId]);

  useEffect(() => {
    // Poll: field teams move dispatches along; keep the view current.
    const first = setTimeout(refresh, 0);
    const t = setInterval(refresh, 15000);
    return () => { clearTimeout(first); clearInterval(t); };
  }, [refresh]);

  const act = async (fn: () => Promise<string | void>) => {
    setBusy(true);
    setFlash(null);
    try {
      const msg = await fn();
      if (msg) setFlash({ kind: "ok", text: msg });
      await refresh();
    } catch (e) {
      setFlash({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  const zoneNames = useMemo(() => Object.fromEntries(needs.map((z) => [z.zone_id, z.name])), [needs]);
  const decidedBy = useMemo(
    () => Object.fromEntries((damage?.zones.features ?? []).map((f) => [f.properties.id, f.properties.decided_by])),
    [damage],
  );
  const cutOff = reach.filter((r) => r.status === "unreachable" && needs.find((n) => n.zone_id === r.zone_id)?.severity !== "none");

  if (offline && !events.length) {
    return (
      <main className="grid h-screen place-items-center p-8 text-center">
        <div>
          <h1 className="text-xl font-semibold">Uddhar API unreachable</h1>
          <p className="mt-2 text-sm text-slate-600">Start it with <code>uvicorn api.main:app --port 8765</code>, then reload.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex h-screen flex-col bg-slate-50 text-slate-900">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-200 bg-white px-4 py-2">
        <div className="flex items-baseline gap-2">
          <span className="text-lg font-bold tracking-tight">Uddhar</span>
          <span className="text-xs text-slate-500">damage → need → dispatch</span>
        </div>
        <select
          value={eventId ?? ""}
          onChange={(e) => { setEventId(e.target.value); setSelectedZone(null); }}
          className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
        >
          {events.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
        </select>
        {event && (
          <div className="flex items-center gap-3 text-xs text-slate-600">
            {STAGES.map((s) => (
              <span key={s} className="flex items-center gap-1" title={event.stages[s]}>
                <span className={`h-2 w-2 rounded-full ${STAGE_DOT[event.stages[s]]}`} />{s}
              </span>
            ))}
            <span>cloud {(100 * event.cloud_fraction).toFixed(0)}%</span>
          </div>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {event && (
            <label className="flex items-center gap-1 text-xs">
              <input
                type="checkbox"
                checked={event.air_transport_verified}
                disabled={busy}
                onChange={(e) => act(async () => {
                  await api.setAir(event.id, e.target.checked);
                  return e.target.checked ? "Air transport verified: cut-off zones can now be served by helicopter."
                    : "Air transport verification revoked.";
                })}
              />
              air transport verified
            </label>
          )}
          <button
            type="button"
            disabled={!eventId || busy}
            onClick={() => act(async () => {
              const p = await api.allocate(eventId!);
              setTab("Dispatches");
              return p.summary;
            })}
            className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {busy ? "Working…" : "Run allocation"}
          </button>
          <button
            type="button"
            disabled={!eventId || busy || !dispatches.some((d) => d.status === "proposed")}
            onClick={() => act(async () => {
              const r = await api.approveAll(eventId!);
              return `Reserved ${r.reserved.length} orders` + (r.refused.length ? `; ${r.refused.length} refused (stock changed)` : "");
            })}
            className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50 disabled:opacity-50"
          >
            Approve all
          </button>
          {eventId && (
            <>
              <a href={api.reportUrl(eventId)} target="_blank" rel="noreferrer" className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50">PDF</a>
              <a href={api.summaryUrl(eventId)} target="_blank" rel="noreferrer" className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50">Radio text</a>
            </>
          )}
        </div>
      </header>

      {(flash || cutOff.length > 0) && (
        <div className="space-y-1 px-4 pt-2">
          {cutOff.length > 0 && !event?.air_transport_verified && (
            <div className="rounded border border-red-200 bg-red-50 px-3 py-1.5 text-sm text-red-800">
              <b>{cutOff.map((r) => r.name).join(", ")}</b> {cutOff.length === 1 ? "has" : "have"} no open road.
              Nothing can reach {cutOff.length === 1 ? "it" : "them"} until air transport is verified.
            </div>
          )}
          {flash && (
            <div className={`rounded px-3 py-1.5 text-sm ${flash.kind === "ok" ? "bg-emerald-50 text-emerald-900" : "bg-red-50 text-red-800"}`}>
              {flash.text}
              <button type="button" onClick={() => setFlash(null)} className="ml-2 text-xs opacity-60 hover:opacity-100">dismiss</button>
            </div>
          )}
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-2 lg:flex-row">
        <section className="relative min-h-[320px] flex-1 overflow-hidden rounded-lg border border-slate-200 bg-white">
          <MapView damage={damage} graph={graph} dispatches={dispatches} selectedZone={selectedZone} onSelectZone={setSelectedZone} />
          <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-white/90 px-2 py-1 text-[11px] leading-4 text-slate-700 shadow">
            <div><span className="inline-block h-2 w-3 bg-red-600" /> destroyed <span className="ml-1 inline-block h-2 w-3 bg-orange-500" /> major <span className="ml-1 inline-block h-2 w-3 bg-yellow-500" /> minor</div>
            <div><span className="inline-block h-0.5 w-4 bg-green-600 align-middle" /> open <span className="ml-1 inline-block h-0.5 w-4 bg-amber-600 align-middle" /> degraded <span className="ml-1 inline-block h-0.5 w-4 border-t-2 border-dashed border-red-600 align-middle" /> blocked</div>
            <div><span className="inline-block h-1 w-4 bg-blue-600/50 align-middle" /> road dispatch <span className="ml-1 inline-block w-4 border-t-2 border-dashed border-violet-600 align-middle" /> helicopter</div>
          </div>
        </section>

        <aside className="flex min-h-0 w-full flex-col rounded-lg border border-slate-200 bg-white lg:w-[560px]">
          <nav className="flex border-b border-slate-200 text-sm">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`flex-1 px-2 py-2 ${tab === t ? "border-b-2 border-blue-600 font-semibold text-blue-700" : "text-slate-600 hover:text-slate-900"}`}
              >
                {t}
                {t === "Dispatches" && dispatches.filter((d) => d.status !== "cancelled").length > 0 &&
                  <span className="ml-1 rounded-full bg-slate-200 px-1.5 text-xs">{dispatches.filter((d) => d.status !== "cancelled").length}</span>}
              </button>
            ))}
          </nav>
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {tab === "Zones" && <ZonesPanel needs={needs} reach={reach} selected={selectedZone} onSelect={setSelectedZone} decidedBy={decidedBy} />}
            {tab === "Dispatches" && (
              <DispatchPanel
                dispatches={dispatches} graph={graph} zoneNames={zoneNames} busy={busy}
                onTransition={(id: string, to: DispatchStatus) => act(async () => { await api.setDispatch(id, to); })}
              />
            )}
            {tab === "Inventory" && <InventoryPanel items={inventory} graph={graph} />}
            {tab === "Agent" && <AgentPanel run={run} />}
            {tab === "Upload" && eventId && <UploadPanel eventId={eventId} onDone={(m) => act(async () => m)} />}
          </div>
          <footer className="border-t border-slate-200 px-3 py-1.5 text-[11px] text-slate-500">
            Imagery-derived first-pass estimate. Uddhar ranks and proposes; people approve every dispatch.
          </footer>
        </aside>
      </div>
    </main>
  );
}
