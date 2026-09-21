"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export default function UploadPanel({ eventId, onDone }: { eventId: string; onDone: (msg: string) => void }) {
  const [pre, setPre] = useState<File | null>(null);
  const [post, setPost] = useState<File | null>(null);
  const [photo, setPhoto] = useState<File | null>(null);
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [reporter, setReporter] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (fn: () => Promise<string>) => {
    setBusy(true);
    setError(null);
    try {
      onDone(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const useMyLocation = () =>
    navigator.geolocation?.getCurrentPosition((p) => {
      setLat(p.coords.latitude.toFixed(5));
      setLon(p.coords.longitude.toFixed(5));
    });

  const input = "block w-full rounded border border-slate-300 px-2 py-1 text-sm";
  return (
    <div className="space-y-6 text-sm">
      <section className="space-y-2">
        <h4 className="font-semibold text-slate-800">Satellite pair (S1)</h4>
        <label className="block text-xs text-slate-600">Pre-event GeoTIFF
          <input type="file" accept=".tif,.tiff" className={input} onChange={(e) => setPre(e.target.files?.[0] ?? null)} />
        </label>
        <label className="block text-xs text-slate-600">Post-event GeoTIFF
          <input type="file" accept=".tif,.tiff" className={input} onChange={(e) => setPost(e.target.files?.[0] ?? null)} />
        </label>
        <button
          type="button"
          disabled={!pre || !post || busy}
          onClick={() => run(async () => {
            const r = await api.analyze(eventId, pre!, post!);
            return `Analysed: ${r.detections} detections, cloud ${(100 * Number(r.cloud_fraction)).toFixed(0)}%, ` +
              `${r.roads_blocked_by_cv ?? 0} roads blocked by CV.`;
          })}
          className="rounded bg-slate-900 px-3 py-1.5 text-white disabled:opacity-40"
        >
          Run change detection
        </button>
      </section>

      <section className="space-y-2">
        <h4 className="font-semibold text-slate-800">Ground photo (S2)</h4>
        <p className="text-xs text-slate-500">
          A field photo outranks the satellite on severity - it works under monsoon cloud.
        </p>
        <input type="file" accept="image/*" capture="environment" className={input}
          onChange={(e) => setPhoto(e.target.files?.[0] ?? null)} />
        <div className="grid grid-cols-2 gap-2">
          <input placeholder="lat" value={lat} onChange={(e) => setLat(e.target.value)} className={input} />
          <input placeholder="lon" value={lon} onChange={(e) => setLon(e.target.value)} className={input} />
        </div>
        <input placeholder="reporter (optional)" value={reporter} onChange={(e) => setReporter(e.target.value)} className={input} />
        <div className="flex gap-2">
          <button type="button" onClick={useMyLocation} className="rounded border border-slate-300 px-3 py-1.5">
            Use my location
          </button>
          <button
            type="button"
            disabled={!photo || !lat || !lon || busy}
            onClick={() => run(async () => {
              const r = await api.uploadGround(eventId, photo!, Number(lat), Number(lon), reporter || undefined);
              return `Ground report: ${r.severity} (confidence ${Number(r.confidence).toFixed(2)})` +
                (r.zone_id ? "" : " - outside every zone");
            })}
            className="rounded bg-slate-900 px-3 py-1.5 text-white disabled:opacity-40"
          >
            Upload photo
          </button>
        </div>
      </section>
      {busy && <p className="text-xs text-slate-500">Working…</p>}
      {error && <p className="rounded bg-red-50 px-2 py-1 text-xs text-red-800">{error}</p>}
    </div>
  );
}
