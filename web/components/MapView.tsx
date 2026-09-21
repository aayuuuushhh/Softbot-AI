"use client";

// MapLibre view: ward damage, detections, the road graph (blocked roads dashed
// red) and planned dispatch routes (helicopter legs dashed purple).
// Loaded with next/dynamic { ssr: false } - MapLibre needs `window`.

import { useEffect, useRef, useState } from "react";
import { LngLatBounds, Map as MapLibreMap, Marker, NavigationControl, Popup, setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import type { DamageGeo, Dispatch, GraphGeo } from "@/lib/types";
import { SEVERITY_COLOR, ROAD_COLOR } from "@/lib/style";

type Props = {
  damage: DamageGeo | null;
  graph: GraphGeo | null;
  dispatches: Dispatch[];
  selectedZone: string | null;
  onSelectZone: (zoneId: string) => void;
};

// See scripts/copy-maplibre-worker.mjs - the worker is served from /public.
setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

type SettableSource = { setData(data: GeoJSON.GeoJSON): void };

function routeFeatures(dispatches: Dispatch[], graph: GraphGeo | null): GeoJSON.FeatureCollection {
  if (!graph) return EMPTY;
  const where = new Map(graph.nodes.features.map((f) => [f.properties.id, (f.geometry as GeoJSON.Point).coordinates]));
  return {
    type: "FeatureCollection",
    features: dispatches
      .filter((d) => d.status !== "cancelled" && d.route.length > 1)
      .map((d) => ({
        type: "Feature" as const,
        geometry: { type: "LineString" as const, coordinates: d.route.map((id) => where.get(id)).filter(Boolean) as number[][] },
        properties: { mode: d.transport_mode, status: d.status, priority: d.priority },
      })),
  };
}

export default function MapView({ damage, graph, dispatches, selectedZone, onSelectZone }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const [loaded, setLoaded] = useState(false);
  const markers = useRef<Marker[]>([]);
  const fitted = useRef(false);
  const selectRef = useRef(onSelectZone);
  useEffect(() => {
    selectRef.current = onSelectZone;
  }, [onSelectZone]);

  // Create the map once.
  useEffect(() => {
    if (!container.current || map.current) return;
    const m = new MapLibreMap({
      container: container.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
            maxzoom: 19,
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.6, "raster-opacity": 0.85 } }],
      },
      center: [85.3, 28.08],
      zoom: 10,
    });
    m.addControl(new NavigationControl({ showCompass: false }), "top-right");

    m.on("load", () => {
      for (const id of ["zones", "detections", "roads", "routes", "nodes"]) {
        m.addSource(id, { type: "geojson", data: EMPTY });
      }
      m.addLayer({
        id: "zones-fill", type: "fill", source: "zones",
        paint: {
          "fill-color": ["match", ["get", "severity"], "destroyed", SEVERITY_COLOR.destroyed, "major", SEVERITY_COLOR.major,
            "minor", SEVERITY_COLOR.minor, SEVERITY_COLOR.none],
          "fill-opacity": 0.28,
        },
      });
      m.addLayer({
        id: "zones-line", type: "line", source: "zones",
        paint: { "line-color": ["case", ["boolean", ["get", "selected"], false], "#0f172a", "#475569"],
          "line-width": ["case", ["boolean", ["get", "selected"], false], 3, 1] },
      });
      m.addLayer({
        id: "detections", type: "fill", source: "detections",
        paint: { "fill-color": ["coalesce", ["get", "color"], "#dc2626"], "fill-opacity": 0.7 },
      });
      m.addLayer({
        id: "roads", type: "line", source: "roads",
        paint: {
          "line-color": ["match", ["get", "status"], "blocked", ROAD_COLOR.blocked, "degraded", ROAD_COLOR.degraded, ROAD_COLOR.open],
          "line-width": ["match", ["get", "status"], "blocked", 3, 2.5],
          "line-dasharray": ["match", ["get", "status"], "blocked", ["literal", [1, 1.5]], ["literal", [1, 0]]],
        },
      });
      m.addLayer({
        id: "routes-road", type: "line", source: "routes", filter: ["==", ["get", "mode"], "road"],
        paint: { "line-color": "#2563eb", "line-width": 5, "line-opacity": 0.45 },
      });
      m.addLayer({
        id: "routes-air", type: "line", source: "routes", filter: ["==", ["get", "mode"], "air"],
        paint: { "line-color": "#7c3aed", "line-width": 3, "line-dasharray": [2, 2] },
      });
      m.addLayer({
        id: "nodes", type: "circle", source: "nodes",
        paint: {
          "circle-radius": ["match", ["get", "kind"], "zone", 4, 8],
          "circle-color": ["match", ["get", "kind"], "holding_center", "#0f766e", "staging_node", "#0891b2", "#1e293b"],
          "circle-stroke-color": "#ffffff", "circle-stroke-width": 2,
        },
      });

      const popup = new Popup({ closeButton: false, closeOnClick: false, maxWidth: "260px" });
      const hover = (layer: string, html: (p: Record<string, unknown>) => string) => {
        m.on("mousemove", layer, (e) => {
          const p = e.features?.[0]?.properties;
          if (!p) return;
          m.getCanvas().style.cursor = "pointer";
          popup.setLngLat(e.lngLat).setHTML(html(p)).addTo(m);
        });
        m.on("mouseleave", layer, () => {
          m.getCanvas().style.cursor = "";
          popup.remove();
        });
      };
      hover("roads", (p) => `<b>${p.from} – ${p.to}</b><br/>${p.status} · ${p.distance_km} km` +
        (p.blocked_reason ? `<br/><i>${p.blocked_reason}</i>` : "") + (p.source ? `<br/>source: ${p.source}` : ""));
      hover("nodes", (p) => `<b>${p.name}</b><br/>${String(p.kind).replace("_", " ")}`);
      m.on("click", "zones-fill", (e) => {
        const id = e.features?.[0]?.properties?.id;
        if (id) selectRef.current(String(id));
      });
      setLoaded(true);
    });
    map.current = m;
    return () => {
      m.remove();
      map.current = null;
      setLoaded(false);
    };
  }, []);

  // Push data whenever it changes.
  useEffect(() => {
    const m = map.current;
    if (!m || !loaded) return;
    const src = (id: string) => m.getSource(id) as unknown as SettableSource | undefined;
    if (damage) {
      src("zones")?.setData({
        ...damage.zones,
        features: damage.zones.features.map((f) => ({ ...f, properties: { ...f.properties, selected: f.properties.id === selectedZone } })),
      } as GeoJSON.FeatureCollection);
      src("detections")?.setData(damage.detections as GeoJSON.FeatureCollection);
    }
    if (graph) {
      src("roads")?.setData(graph.edges as GeoJSON.FeatureCollection);
      src("nodes")?.setData(graph.nodes as GeoJSON.FeatureCollection);
    }
    src("routes")?.setData(routeFeatures(dispatches, graph));

    // Zone name labels as HTML markers (the raster basemap carries no glyphs).
    markers.current.forEach((mk) => mk.remove());
    markers.current = (damage?.zones.features ?? []).map((f) => {
      const el = document.createElement("div");
      el.className = "zone-label";
      el.textContent = f.properties.name;
      const ring = (f.geometry as GeoJSON.Polygon).coordinates[0];
      const lon = ring.reduce((s, c) => s + c[0], 0) / ring.length;
      const lat = Math.max(...ring.map((c) => c[1]));
      return new Marker({ element: el, anchor: "bottom" }).setLngLat([lon, lat]).addTo(m);
    });

    if (!fitted.current && damage?.zones.features.length) {
      const b = new LngLatBounds();
      damage.zones.features.forEach((f) => (f.geometry as GeoJSON.Polygon).coordinates[0].forEach((c) => b.extend(c as [number, number])));
      graph?.nodes.features.forEach((f) => b.extend((f.geometry as GeoJSON.Point).coordinates as [number, number]));
      m.fitBounds(b, { padding: 40, duration: 0 });
      fitted.current = true;
    }
  }, [loaded, damage, graph, dispatches, selectedZone]);

  return <div ref={container} className="h-full w-full" />;
}
