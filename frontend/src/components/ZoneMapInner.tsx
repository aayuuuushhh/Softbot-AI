// frontend/src/components/ZoneMapInner.tsx
// Actual Leaflet map. Only ever loaded client-side via next/dynamic in
// ZoneMap.tsx (ssr: false) — Leaflet touches `window` at import time.

"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CircleMarker,
  ImageOverlay,
  LayersControl,
  MapContainer,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { FOCUS_ZONE_COUNT, type AnalysisResult, type Zone } from "@/lib/api";
import { formatLatLng, googleEarthUrl, googleMapsUrl } from "@/lib/geo";

interface Props {
  analysis: AnalysisResult;
  postImageUrl?: string;
}

type GeoZone = Zone & { centroid_lat: number; centroid_lng: number };

function rankColor(rank: number) {
  if (rank === 1) return "#ef4444";
  if (rank === 2) return "#f97316";
  if (rank === 3) return "#eab308";
  return "#22c55e";
}

/*
  Satellite imagery is the default basemap: this is a damage-assessment tool, and
  a road map shows none of the rooftops, terrain or flooding a coordinator is
  actually looking for. Esri's World Imagery is free and needs no API key. Street
  tiles stay available as an alternate layer for route planning.
*/
const ESRI_IMAGERY_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const ESRI_IMAGERY_ATTRIBUTION =
  'Imagery &copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics';

// World Imagery ships no place names; this transparent overlay adds roads and
// labels on top so the scene stays navigable.
const ESRI_LABELS_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}";

function FitImageBounds({ bounds }: { bounds: L.LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { padding: [12, 12] });
  }, [map, bounds]);
  return null;
}

function zoneMarkers(geoZones: GeoZone[], imageMode: boolean) {
  return geoZones.map((zone) => {
    // Image mode stores pixel (x,y) as (lng, lat) = (x, y).
    const center: [number, number] = [zone.centroid_lat, zone.centroid_lng];

    return (
      <CircleMarker
        key={zone.rank}
        center={center}
        radius={zone.rank <= 3 ? 12 : 8}
        pathOptions={{
          color: rankColor(zone.rank),
          fillColor: rankColor(zone.rank),
          fillOpacity: 0.55,
          weight: 2,
        }}
      >
        <Popup>
          <div className="text-xs">
            <p className="font-bold">Zone {zone.rank}</p>
            <p>
              Destroyed: {zone.building_counts.destroyed} · Major:{" "}
              {zone.building_counts.major} buildings
            </p>
            <p>Priority score: {zone.priority_score}</p>

            {imageMode ? (
              // No georeference, so these are image pixels, not a place on Earth.
              // Linking them to Google Maps would point at latitude 512.
              <p>
                Pixel: ({Math.round(zone.centroid_lng)},{" "}
                {Math.round(zone.centroid_lat)})
              </p>
            ) : (
              <>
                <p className="mt-1 font-mono">
                  {formatLatLng(zone.centroid_lat, zone.centroid_lng)}
                </p>

                <p className="mt-1">
                  <a
                    href={googleMapsUrl(zone.centroid_lat, zone.centroid_lng)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google Maps
                  </a>
                  {" · "}
                  <a
                    href={googleEarthUrl(zone.centroid_lat, zone.centroid_lng)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google Earth
                  </a>
                </p>
              </>
            )}
          </div>
        </Popup>
      </CircleMarker>
    );
  });
}

export default function ZoneMapInner({ analysis, postImageUrl }: Props) {
  const imageMode = analysis.geo_mode === "image" || !analysis.geo_available;
  const [naturalSize, setNaturalSize] = useState<[number, number] | null>(
    analysis.image_size
      ? [analysis.image_size[0], analysis.image_size[1]]
      : null,
  );

  useEffect(() => {
    if (!imageMode || !postImageUrl || analysis.image_size) return;
    const img = new Image();
    img.onload = () => setNaturalSize([img.naturalWidth, img.naturalHeight]);
    img.src = postImageUrl;
  }, [imageMode, postImageUrl, analysis.image_size]);

  const geoZones = useMemo(() => {
    return analysis.zones
      .map((zone) => {
        if (
          typeof zone.centroid_lat === "number" &&
          typeof zone.centroid_lng === "number"
        ) {
          return zone as GeoZone;
        }
        const [x = 0, y = 0, w = 0, h = 0] = zone.bbox ?? [];
        if (!w && !h) return null;
        return {
          ...zone,
          centroid_lat: y + h / 2,
          centroid_lng: x + w / 2,
        } as GeoZone;
      })
      .filter((z): z is GeoZone => z != null)
      /*
        The same zones the overlay draws and the brief plans for. Plotting every
        ranked zone put a dozen markers on a map whose risk panel discussed five
        of them, which reads as two different analyses of the same scene.
      */
      .sort((a, b) => a.rank - b.rank)
      .slice(0, FOCUS_ZONE_COUNT);
  }, [analysis.zones]);

  const maskUrl = useMemo(
    () =>
      analysis.mask_base64
        ? `data:image/png;base64,${analysis.mask_base64}`
        : null,
    [analysis.mask_base64],
  );

  const imageBounds = useMemo(() => {
    if (!imageMode) return null;
    const w = naturalSize?.[0] ?? analysis.image_size?.[0] ?? 1024;
    const h = naturalSize?.[1] ?? analysis.image_size?.[1] ?? 1024;
    // CRS.Simple: [y, x] with y increasing downward to match image pixels.
    return L.latLngBounds([
      [0, 0],
      [h, w],
    ]);
  }, [imageMode, naturalSize, analysis.image_size]);

  if (geoZones.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-diq-muted">
        No ranked zones to plot.
      </div>
    );
  }

  if (imageMode && imageBounds) {
    const h = naturalSize?.[1] ?? analysis.image_size?.[1] ?? 1024;
    const w = naturalSize?.[0] ?? analysis.image_size?.[0] ?? 1024;
    return (
      <MapContainer
        crs={L.CRS.Simple}
        center={[h / 2, w / 2]}
        zoom={-1}
        minZoom={-3}
        maxZoom={4}
        scrollWheelZoom
        className="h-full w-full bg-white"
      >
        <FitImageBounds bounds={imageBounds} />
        {postImageUrl && (
          <ImageOverlay url={postImageUrl} bounds={imageBounds} opacity={0.95} />
        )}

        {/*
          The same damage mask the After panel paints, laid over the same post
          image. Zone markers on bare imagery ask the coordinator to take the
          ranking on faith; with the mask underneath, a circle sits on visible
          damage and the map argues for itself. The mask is produced at the post
          image's pixel size, so it shares these bounds exactly. Kept as a
          toggleable layer — reading the raw scene under a marker is a real
          need, and there is no other way back to it once it is covered.
        */}
        {maskUrl && (
          <LayersControl position="topright">
            <LayersControl.Overlay checked name="Damage overlay">
              <ImageOverlay url={maskUrl} bounds={imageBounds} opacity={0.85} />
            </LayersControl.Overlay>
          </LayersControl>
        )}

        {zoneMarkers(geoZones, true)}
      </MapContainer>
    );
  }

  const center: [number, number] = [
    geoZones.reduce((sum, z) => sum + z.centroid_lat, 0) / geoZones.length,
    geoZones.reduce((sum, z) => sum + z.centroid_lng, 0) / geoZones.length,
  ];

  return (
    <MapContainer
      center={center}
      zoom={16}
      scrollWheelZoom={false}
      className="h-full w-full"
    >
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Satellite">
          <TileLayer
            attribution={ESRI_IMAGERY_ATTRIBUTION}
            url={ESRI_IMAGERY_URL}
            maxZoom={19}
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Street">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>

        <LayersControl.Overlay checked name="Place labels">
          <TileLayer url={ESRI_LABELS_URL} maxZoom={19} />
        </LayersControl.Overlay>
      </LayersControl>

      {zoneMarkers(geoZones, false)}
    </MapContainer>
  );
}
