// frontend/src/components/DamageCanvas.tsx

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { FOCUS_ZONE_COUNT, type AnalysisResult } from "@/lib/api";

interface Props {
  postImageUrl: string;
  analysis: AnalysisResult | null;
}

type CanvasStatus = "idle" | "loading" | "ready" | "error";

const ZONE_COLORS = {
  top: {
    stroke: "rgba(239, 68, 68, 0.98)",
    fill: "rgba(239, 68, 68, 0.16)",
    label: "rgba(239, 68, 68, 0.96)",
    glow: "rgba(239, 68, 68, 0.35)",
  },
  high: {
    stroke: "rgba(249, 115, 22, 0.95)",
    fill: "rgba(249, 115, 22, 0.12)",
    label: "rgba(249, 115, 22, 0.96)",
    glow: "rgba(249, 115, 22, 0.25)",
  },
  medium: {
    stroke: "rgba(234, 179, 8, 0.92)",
    fill: "rgba(234, 179, 8, 0.1)",
    label: "rgba(234, 179, 8, 0.94)",
    glow: "rgba(234, 179, 8, 0.18)",
  },
  neutral: {
    stroke: "rgba(226, 232, 240, 0.78)",
    fill: "rgba(226, 232, 240, 0.07)",
    label: "rgba(15, 23, 42, 0.94)",
    glow: "rgba(226, 232, 240, 0.12)",
  },
};

function getZoneStyle(rank: number) {
  if (rank === 1) return ZONE_COLORS.top;
  if (rank === 2) return ZONE_COLORS.high;
  if (rank === 3) return ZONE_COLORS.medium;
  return ZONE_COLORS.neutral;
}

function getPriorityLabel(rank: number) {
  if (rank === 1) return "Priority 1";
  if (rank === 2) return "Priority 2";
  if (rank === 3) return "Priority 3";
  return "Monitor";
}

/*
  The backend contract is [x, y, width, height] (see backend/app/schemas.py).
  This used to guess between that and [x_min, y_min, x_max, y_max] with a
  magic size threshold — which silently mangles any zone whose width exceeds
  its x offset. We own both sides of this API; don't guess.
*/
function normalizeBbox(bbox: number[]) {
  const [x = 0, y = 0, width = 0, height = 0] = bbox;

  return {
    x,
    y,
    width: Math.max(width, 1),
    height: Math.max(height, 1),
  };
}

type Obstacle = { x: number; y: number; w: number; h: number };

/*
  The After panel paints zone labels onto this canvas, but the HTML overlays
  above it (the "AI Damage Overlay After" header pill and the "Damage Mask
  Active" badge) sit at the top-left and hide any label drawn underneath them.
  Given a label's rect, nudge it straight down until it clears every obstacle so
  the zone name stays readable.
*/
function clearObstacles(
  labelX: number,
  labelY: number,
  labelWidth: number,
  labelHeight: number,
  obstacles: Obstacle[],
) {
  let y = labelY;

  for (const o of obstacles) {
    const overlapsX = labelX < o.x + o.w && labelX + labelWidth > o.x;
    const overlapsY = y < o.y + o.h && y + labelHeight > o.y;

    if (overlapsX && overlapsY) {
      y = o.y + o.h + 6;
    }
  }

  return y;
}

function getContainedImageRect(params: {
  containerWidth: number;
  containerHeight: number;
  imageWidth: number;
  imageHeight: number;
}) {
  const { containerWidth, containerHeight, imageWidth, imageHeight } = params;

  if (
    containerWidth <= 0 ||
    containerHeight <= 0 ||
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {
    return {
      x: 0,
      y: 0,
      width: containerWidth,
      height: containerHeight,
      scaleX: 1,
      scaleY: 1,
    };
  }

  const containerRatio = containerWidth / containerHeight;
  const imageRatio = imageWidth / imageHeight;

  let renderedWidth = containerWidth;
  let renderedHeight = containerHeight;

  if (imageRatio > containerRatio) {
    renderedWidth = containerWidth;
    renderedHeight = containerWidth / imageRatio;
  } else {
    renderedHeight = containerHeight;
    renderedWidth = containerHeight * imageRatio;
  }

  const x = (containerWidth - renderedWidth) / 2;
  const y = (containerHeight - renderedHeight) / 2;

  return {
    x,
    y,
    width: renderedWidth,
    height: renderedHeight,
    scaleX: renderedWidth / imageWidth,
    scaleY: renderedHeight / imageHeight,
  };
}

export default function DamageCanvas({ postImageUrl, analysis }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [status, setStatus] = useState<CanvasStatus>("idle");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const topZones = useMemo(() => {
    return [...(analysis?.zones ?? [])]
      .sort((a, b) => a.rank - b.rank)
      .slice(0, FOCUS_ZONE_COUNT);
  }, [analysis]);

  useEffect(() => {
    if (!postImageUrl) {
      setStatus("idle");
      setImageSize({ width: 0, height: 0 });
      return;
    }

    setStatus("loading");
    setImageSize({ width: 0, height: 0 });
  }, [postImageUrl]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateContainerSize = () => {
      const rect = container.getBoundingClientRect();

      setContainerSize({
        width: Math.max(Math.round(rect.width), 1),
        height: Math.max(Math.round(rect.height), 1),
      });
    };

    updateContainerSize();

    const resizeObserver = new ResizeObserver(updateContainerSize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    canvas.width = containerSize.width;
    canvas.height = containerSize.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (
      !analysis ||
      !imageSize.width ||
      !imageSize.height ||
      !containerSize.width ||
      !containerSize.height
    ) {
      return;
    }

    const imageRect = getContainedImageRect({
      containerWidth: containerSize.width,
      containerHeight: containerSize.height,
      imageWidth: imageSize.width,
      imageHeight: imageSize.height,
    });

    // Overlays that float above this canvas (see the After panel in page.tsx).
    // Coordinates are in container space, which matches the canvas 1:1.
    const obstacles: Obstacle[] = [
      // "AI Damage Overlay After" header pill (top-4 left-4, max-w ~52%).
      { x: 16, y: 16, w: containerSize.width * 0.56, h: 40 },
      // "Damage Mask Active" badge (top-16 left-4), only shown once analysed.
      { x: 16, y: 62, w: 176, h: 32 },
    ];

    const drawZones = () => {
      if (!analysis.zones?.length) return;

      ctx.save();

      topZones.forEach((zone) => {
        const { x, y, width, height } = normalizeBbox(zone.bbox);
        const style = getZoneStyle(zone.rank);
        const priorityLabel = getPriorityLabel(zone.rank);
        const damagedBuildings =
          zone.building_counts.destroyed + zone.building_counts.major;

        const drawX = imageRect.x + x * imageRect.scaleX;
        const drawY = imageRect.y + y * imageRect.scaleY;
        const drawWidth = width * imageRect.scaleX;
        const drawHeight = height * imageRect.scaleY;

        ctx.save();

        ctx.shadowColor = style.glow;
        ctx.shadowBlur = zone.rank === 1 ? 18 : 10;

        ctx.lineWidth = zone.rank === 1 ? 4 : 3;
        ctx.strokeStyle = style.stroke;
        ctx.fillStyle = style.fill;

        ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);

        ctx.restore();

        const label = `ZONE ${zone.rank}`;
        const subLabel = `${damagedBuildings} damaged`;

        const labelWidth = zone.rank === 1 ? 124 : 108;
        const labelHeight = 42;
        const labelX = drawX + 8;
        const labelY = clearObstacles(
          labelX,
          drawY + 8,
          labelWidth,
          labelHeight,
          obstacles,
        );

        ctx.fillStyle = style.label;
        ctx.beginPath();
        ctx.roundRect(labelX, labelY, labelWidth, labelHeight, 8);
        ctx.fill();

        ctx.fillStyle = "rgba(255, 255, 255, 0.98)";
        ctx.font = "800 16px Arial, sans-serif";
        ctx.fillText(label, labelX + 10, labelY + 18);

        ctx.font = "700 10px Arial, sans-serif";
        ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
        ctx.fillText(priorityLabel, labelX + 10, labelY + 33);

        if (zone.rank === 1) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
          ctx.font = "700 10px Arial, sans-serif";
          ctx.fillText(subLabel, labelX + 72, labelY + 33);
        }
      });

      ctx.restore();
    };

    ctx.save();
    ctx.fillStyle = "rgba(2, 6, 23, 0.08)";
    ctx.fillRect(imageRect.x, imageRect.y, imageRect.width, imageRect.height);
    ctx.restore();

    if (analysis.mask_base64) {
      let cancelled = false;
      const overlay = new Image();

      overlay.onload = () => {
        if (cancelled) return;

        ctx.save();
        ctx.globalAlpha = 0.92;
        ctx.drawImage(
          overlay,
          imageRect.x,
          imageRect.y,
          imageRect.width,
          imageRect.height,
        );
        ctx.restore();

        drawZones();
      };

      overlay.onerror = () => {
        if (cancelled) return;
        drawZones();
      };

      overlay.src = `data:image/png;base64,${analysis.mask_base64}`;

      return () => {
        cancelled = true;
      };
    }

    drawZones();
  }, [analysis, containerSize, imageSize, topZones]);

  const handleImageLoad = () => {
    const image = imageRef.current;

    if (!image) return;

    setImageSize({
      width: image.naturalWidth,
      height: image.naturalHeight,
    });

    setStatus("ready");
  };

  const handleImageError = () => {
    setStatus("error");
    setImageSize({ width: 0, height: 0 });
  };

  const showLoading = status === "loading";
  const showError = status === "error";
  const showAnalysisBadge = status === "ready" && Boolean(analysis);
  const showReadyStatus = status === "ready" && Boolean(analysis);

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[540px] overflow-hidden bg-white"
    >
      {postImageUrl ? (
        <img
          ref={imageRef}
          src={postImageUrl}
          alt="Post-disaster satellite imagery"
          className="absolute inset-0 h-full w-full object-contain"
          onLoad={handleImageLoad}
          onError={handleImageError}
          draggable={false}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-white">
          <p className="text-sm text-diq-muted">No post-disaster image loaded.</p>
        </div>
      )}

      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
        role="img"
        aria-label="Damage overlay showing ranked priority zones on the post-disaster image"
      />

      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-slate-950/10 via-transparent to-slate-950/20" />

      {showAnalysisBadge && (
        <div className="absolute left-4 top-16 z-10 flex items-center gap-2 rounded-full border border-red-500/40 bg-red-950/70 px-3 py-1.5 shadow-sm shadow-slate-900/5 ">
          <span className="h-2 w-2 animate-pulse rounded-full bg-red-400 shadow-[0_0_12px_rgba(248,113,113,0.9)]" />

          <span className="text-[10px] font-black uppercase tracking-[0.16em] text-red-800">
            Damage Mask Active
          </span>
        </div>
      )}

      {showLoading && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-white ">
          <div className="rounded-2xl border border-diq-line bg-diq-panel px-7 py-6 text-center shadow-sm shadow-slate-900/5">
            <div className="mx-auto h-9 w-9 animate-spin rounded-full border-2 border-diq-line border-t-diq-orange" />

            <p className="mt-4 font-label text-xs uppercase tracking-[0.18em] text-slate-700">
              Loading imagery
            </p>

            <p className="mt-1 text-sm text-diq-muted">
              Rendering post-disaster view...
            </p>
          </div>
        </div>
      )}

      {showError && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-white">
          <div className="max-w-sm rounded-2xl border border-red-200 bg-red-50 px-6 py-5 text-center shadow-sm shadow-slate-900/5">
            <p className="text-sm font-bold text-red-800">
              Could not load post-disaster image.
            </p>

            <p className="mt-2 text-xs leading-5 text-red-800/70">
              The backend image route may work directly, but the frontend image
              URL being passed into this component is still failing. Check the
              final resolved URL in the browser network tab.
            </p>
          </div>
        </div>
      )}

      {showReadyStatus && (
        <div className="absolute bottom-4 left-4 right-20 z-10 rounded-xl border border-diq-line bg-slate-50 px-4 py-3 shadow-sm shadow-slate-900/5 ">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-diq-muted">
                Overlay Status
              </p>

              <p className="text-sm font-semibold text-diq-ink">
                Damage mask rendered successfully
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-green-50 px-2.5 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-green-800">
                Active
              </span>

              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-diq-muted">
                Top {topZones.length || 0} Zones
              </span>

              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-blue-800">
                AI Classified
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}