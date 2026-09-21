import type { DispatchStatus, Severity } from "./types";

export const SEVERITY_COLOR: Record<Severity, string> = {
  none: "#94a3b8",
  minor: "#eab308",
  major: "#f97316",
  destroyed: "#dc2626",
};

export const ROAD_COLOR = { open: "#16a34a", degraded: "#d97706", blocked: "#dc2626" };

export const SEVERITY_BADGE: Record<Severity, string> = {
  none: "bg-slate-100 text-slate-600",
  minor: "bg-yellow-100 text-yellow-800",
  major: "bg-orange-100 text-orange-800",
  destroyed: "bg-red-100 text-red-800",
};

export const STATUS_BADGE: Record<DispatchStatus, string> = {
  proposed: "bg-slate-100 text-slate-700",
  reserved: "bg-blue-100 text-blue-800",
  in_transit: "bg-violet-100 text-violet-800",
  delivered: "bg-emerald-100 text-emerald-800",
  cancelled: "bg-slate-200 text-slate-500 line-through",
};

export const NEXT_STATUS: Record<DispatchStatus, { to: DispatchStatus; label: string }[]> = {
  proposed: [{ to: "reserved", label: "Approve" }, { to: "cancelled", label: "Reject" }],
  reserved: [{ to: "in_transit", label: "Depart" }, { to: "cancelled", label: "Cancel" }],
  in_transit: [{ to: "delivered", label: "Delivered" }, { to: "cancelled", label: "Cancel" }],
  delivered: [],
  cancelled: [],
};

export function fmt(n: number | undefined | null): string {
  return (n ?? 0).toLocaleString("en-US");
}
