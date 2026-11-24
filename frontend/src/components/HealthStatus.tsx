import { Activity, Database, RefreshCw } from "lucide-react";
import type { HealthState } from "../hooks/useHealth";
import type { HealthResponse } from "../types/api";

const backendLabels: Record<HealthState, string> = {
  checking: "Checking backend",
  connected: "Backend online",
  degraded: "Backend degraded",
  offline: "Backend offline",
};

function statusTone(state: HealthState | "connected" | "unavailable" | "not_configured") {
  if (state === "connected") return "bg-emerald-50 text-emerald-700 ring-emerald-200";
  if (state === "checking") return "bg-slate-100 text-slate-600 ring-slate-200";
  return "bg-rose-50 text-rose-700 ring-rose-200";
}

export function HealthStatus({
  state,
  health,
  message,
  refreshing,
  onRefresh,
}: {
  state: HealthState;
  health: HealthResponse | null;
  message: string | null;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  const dbState = health?.database_status ?? (state === "offline" ? "unavailable" : "not_configured");
  const dbLabel = dbState === "connected" ? "Database connected" : dbState === "unavailable" ? "Database unavailable" : "Database not configured";
  return (
    <div className="flex flex-wrap items-center gap-2" aria-live="polite">
      <span className={`inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold ring-1 ring-inset ${statusTone(state)}`}>
        <Activity size={14} aria-hidden="true" />
        {backendLabels[state]}
      </span>
      <span className={`inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold ring-1 ring-inset ${statusTone(dbState)}`}>
        <Database size={14} aria-hidden="true" />
        {dbLabel}
      </span>
      <button
        type="button"
        onClick={onRefresh}
        disabled={refreshing}
        aria-label="Refresh connection status"
        className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 transition hover:border-blue-200 hover:text-blue-700 disabled:cursor-wait disabled:opacity-60"
      >
        <RefreshCw size={15} className={refreshing ? "animate-spin" : ""} aria-hidden="true" />
      </button>
      {message && <span className="sr-only">{message}</span>}
    </div>
  );
}
