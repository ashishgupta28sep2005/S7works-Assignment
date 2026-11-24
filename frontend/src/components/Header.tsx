import { Activity, BookOpenText, ShieldCheck } from "lucide-react";
import { API_BASE_URL } from "../services/api";
import { HealthStatus } from "./HealthStatus";
import type { HealthState } from "../hooks/useHealth";
import type { HealthResponse } from "../types/api";

export function Header(props: {
  health: HealthResponse | null;
  state: HealthState;
  message: string | null;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-7 lg:px-10">
        <a href="#main" className="group flex min-w-0 items-center gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-700 text-white shadow-sm shadow-blue-900/15">
            <Activity size={20} aria-hidden="true" />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-bold tracking-tight text-slate-950 sm:text-base">TCS Financial Intelligence</span>
            <span className="mt-0.5 flex items-center gap-1.5 text-[11px] font-medium text-slate-500">
              <ShieldCheck size={12} className="text-emerald-600" aria-hidden="true" /> Evidence-first forecasting
            </span>
          </span>
        </a>

        <div className="flex flex-wrap items-center justify-end gap-3">
          <HealthStatus {...props} />
          <a
            href={`${API_BASE_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600 transition hover:border-blue-200 hover:text-blue-700 sm:inline-flex"
          >
            <BookOpenText size={14} aria-hidden="true" /> API docs
          </a>
        </div>
      </div>
    </header>
  );
}
