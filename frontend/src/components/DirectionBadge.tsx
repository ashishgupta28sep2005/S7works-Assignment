import { ArrowDownRight, ArrowUpRight, Minus, CircleHelp } from "lucide-react";
import type { ForecastAssessment } from "../types/api";

const styles = {
  positive: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  stable: "bg-sky-50 text-sky-700 ring-sky-200",
  cautious: "bg-amber-50 text-amber-800 ring-amber-200",
  negative: "bg-rose-50 text-rose-700 ring-rose-200",
  uncertain: "bg-slate-100 text-slate-600 ring-slate-200",
} as const;

const icons = {
  positive: ArrowUpRight,
  stable: Minus,
  cautious: ArrowDownRight,
  negative: ArrowDownRight,
  uncertain: CircleHelp,
} as const;

export function DirectionBadge({ direction }: { direction: ForecastAssessment["direction"] }) {
  const Icon = icons[direction];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold capitalize ring-1 ring-inset ${styles[direction]}`}>
      <Icon size={14} aria-hidden="true" />
      {direction}
    </span>
  );
}
