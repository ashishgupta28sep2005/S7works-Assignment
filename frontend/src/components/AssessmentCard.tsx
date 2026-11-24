import { AlertCircle, CheckCircle2, CircleHelp, Info } from "lucide-react";
import type { ForecastAssessment } from "../types/api";
import { DirectionBadge } from "./DirectionBadge";
import { SourceReferenceCard } from "./SourceReferenceCard";

const evidenceStyles = {
  supported: "text-emerald-700",
  limited: "text-amber-700",
  conflicting: "text-orange-700",
  unavailable: "text-slate-500",
} as const;

const evidenceIcons = {
  supported: CheckCircle2,
  limited: Info,
  conflicting: AlertCircle,
  unavailable: CircleHelp,
} as const;

export function AssessmentCard({ title, assessment }: { title: string; assessment: ForecastAssessment }) {
  const EvidenceIcon = evidenceIcons[assessment.evidence_status];
  return (
    <article className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/50">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-base font-semibold tracking-tight text-slate-900">{title}</h3>
        <DirectionBadge direction={assessment.direction} />
      </div>
      <p className="mt-4 flex-1 text-sm leading-6 text-slate-700">{assessment.outlook}</p>
      <div className={`mt-4 flex items-center gap-2 text-xs font-semibold capitalize ${evidenceStyles[assessment.evidence_status]}`}>
        <EvidenceIcon size={15} aria-hidden="true" />
        Evidence {assessment.evidence_status}
      </div>
      {assessment.source_references.length > 0 ? (
        <div className="mt-4 space-y-3">
          {assessment.source_references.map((reference) => (
            <SourceReferenceCard key={reference.source.source_id} reference={reference} />
          ))}
        </div>
      ) : assessment.evidence_status === "unavailable" ? (
        <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
          The backend returned no supporting source references for this unavailable outlook.
        </p>
      ) : null}
    </article>
  );
}
