import { FileText, MessageSquareQuote } from "lucide-react";
import type { SourceReference } from "../types/api";
import { sourceDocument, sourcePage, sourcePeriod, sourceRelevance, sourceSpeaker } from "../utils/forecast";

export function SourceReferenceCard({ reference }: { reference: SourceReference }) {
  const source = reference.source;
  const document = sourceDocument(source);
  const period = sourcePeriod(source);
  const speaker = sourceSpeaker(source);
  const relevance = sourceRelevance(source);
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
          {speaker ? <MessageSquareQuote size={17} aria-hidden="true" /> : <FileText size={17} aria-hidden="true" />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="break-all text-sm font-semibold text-slate-900">{document}</h4>
            <span className="rounded-md bg-slate-100 px-2 py-1 font-mono text-[11px] font-semibold text-slate-600">
              {source.source_id}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
            {period && <span>{period}</span>}
            <span>Page {sourcePage(source)}</span>
            {speaker && <span>Speaker: {speaker}</span>}
            {relevance !== null && <span>Relevance {(relevance * 100).toFixed(0)}%</span>}
            {"metric_name" in source && <span className="capitalize">Metric: {source.metric_name.replaceAll("_", " ")}</span>}
          </div>
          {reference.supporting_evidence.length > 0 && (
            <ul className="mt-3 space-y-2">
              {reference.supporting_evidence.map((evidence, index) => (
                <li key={`${source.source_id}-${index}`} className="border-l-2 border-blue-200 pl-3 text-sm leading-6 text-slate-700">
                  {evidence}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </article>
  );
}
