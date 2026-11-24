import { AlertTriangle, ArrowRight, BarChart3, BookOpen, BriefcaseBusiness, Check, CircleHelp, Gauge, Lightbulb, ListChecks, TrendingUp } from "lucide-react";
import type { ForecastAPIResponse, SourceReference } from "../types/api";
import { collectSources } from "../utils/forecast";
import { AssessmentCard } from "./AssessmentCard";
import { SourceReferenceCard } from "./SourceReferenceCard";

function StatCard({ label, value, icon: Icon, detail }: { label: string; value: string; icon: typeof Gauge; detail?: string }) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm shadow-slate-200/40">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</span>
        <Icon size={16} className="text-blue-700" aria-hidden="true" />
      </div>
      <p className="mt-3 truncate text-xl font-bold tracking-tight text-slate-950">{value}</p>
      {detail && <p className="mt-1 truncate text-xs text-slate-500">{detail}</p>}
    </article>
  );
}

function ListSection({
  title,
  icon: Icon,
  items,
  emptyText,
}: {
  title: string;
  icon: typeof Gauge;
  items: { item: string; reasoning: string; source_references: SourceReference[] }[];
  emptyText: string;
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/40">
      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-700"><Icon size={16} aria-hidden="true" /></span>
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      </div>
      {items.length === 0 ? (
        <p className="mt-4 rounded-xl bg-slate-50 px-3.5 py-3 text-sm leading-6 text-slate-500">{emptyText}</p>
      ) : (
        <ul className="mt-4 space-y-3">
          {items.map((item, index) => (
            <li key={`${title}-${index}`} className="rounded-xl border border-slate-100 bg-slate-50/80 p-3.5">
              <p className="text-sm font-semibold text-slate-800">{item.item}</p>
              <p className="mt-1.5 text-sm leading-6 text-slate-600">{item.reasoning}</p>
              {item.source_references.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {item.source_references.map((ref) => (
                    <span key={ref.source.source_id} className="rounded-md bg-white px-2 py-1 font-mono text-[10px] font-semibold text-blue-700 ring-1 ring-slate-200">
                      {ref.source.source_id}
                    </span>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function ForecastView({ response }: { response: ForecastAPIResponse }) {
  const forecast = response.forecast;
  const sources = collectSources(response);
  const uniquePeriods = [...new Set(forecast.based_on_reporting_periods)];
  const confidenceText = forecast.confidence_level[0].toUpperCase() + forecast.confidence_level.slice(1);

  return (
    <div className="space-y-6" aria-label="Forecast result">
      <section className="overflow-hidden rounded-2xl border border-blue-200 bg-white shadow-sm shadow-blue-100/70">
        <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="p-5 sm:p-7">
            <div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-[0.15em] text-blue-700">
              <span>Quarterly outlook</span><ArrowRight size={13} aria-hidden="true" /><span>{forecast.forecast_for}</span>
            </div>
            <h2 className="mt-3 text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">{forecast.company} outlook</h2>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-600">Confidence: {confidenceText}</span>
              {uniquePeriods.map((period) => <span key={period} className="rounded-full bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700">Based on {period}</span>)}
            </div>
            <p className="mt-6 max-w-3xl text-base leading-7 text-slate-700">{forecast.overall_business_outlook.outlook}</p>
            <div className="mt-5 flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Overall direction</span>
              <span className="rounded-full bg-blue-50 px-3 py-1 text-sm font-semibold capitalize text-blue-800">{forecast.overall_business_outlook.direction}</span>
            </div>
          </div>
          <aside className="flex flex-col justify-between border-t border-slate-100 bg-slate-50/80 p-5 lg:border-l lg:border-t-0">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.13em] text-slate-500">Request reference</p>
              <p className="mt-2 break-all font-mono text-xs leading-5 text-slate-700">{response.request_id}</p>
              <p className="mt-4 text-xs leading-5 text-slate-500">Saved with the forecast response by the backend.</p>
            </div>
            <div className="mt-5 flex items-center gap-2 text-xs font-semibold text-emerald-700">
              <Check size={15} aria-hidden="true" /> Validated response
            </div>
          </aside>
        </div>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Forecast quarter" value={forecast.forecast_for} icon={TrendingUp} detail="Upcoming reporting period" />
        <StatCard label="Confidence" value={confidenceText} icon={Gauge} detail="Calibrated to available evidence" />
        <StatCard label="Source periods" value={String(uniquePeriods.length)} icon={BookOpen} detail={uniquePeriods.join(" · ") || "None returned"} />
        <StatCard label="Source records" value={String(sources.length)} icon={ListChecks} detail="Distinct references returned" />
      </div>

      <section>
        <div className="mb-4 flex items-end justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.15em] text-blue-700">Outlook by area</p>
            <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-950">Financial & management view</h2>
          </div>
          <span className="hidden text-xs text-slate-500 sm:inline">Every section uses the returned evidence status.</span>
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <AssessmentCard title="Revenue" assessment={forecast.revenue_outlook} />
          <AssessmentCard title="Profitability" assessment={forecast.profitability_outlook} />
          <AssessmentCard title="Operating margin" assessment={forecast.operating_margin_outlook} />
          <AssessmentCard title="Management outlook" assessment={forecast.management_outlook} />
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <ListSection title="Business risks" icon={AlertTriangle} items={forecast.business_risks} emptyText="No business risks were returned by the forecast service." />
        <ListSection title="Business opportunities" icon={BriefcaseBusiness} items={forecast.business_opportunities} emptyText="No business opportunities were returned by the forecast service." />
        <ListSection title="Key growth drivers" icon={Lightbulb} items={forecast.key_growth_drivers} emptyText="No growth drivers were returned by the forecast service." />
        <ListSection title="Key monitoring indicators" icon={ListChecks} items={forecast.key_monitoring_indicators} emptyText="No monitoring indicators were returned by the forecast service." />
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(300px,0.8fr)]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/40 sm:p-6">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-700"><BookOpen size={16} aria-hidden="true" /></span>
            <div>
              <h2 className="text-base font-semibold text-slate-900">Source references</h2>
              <p className="mt-0.5 text-xs text-slate-500">Documents and evidence returned by the API.</p>
            </div>
          </div>
          {sources.length > 0 ? (
            <div className="mt-4 grid gap-3 lg:grid-cols-2">
              {sources.map((source) => <SourceReferenceCard key={source.source.source_id} reference={source} />)}
            </div>
          ) : (
            <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5 text-center">
              <CircleHelp className="mx-auto text-slate-400" size={22} aria-hidden="true" />
              <p className="mt-2 text-sm font-semibold text-slate-700">No source references returned</p>
              <p className="mt-1 text-xs text-slate-500">The application has not added or inferred any source details.</p>
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/40">
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-50 text-amber-700"><AlertTriangle size={16} aria-hidden="true" /></span>
              <h2 className="text-base font-semibold text-slate-900">Data limitations</h2>
            </div>
            {forecast.data_limitations.length > 0 ? (
              <ul className="mt-4 space-y-2.5">
                {forecast.data_limitations.map((limitation, index) => (
                  <li key={`${limitation}-${index}`} className="flex gap-2.5 text-sm leading-6 text-slate-700">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />{limitation}
                  </li>
                ))}
              </ul>
            ) : <p className="mt-4 text-sm text-slate-500">No data limitations were returned.</p>}
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/40">
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-600"><BarChart3 size={16} aria-hidden="true" /></span>
              <h2 className="text-base font-semibold text-slate-900">Historical performance</h2>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">Historical chart data is not currently exposed by the forecast API. No figures are charted here.</p>
          </div>
        </aside>
      </section>

      {forecast.reasoning.length > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/40 sm:p-6">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-700"><Lightbulb size={16} aria-hidden="true" /></span>
            <h2 className="text-base font-semibold text-slate-900">Forecast reasoning</h2>
          </div>
          <ol className="mt-4 space-y-3">
            {forecast.reasoning.map((step, index) => (
              <li key={`reason-${index}`} className="flex gap-3 rounded-xl bg-slate-50 p-4">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white text-xs font-bold text-slate-600 ring-1 ring-slate-200">{index + 1}</span>
                <div className="min-w-0">
                  <p className="text-sm leading-6 text-slate-700">{step.inference}</p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {step.source_references.map((ref) => <span key={ref.source.source_id} className="rounded-md bg-white px-2 py-1 font-mono text-[10px] font-semibold text-blue-700 ring-1 ring-slate-200">{ref.source.source_id}</span>)}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
