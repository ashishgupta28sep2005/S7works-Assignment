import type { FormEvent } from "react";
import { LoaderCircle, Sparkles } from "lucide-react";

export function ForecastForm({
  company,
  question,
  loading,
  onCompanyChange,
  onQuestionChange,
  onSubmit,
  onClear,
  hasForecast,
}: {
  company: string;
  question: string;
  loading: boolean;
  onCompanyChange: (value: string) => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onClear: () => void;
  hasForecast: boolean;
}) {
  return (
    <form noValidate onSubmit={onSubmit} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/60 sm:p-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-blue-700">Forecast workspace</p>
          <h2 className="mt-2 text-xl font-semibold tracking-tight text-slate-900">Generate an outlook</h2>
        </div>
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
          <Sparkles size={19} aria-hidden="true" />
        </span>
      </div>

      <div className="mt-6 space-y-5">
        <div>
          <label htmlFor="company" className="mb-2 block text-sm font-semibold text-slate-700">Company</label>
          <input
            id="company"
            name="company"
            value={company}
            onChange={(event) => onCompanyChange(event.target.value)}
            maxLength={255}
            required
            autoComplete="organization"
            className="h-11 w-full rounded-xl border border-slate-300 bg-white px-3.5 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
          />
        </div>
        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <label htmlFor="question" className="block text-sm font-semibold text-slate-700">Forecast question</label>
            <span className="text-xs text-slate-400">{question.length}/4000</span>
          </div>
          <textarea
            id="question"
            name="question"
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            maxLength={4000}
            required
            rows={4}
            className="w-full resize-y rounded-xl border border-slate-300 bg-white px-3.5 py-3 text-sm leading-6 text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
          />
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-xl bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700 disabled:cursor-wait disabled:bg-blue-400"
        >
          {loading ? <LoaderCircle size={17} className="animate-spin" aria-hidden="true" /> : <Sparkles size={16} aria-hidden="true" />}
          {loading ? "Generating forecast…" : "Generate forecast"}
        </button>
        {hasForecast && (
          <button
            type="button"
            onClick={onClear}
            disabled={loading}
            className="min-h-11 rounded-xl border border-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
          >
            Reset workspace
          </button>
        )}
      </div>
      <p className="mt-4 text-xs leading-5 text-slate-500">
        Forecasts are qualitative and grounded in the documents currently available to the backend.
      </p>
    </form>
  );
}
