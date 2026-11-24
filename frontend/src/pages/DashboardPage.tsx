import { useState, type FormEvent } from "react";
import { AlertCircle, Database, FileSearch, ShieldCheck } from "lucide-react";
import { ForecastForm } from "../components/ForecastForm";
import { ForecastView } from "../components/ForecastView";
import { Header } from "../components/Header";
import { useForecast } from "../hooks/useForecast";
import { useHealth } from "../hooks/useHealth";

const INITIAL_COMPANY = "TCS";
const INITIAL_QUESTION = "Give a qualitative outlook for the upcoming quarter";

export function DashboardPage() {
  const [company, setCompany] = useState(INITIAL_COMPANY);
  const [question, setQuestion] = useState(INITIAL_QUESTION);
  const { health, state, message: healthMessage, refreshing, refresh } = useHealth();
  const { response, loading, error, generate, clear, setError } = useForecast();

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await generate(company, question);
  }

  function reset() {
    clear();
    setCompany(INITIAL_COMPANY);
    setQuestion(INITIAL_QUESTION);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-[#f4f7fb] text-slate-900">
      <Header
        health={health}
        state={state}
        message={healthMessage}
        refreshing={refreshing}
        onRefresh={() => void refresh()}
      />

      <main id="main" className="mx-auto max-w-[1440px] px-4 pb-12 pt-8 sm:px-7 sm:pt-10 lg:px-10">
        <section className="mb-8 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.16em] text-blue-700">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-600" /> Financial intelligence workspace
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">A clearer view of what may come next.</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">
              Explore a qualitative quarterly outlook grounded in TCS filings and management commentary, with source references and evidence limits kept visible.
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600">
              <ShieldCheck size={15} className="text-emerald-600" aria-hidden="true" /> Evidence linked
            </span>
            <span className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600">
              <Database size={15} className="text-blue-700" aria-hidden="true" /> MySQL-backed
            </span>
          </div>
        </section>

        {state === "offline" && (
          <div role="alert" className="mb-5 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3.5 text-sm text-rose-800">
            <AlertCircle className="mt-0.5 shrink-0" size={17} aria-hidden="true" />
            <div><p className="font-semibold">Backend connection unavailable</p><p className="mt-1 text-rose-700">{healthMessage ?? "The backend did not return a health response."}</p></div>
          </div>
        )}
        {state === "degraded" && (
          <div role="status" className="mb-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3.5 text-sm text-amber-900">
            <AlertCircle className="mt-0.5 shrink-0" size={17} aria-hidden="true" />
            <div><p className="font-semibold">Backend is running with limited availability</p><p className="mt-1 text-amber-800">Check the database status before generating a forecast.</p></div>
          </div>
        )}

        <div className="grid items-start gap-5 xl:grid-cols-[minmax(310px,0.78fr)_minmax(0,1.65fr)]">
          <div className="space-y-4 xl:sticky xl:top-24">
            <ForecastForm
              company={company}
              question={question}
              loading={loading}
              onCompanyChange={setCompany}
              onQuestionChange={setQuestion}
              onSubmit={submit}
              onClear={reset}
              hasForecast={response !== null}
            />
            {error && (
              <div role="alert" className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
                <AlertCircle className="mt-0.5 shrink-0" size={17} aria-hidden="true" />
                <div><p className="font-semibold">Forecast not completed</p><p className="mt-1 leading-5">{error}</p></div>
              </div>
            )}
          </div>

          <div className="min-w-0" aria-live="polite">
            {loading ? (
              <section className="rounded-2xl border border-blue-200 bg-white p-7 shadow-sm" aria-label="Forecast loading">
                <div className="flex items-start gap-4">
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700"><FileSearch size={20} className="animate-pulse" aria-hidden="true" /></span>
                  <div className="w-full">
                    <h2 className="text-lg font-semibold text-slate-900">Building an evidence-grounded outlook</h2>
                    <p className="mt-1 text-sm leading-6 text-slate-600">Reviewing available financial statements and management commentary. This can take a little while.</p>
                    <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full w-1/3 animate-pulse rounded-full bg-blue-600" /></div>
                  </div>
                </div>
              </section>
            ) : response ? (
              <ForecastView response={response} />
            ) : (
              <section className="flex min-h-[370px] flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/75 px-6 py-12 text-center sm:min-h-[430px]">
                <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-700"><FileSearch size={25} aria-hidden="true" /></span>
                <p className="mt-5 text-xs font-bold uppercase tracking-[0.15em] text-blue-700">Your analysis workspace</p>
                <h2 className="mt-2 text-xl font-semibold tracking-tight text-slate-900">Your next-quarter outlook will appear here</h2>
                <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">Submit a question to ask the backend to analyze its available filings and earnings-call evidence. No forecast is shown until the API returns one.</p>
                <div className="mt-6 flex flex-wrap justify-center gap-2 text-xs font-medium text-slate-500">
                  <span className="rounded-full bg-slate-100 px-3 py-1.5">Source-aware</span>
                  <span className="rounded-full bg-slate-100 px-3 py-1.5">Qualitative only</span>
                  <span className="rounded-full bg-slate-100 px-3 py-1.5">Missing data stays visible</span>
                </div>
              </section>
            )}
          </div>
        </div>

        <footer className="mt-12 flex flex-col justify-between gap-2 border-t border-slate-200 pt-5 text-xs text-slate-500 sm:flex-row sm:items-center">
          <p>TCS Financial Intelligence <span className="mx-1 text-slate-300">/</span> Evidence-led outlooks</p>
          <p>Forecasts are analytical views, not confirmed outcomes or investment advice.</p>
        </footer>
      </main>
    </div>
  );
}
