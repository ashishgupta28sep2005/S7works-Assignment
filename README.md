# TCS Financial Forecasting AI Agent

A modular Python application scaffold for producing a source-grounded qualitative outlook for Tata Consultancy Services (TCS). Phase 2 adds local PDF ingestion, content-based document classification, page-level text and table extraction, duplicate detection, and ingestion status reporting.

## Environment

Use Python 3.12.10 and the virtual environment created in Phase 1:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Phase 2 runtime and test dependencies (the remaining requirements are for later project phases and are not needed yet):

```powershell
python -m pip install "PyMuPDF>=1.24,<2.0" "pdfplumber>=0.11,<1.0" "python-dotenv>=1.0,<2.0" "pydantic>=2.8,<3.0" "pytest>=8.0,<9.0"
```

The corresponding complete project dependency list is in `requirements.txt`. LangChain, vector database, embedding, API, and MySQL packages are listed for later phases and are intentionally not required for the Phase 2 ingestion command.

## PDF locations and processing

Place financial reports in `data/financial_reports/` and earnings call transcripts in `data/earnings_transcripts/`. Classification uses extracted document content and headings, not the containing directory or filename alone. Unknown or ambiguous content remains `UNKNOWN` with an explanation.

Run from the project root:

```powershell
python -m app.ingestion.document_manager
```

The command logs per-file failures and prints a compact JSON summary with document classification, page and table counts, duplicate counts, and OCR-required statuses. Detailed extracted page text and tables are available through the returned `ProcessedDocument` models. Scanned PDFs are detected when pages contain images but little extractable text; OCR is not performed and no text is invented. Corrupted or password-protected PDFs are reported as failures while other files continue processing.

## Phase 3 financial extraction

Run the deterministic financial statement extractor from the project root:

```powershell
python -m app.tools.financial_data_extractor
```

It reuses the Phase 2 PDF loader and classifier, reads `data/financial_reports/`, and writes repeatable structured JSON to `outputs/financial_extractions/financial_data.json`. Every extracted metric carries source filename, page, source row text, statement basis, original formatted value, normalized Decimal value, reporting period, and unit. Missing metrics remain null and appear in the statement's `unrecognized_metrics` list.

The extractor keeps consolidated and standalone statements in separate records and calculates quarter-over-quarter comparisons only for matching quarterly periods and statement bases. The March 2026 report contributes FY26 figures only where the annual column is identifiable. Its standalone section contains quarter-only figures, so that record is marked for review and its values are left unset. Operating profit and operating margin are left missing where no explicit row is found; they are not inferred from profit before tax.

Growth percentages are calculated from the same-period comparative values shown in the source table. Quarter-over-quarter comparisons are separately labeled and do not represent year-over-year growth. Decimal values are serialized as JSON strings to preserve precision. Example verification:

```powershell
pytest
python -m app.tools.financial_data_extractor
```

`tests/test_financial_extractor.py` uses synthetic statement layouts. `tests/test_financial_extractor_integration.py` checks the actual report PDFs in `data/financial_reports/` and skips if those files are absent.

Run technical ingestion tests (all PDF fixtures are synthetic and are not TCS evidence):

```powershell
pytest
```

## Phase 4 transcript retrieval

Phase 4 adds transcript-only, page-grounded retrieval using speaker-aware text chunks, a local Hugging Face sentence-transformer, and a persistent FAISS cosine-similarity index. Financial reports are deliberately excluded from this transcript index. Each retrieved passage retains its source PDF, reporting period, page number, and speaker when available. A configurable similarity threshold marks weak matches as not relevant; an empty or low-relevance result should not be treated as evidence.

The model and index are local. The model is downloaded on first use and cached in `.cache/sentence-transformers/`; vector index generations are written to `vectorstore/`. Install the tested Phase 4 embedding stack into the existing venv:

```powershell
python -m pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Index transcripts from the project root:

```powershell
python -m app.rag.ingest_transcripts
```

Use `--rebuild` to build a fresh index generation from all available transcript PDFs. Re-running without that flag skips already indexed source documents by content hash. The index verifies checksums, model identity, vector dimensions, and chunk metadata before loading. Previous generations are retained as recovery points. Current search supports optional exact reporting-period and filename filters; see `TranscriptRetriever` in `app/rag/retriever.py` for application use.

Only two transcript PDFs are currently present in `data/earnings_transcripts/`; the requested third historical transcript is missing. The ingestion command reports this limitation. No fabricated transcript text is used. The embedding model runs on CPU by default and may require internet access for its first download.

Run tests from the project root:

```powershell
python -m pytest
```

RAG pipeline test data is synthetic and solely for technical behavior; actual TCS documents are covered only by the separate integration tests.

## Phase 5 qualitative transcript analysis

`QualitativeAnalysisTool` combines the Phase 4 retriever with a configurable LLM provider. Groq is the initial provider and uses the supported `langchain-groq` integration. Set `LLM_PROVIDER=groq`, `LLM_MODEL`, and `GROQ_API_KEY` in a local `.env` file copied from `.env.example`; never commit the filled `.env`. The tool raises a clear configuration error if the key is missing. Provider creation is separate from analysis, and the constructor accepts an injected LLM for tests and alternate providers.

Example application use:

```python
from app.config.settings import get_settings
from app.rag.embeddings import HuggingFaceEmbedder
from app.rag.retriever import TranscriptRetriever
from app.rag.vector_store import PersistentFaissStore
from app.tools.qualitative_analysis import QualitativeAnalysisTool

settings = get_settings()
embedder = HuggingFaceEmbedder()
store = PersistentFaissStore.open(
    settings.vectorstore_dir,
    dimension=embedder.dimension,
    embedding_model=settings.embedding_model,
)
tool = QualitativeAnalysisTool(TranscriptRetriever(store, embedder))
result = tool.analyze("Summarize management's commentary on demand and margins")
print(result.model_dump_json(indent=2, exclude_none=True))
```

By default it retrieves separately for Q4 FY2025-26 and Q2 FY27 to reduce period imbalance, expanding retrieval candidates per period before filtering to chunks marked as management speech. Phase 4 chunks now retain a backward-compatible speaker-role field and separate dialogue at speaker-role boundaries; rebuild the index with `python -m app.rag.ingest_transcripts --rebuild` after upgrading so the stored role labels are populated. Analyst and moderator passages are excluded from qualitative evidence, preventing analyst questions from being treated as management commitments. Pass `reporting_periods=[...]` to analyze selected indexed periods. The Pydantic result contains recurring themes, period-grounded management sentiment, forward-looking statements, risks, opportunities, limitations, and source references with actual document, reporting period, page, optional speaker, and retrieval score. A purported recurring theme is retained only if its cited passages span at least two reporting periods.

The LLM receives only retrieved passages and source IDs. The prompt prohibits outside knowledge, fabricated figures, interpreting analyst questions as management commitments, and confusing historical performance with guidance. The application maps model-provided IDs to trusted retrieved metadata, rejects unknown IDs, checks quoted evidence against cited passages, and rejects numerical claims not present in their cited text. Empty retrieval returns an explicit limitation without calling the LLM. These checks support grounding but do not guarantee the model's interpretation is correct; review analysis against the cited passages.

Automated tests inject mocked retrievers and LLMs and make no external or paid calls:

```powershell
python -m pytest tests/test_qualitative_analysis.py -q
python -m pytest -q
```

## Phase 6 master financial analysis

`FinancialAnalysisAgent` uses deterministic orchestration: it calls the Phase 3 `FinancialDataExtractorTool` runner once, calls the Phase 5 `QualitativeAnalysisTool` once, validates their Pydantic outputs, then makes one structured Groq request to synthesize cross-analysis. The Phase 5 and master calls share the same `ChatGroq` client; client construction lives in `app/config/llm.py`, separate from orchestration. LangGraph is not needed for these two fixed tool calls.

The current document corpus is TCS-only. The agent accepts a company name and question, discovers financial periods from extracted statements and transcript periods from the management-only index, and returns `FinancialAnalysisResult`. It embeds the complete original specialist outputs unchanged, plus cross-analysis findings, data-quality notes, and component statuses. Findings cite application-generated financial metric IDs and transcript source IDs; those IDs are resolved into the source document, page, period, and statement basis returned by the tools. Cross-analysis validation rejects mixed consolidated/standalone or annual/quarterly references and unsupported numbers. Missing metrics remain null and are reported as data-quality notes.

Set these values in the ignored local `.env` file (the example file contains placeholders only):

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TIMEOUT_SECONDS=45
RUN_GROQ_INTEGRATION_TESTS=false
```

`GROQ_MODEL` takes precedence; `LLM_MODEL` remains a compatibility fallback for earlier local configurations. The configured model is not silently replaced. Provider authentication, rate-limit, timeout, and invalid-structured-response failures are returned as sanitized status messages. Automatic provider retries are disabled. If either specialist fails, the other result is retained and the overall status is `partial_success`; if both fail, the result is `failed`. If master reasoning fails, both specialist outputs are still preserved.

Run from the project root:

```powershell
python -m app.agents.financial_analysis_agent --company TCS --question "Relate reported revenue to management's demand commentary"
```

The response is JSON with `company`, `analysis_question`, `reporting_periods`, `financial_analysis`, `qualitative_analysis`, `cross_analysis`, `data_quality_notes`, and `tool_execution_status`. The two specialist output fields preserve their own metric-level and transcript-level source provenance; the master can only reference source IDs produced by those outputs.

Valid empty-evidence response shape (no figures or claims invented):

```json
{
  "company": "TCS",
  "analysis_question": "Relate reported revenue to management's demand commentary",
  "reporting_periods": [],
  "financial_analysis": {
    "company": "TCS",
    "statements": [],
    "quarterly_comparisons": [],
    "errors": []
  },
  "qualitative_analysis": {
    "company": "TCS",
    "analysis_question": "Relate reported revenue to management's demand commentary",
    "reporting_periods_analyzed": [],
    "recurring_themes": [],
    "management_sentiment": [],
    "forward_looking_statements": [],
    "business_risks": [],
    "business_opportunities": [],
    "limitations": ["No relevant transcript passages were retrieved; no analysis was generated."],
    "sources": []
  },
  "cross_analysis": [],
  "data_quality_notes": [],
  "tool_execution_status": {
    "financial_extraction": "success",
    "qualitative_analysis": "success",
    "master_reasoning": "success",
    "overall_status": "success",
    "errors": {}
  }
}
```

With the configured corpus, the specialist fields contain their full actual extracted results instead of empty lists. Automated examples use mocks; see the optional live Phase 5 check below for the explicitly enabled Groq integration test.

Normal tests make no Groq calls. The separate live Phase 5 integration test is skipped by default. To explicitly enable its single request using the configured key and model:

```powershell
$env:RUN_GROQ_INTEGRATION_TESTS = "true"
python -m pytest tests/test_groq_integration.py -q
```

Unset the variable afterward or start a fresh terminal. The committed `.env.example` keeps this flag set to `false`.

## Phase 7 upcoming-quarter qualitative forecast

`ForecastAgent` in `app/agents/forecast_agent.py` calls `FinancialAnalysisAgent.analyze()` once, then performs one structured forecast synthesis using that agent's shared Groq client. It derives the next quarter from the latest explicitly identified quarter and returns `QuarterlyForecast` from `app/schemas/forecast.py`. It does not run an API server or persist forecasts.

Run from the project root after configuring `.env` as described in Phase 6:

```powershell
python -m app.agents.forecast_agent --company TCS --question "Give a qualitative outlook for the upcoming quarter"
```

Forecasts are qualitative. Numerical future claims are rejected, and every outlook item must cite source IDs returned by Phase 6. The app resolves those IDs to the original document, reporting period, page, metric, or transcript speaker. Only quarterly consolidated financial metrics are used for financial trends; annual and standalone statements are excluded. When the extracted latest-quarter operating-margin metric is absent, that outlook is explicitly unavailable even if historical transcript commentary discusses margin. The output also carries data limitations and a confidence level constrained by available evidence.

Illustrative source-grounded example (historical evidence only; this is not a generated forecast): Phase 3 extracted consolidated quarterly revenue of INR 72,275 crores and net profit of INR 13,420 crores for Q1 FY27 from `June-2026TCS.pdf`, page 8. For Q2 FY27 it extracted revenue of INR 73,188 crores and net profit of INR 13,934 crores from `Sep-2026TCS.pdf`, page 8. Those two quarters establish Q3 FY27 as the next period. The extracted operating-margin field is unavailable in both reports. Separately, the Q2 FY27 earnings-call transcript (`CC-Oct26tcs.pdf`, page 5; Samir Seksaria) records historical management commentary that constant-currency growth was 50 basis points sequentially and 2.8% year over year, and that operating margins remained stable sequentially at 24%. That quote is historical context, not a future commitment and does not fill the missing extracted financial-statement margin metric.

Normal automated forecast tests inject mocked Phase 6 and LLM components and make no network calls. The optional live structured forecast test is skipped unless explicitly enabled:

```powershell
$env:RUN_GROQ_INTEGRATION_TESTS = "true"
python -m pytest tests/test_forecast_agent.py -q
```

That optional test uses a clearly synthetic Example Corp fixture and makes one Groq request. Keep the flag unset or `false` for ordinary test runs.

## Phase 8 MySQL persistence

Phase 8 adds a SQLAlchemy 2.x persistence layer with the PyMySQL driver. The connection reads `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, and `MYSQL_DATABASE` from the ignored local `.env` file or process environment. `.env.example` contains placeholders only; copy the values into `.env` without replacing the existing Groq settings. This project expects the already-created database `tcs_financial_forecasting` on MySQL Server 8.0.46. The app does not create or drop databases.

If setting up a fresh local instance, create the database once as a MySQL administrator:

```sql
CREATE DATABASE IF NOT EXISTS tcs_financial_forecasting
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
```

Then set the local database variables in `.env` (replace the placeholders locally; do not commit the file):

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=tcs_financial_forecasting
```

`app/database/connection.py` validates configuration and constructs a credential-safe SQLAlchemy URL. `app/database/models.py` defines the tables and constraints. `app/database/repository.py` performs transactional writes and lookup. `app/database/initialize.py` creates only missing tables via `metadata.create_all(checkfirst=True)`; it never drops existing objects. The request service in `app/services/forecast_request_service.py` is callable without a web framework and keeps database work separate from the Phase 7 reasoning pipeline.

The `forecast_requests` table stores the unique request ID, company, question, UTC timestamps, status, and sanitized error text. `forecast_responses` stores a validated forecast JSON document, forecast period, confidence, and response timestamp. Its unique foreign key links one response to one request. Credentials are not included in either table.

Initialize the tables from the project root after the database exists and `.env` is configured:

```powershell
python -m app.database.initialize
```

Run the request logging demonstration (this invokes Phase 7 and therefore requires the configured Groq credentials and local transcript index):

```powershell
python -m app.services.forecast_request_service --company TCS --question "Give a qualitative outlook for the upcoming quarter"
```

The service creates a UUID request record in `processing`, calls the existing forecast agent, and atomically saves the forecast JSON with `completed` status. If generation fails, it records `failed` with a fixed sanitized message; raw exception text, keys, and passwords are not persisted or printed. `ForecastRepository.get_request(request_id)` returns the request with its related response loaded.

Ordinary database tests use in-memory SQLite and do not require MySQL, Groq, or network access:

```powershell
python -m pytest tests/test_database.py -q
python -m pytest -q
```

An optional local MySQL connection/table check is disabled by default. Enable it only when `.env` contains valid database settings and MySQL is running:

```powershell
$env:RUN_MYSQL_INTEGRATION_TESTS = "true"
python -m pytest tests/test_database.py -q
```

## Phase 9 HTTP API

`app/main.py` creates the FastAPI application and composes the existing SQLAlchemy connection, `ForecastRepository`, and Phase 8 `ForecastRequestService`. The router delegates forecast generation and persistence entirely to that service; Phase 7 schemas and reasoning are reused unchanged. Startup does not create or drop database tables—run the Phase 8 initializer first. If database configuration is missing, the app still starts in degraded mode so `/health` can report the condition safely.

### Endpoints

- `GET /health` returns application status and database connectivity (`connected`, `unavailable`, or `not_configured`). It never returns connection details.
- `POST /api/v1/forecast` validates `company` and `question`, invokes the Phase 8 service, and returns its request ID and existing `QuarterlyForecast` object.

Request JSON:

```json
{
  "company": "TCS",
  "question": "Give a qualitative outlook for the upcoming quarter"
}
```

Successful response shape (illustrative schema example, not a generated forecast):

```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "forecast": {
    "company": "TCS",
    "forecast_for": "Q3 FY27",
    "based_on_reporting_periods": ["Q2 FY27"],
    "overall_business_outlook": {"direction": "uncertain", "evidence_status": "unavailable", "outlook": "Insufficient evidence is available for this outlook.", "source_references": []},
    "revenue_outlook": {"direction": "uncertain", "evidence_status": "unavailable", "outlook": "Insufficient evidence is available for this outlook.", "source_references": []},
    "profitability_outlook": {"direction": "uncertain", "evidence_status": "unavailable", "outlook": "Insufficient evidence is available for this outlook.", "source_references": []},
    "operating_margin_outlook": {"direction": "uncertain", "evidence_status": "unavailable", "outlook": "Insufficient evidence is available for this outlook.", "source_references": []},
    "management_outlook": {"direction": "uncertain", "evidence_status": "unavailable", "outlook": "Insufficient evidence is available for this outlook.", "source_references": []},
    "business_risks": [],
    "business_opportunities": [],
    "key_growth_drivers": [],
    "key_monitoring_indicators": [],
    "confidence_level": "low",
    "reasoning": [],
    "source_references": [],
    "data_limitations": ["Illustrative response shape only."]
  }
}
```

HTTP status codes: `200` for a completed forecast; `422` for invalid or missing request fields; `502` when the Phase 8 service records a forecast failure; `503` when the request cannot be persisted because the database/service is unavailable; `500` for an unexpected internal failure. Error bodies are generic and do not echo rejected values, provider exceptions, or connection details. Server logs retain the exception category without logging its message.

### Configuration and run

Use the existing `.env` values documented in Phase 8 and Phase 6. The API uses the same `MYSQL_*` and `GROQ_*` configuration; it does not introduce another configuration system. Never replace the local `.env` with `.env.example` and do not commit credentials. Confirm the existing database tables are initialized before starting.

Start from the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open Swagger UI at `http://127.0.0.1:8000/docs`, ReDoc at `http://127.0.0.1:8000/redoc`, and the OpenAPI document at `http://127.0.0.1:8000/openapi.json`.

Run all automated tests (API and database unit tests use mocks or in-memory SQLite; no Groq or MySQL request is made):

```powershell
python -m pytest -q
```

The opt-in end-to-end API integration test performs a real forecast call through Groq and writes a request/response row to MySQL. Enable it only intentionally with valid local configuration:

```powershell
$env:RUN_API_INTEGRATION_TESTS = "true"
python -m pytest tests/test_api_integration.py -q
```

## Phase 12 React web application

The separate `frontend/` application is built with React, TypeScript, Vite, and Tailwind CSS. It presents the real FastAPI health and forecast responses; it contains no forecasting logic, credentials, generated example forecast, or local storage. No chart is drawn because `/api/v1/forecast` does not return a historical financial time series. The interface says so directly.

The browser client validates response payloads against runtime Zod schemas based on `HealthResponse`, `ForecastAPIResponse`, and the existing `QuarterlyForecast` model. Its API origin is configured with `VITE_API_BASE_URL` in `frontend/.env`; copy `frontend/.env.example` to that new local file and adjust the origin if needed. Only the public backend URL belongs in a `VITE_*` variable—never place Groq or MySQL credentials in the frontend.

FastAPI CORS reads `CORS_ALLOWED_ORIGINS` through the existing settings object. The default development origin is `http://localhost:5173`; an example configuration also permits `http://127.0.0.1:5173`. Origins must be explicit HTTP(S) origins; wildcard CORS is rejected. Add the actual frontend origin to the backend `.env` only if it differs from those defaults.

Start the backend from the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the Vite app:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). The top bar reads backend and database connectivity from `GET /health`; manual refresh calls the same live endpoint. The forecast form posts to `/api/v1/forecast`, disables duplicate submissions while waiting, and renders the returned request ID, outlooks, reasoning, references, and limitations. Reset clears the React state and restores the initial form values.

The frontend tests use mocked health and forecast responses, including synthetic source metadata for UI behavior; no forecast is fabricated in the running application. Run frontend tests and production build from `frontend/`, then the backend suite from the project root:

```powershell
npm test
npm run build
cd ..
python -m pytest -q
```

## Phase boundaries

Phases 1?9 provide PDF ingestion, extraction, retrieval, analysis, forecast generation, MySQL persistence, and FastAPI. Phase 12 adds the React frontend connected to those existing endpoints; no Phase 10/11 scope is inferred from the incomplete assignment brief.
