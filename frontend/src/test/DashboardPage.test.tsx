import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DashboardPage } from "../pages/DashboardPage";
import { emptyForecastResponse, forecastResponse, healthyResponse } from "./fixtures";

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function mockFetch(options: {
  health?: unknown;
  healthStatus?: number;
  forecast?: unknown;
  forecastStatus?: number;
  networkError?: boolean;
  pendingForecast?: Promise<Response>;
} = {}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/health")) {
      if (options.networkError) throw new TypeError("connection failed");
      return response(options.health ?? healthyResponse, options.healthStatus ?? 200);
    }
    if (options.networkError) throw new TypeError("connection failed");
    if (options.pendingForecast) return options.pendingForecast;
    return response(
      options.forecast ?? forecastResponse,
      options.forecastStatus ?? 200,
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("TCS Financial Intelligence dashboard", () => {
  it("renders the dashboard and a real health check status", async () => {
    mockFetch();
    render(<DashboardPage />);
    expect(screen.getByText(/financial intelligence workspace/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /your next-quarter outlook will appear here/i })).toBeInTheDocument();
    expect(await screen.findByText("Backend online")).toBeInTheDocument();
    expect(screen.getByText("Database connected")).toBeInTheDocument();
  });

  it("validates blank company and question without submitting", async () => {
    const fetchMock = mockFetch();
    render(<DashboardPage />);
    const company = screen.getByRole("textbox", { name: "Company" });
    const question = screen.getByRole("textbox", { name: "Forecast question" });
    fireEvent.change(company, { target: { value: " " } });
    fireEvent.change(question, { target: { value: " " } });
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter both a company and a forecast question.");
    expect(fetchMock).not.toHaveBeenCalledWith(expect.stringContaining("/api/v1/forecast"), expect.anything());
  });

  it("submits the request and renders the returned forecast and request ID", async () => {
    const fetchMock = mockFetch();
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("heading", { name: "Example Corp outlook" })).toBeInTheDocument();
    expect(screen.getByText(forecastResponse.request_id)).toBeInTheDocument();
    expect(screen.getByText("A cautious qualitative outlook is supported by the returned sample evidence.")).toBeInTheDocument();
    expect(screen.getAllByText("synthetic-example-report.pdf")).toHaveLength(2);
    const postCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/v1/forecast"));
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toEqual({
      company: "TCS",
      question: "Give a qualitative outlook for the upcoming quarter",
    });
  });

  it("shows a loading state and disables duplicate submissions", async () => {
    let finish!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => { finish = resolve; });
    const fetchMock = mockFetch({ pendingForecast: pending });
    render(<DashboardPage />);
    const button = screen.getByRole("button", { name: /generate forecast/i });
    await userEvent.click(button);
    expect(await screen.findByRole("button", { name: /generating forecast/i })).toBeDisabled();
    expect(screen.getByLabelText("Forecast loading")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /generating forecast/i }));
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/api/v1/forecast"))).toHaveLength(1);
    finish(response(forecastResponse));
    expect(await screen.findByRole("heading", { name: "Example Corp outlook" })).toBeInTheDocument();
  });

  it("shows an API validation error", async () => {
    mockFetch({ forecast: { error: "Invalid request body.", issues: [] }, forecastStatus: 422 });
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid request body.");
  });

  it("shows a safe server or database error", async () => {
    mockFetch({ forecast: { error: "Database is unavailable; the request could not be recorded." }, forecastStatus: 503 });
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Database is unavailable; the request could not be recorded.");
  });

  it("handles a generic server error without leaking backend internals", async () => {
    mockFetch({ forecast: { error: "An internal error prevented forecast processing." }, forecastStatus: 500 });
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("An internal error prevented forecast processing.");
    expect(screen.queryByText(/password|api[_ ]key|traceback/i)).not.toBeInTheDocument();
  });

  it("shows a clear message when the network fails", async () => {
    mockFetch({ networkError: true });
    render(<DashboardPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Backend connection unavailable");
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(screen.getAllByText("Could not reach the forecasting service. Check that the backend is running and try again.").length).toBeGreaterThan(0);
  });

  it("does not show backend connected without a successful health response", async () => {
    mockFetch({ healthStatus: 503, health: { error: "Unavailable" } });
    render(<DashboardPage />);
    expect(await screen.findByText("Backend offline")).toBeInTheDocument();
    expect(screen.queryByText("Backend online")).not.toBeInTheDocument();
  });

  it("shows the actual database disconnected state from health", async () => {
    mockFetch({ health: { status: "degraded", application: "TCS Financial Forecasting Agent", database_status: "unavailable" } });
    render(<DashboardPage />);
    expect(await screen.findByText("Backend degraded")).toBeInTheDocument();
    expect(screen.getByText("Database unavailable")).toBeInTheDocument();
  });

  it("renders only returned sources and gracefully handles missing references", async () => {
    mockFetch({ forecast: emptyForecastResponse });
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("heading", { name: "Example Corp outlook" })).toBeInTheDocument();
    expect(screen.getByText("No source references returned")).toBeInTheDocument();
    expect(screen.queryByText("synthetic-example-report.pdf")).not.toBeInTheDocument();
  });

  it("makes unavailable operating margin explicit without inventing evidence", async () => {
    mockFetch();
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    const marginHeading = await screen.findByRole("heading", { name: "Operating margin" });
    const card = marginHeading.closest("article");
    expect(card).toHaveTextContent("Evidence unavailable");
    expect(card).toHaveTextContent("no supporting source references");
    expect(card).not.toHaveTextContent("24%");
  });

  it("renders the empty state and does not fabricate a forecast", () => {
    mockFetch();
    render(<DashboardPage />);
    expect(screen.getByText(/No forecast is shown until the API returns one/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Example Corp outlook" })).not.toBeInTheDocument();
  });

  it("rejects an empty or malformed forecast response", async () => {
    mockFetch({ forecast: { request_id: "bad", status: "completed", forecast: {} } });
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByText("The backend returned a forecast that did not match its published schema.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Example Corp outlook" })).not.toBeInTheDocument();
  });

  it("clears the most recent forecast when the user resets the workspace", async () => {
    mockFetch();
    render(<DashboardPage />);
    await userEvent.click(screen.getByRole("button", { name: /generate forecast/i }));
    expect(await screen.findByRole("heading", { name: "Example Corp outlook" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reset workspace" }));
    expect(screen.getByRole("heading", { name: /your next-quarter outlook will appear here/i })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Company" })).toHaveValue("TCS");
  });
});
