import { useCallback, useEffect, useState } from "react";
import { ApiClientError, fetchHealth } from "../services/api";
import type { HealthResponse } from "../types/api";

export type HealthState = "checking" | "connected" | "degraded" | "offline";

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [state, setState] = useState<HealthState>("checking");
  const [message, setMessage] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const current = await fetchHealth();
      setHealth(current);
      setState(current.status === "healthy" ? "connected" : "degraded");
      setMessage(null);
    } catch (error) {
      setHealth(null);
      setState("offline");
      setMessage(error instanceof ApiClientError ? error.message : "Backend health check failed.");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { health, state, message, refreshing, refresh };
}
