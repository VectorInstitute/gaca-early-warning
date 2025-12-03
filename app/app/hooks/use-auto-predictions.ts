import { useState, useEffect, useCallback, useRef } from "react";
import type { Prediction } from "../types";
import { API_BASE_URL } from "../constants";

export interface ForecastStatus {
  scheduler: {
    is_running: boolean;
    scheduler_active: boolean;
    last_run_timestamp: string | null;
    last_data_timestamp: string | null;
    next_scheduled_run: string | null;
  };
  last_forecast: {
    run_timestamp: string;
    earliest_forecast: string;
    latest_forecast: string;
    prediction_count: number;
  } | null;
}

interface UseAutoPredictionsOptions {
  refreshInterval?: number; // Milliseconds between auto-refresh (default: 30 minutes)
  enabled?: boolean; // Enable/disable auto-refresh (default: true)
}

/**
 * Custom hook to automatically fetch latest predictions from BigQuery
 * with smart polling that minimizes BigQuery costs by checking timestamps first
 */
export function useAutoPredictions(options: UseAutoPredictionsOptions = {}) {
  const { refreshInterval = 30 * 60 * 1000, enabled = true } = options;

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [status, setStatus] = useState<ForecastStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<Date | null>(null);

  // Track the last run_timestamp we've loaded to avoid redundant fetches
  const lastLoadedTimestamp = useRef<string | null>(null);

  const checkForNewData = useCallback(async (): Promise<boolean> => {
    try {
      // Lightweight check: only queries MAX(run_timestamp) from BigQuery
      const response = await fetch(`${API_BASE_URL}/forecasts/latest-timestamp`);

      if (!response.ok) {
        console.error("Failed to check timestamp:", response.status);
        return false;
      }

      const data = await response.json();
      const latestTimestamp = data.last_run_timestamp;

      // No data available yet
      if (!latestTimestamp) {
        return false;
      }

      // Check if this is new data
      const isNewData = latestTimestamp !== lastLoadedTimestamp.current;

      if (isNewData) {
        console.log(`New forecast detected: ${latestTimestamp}`);
      }

      return isNewData;
    } catch (err) {
      console.error("Error checking for new data:", err);
      return false;
    }
  }, []);

  const fetchPredictions = useCallback(
    async (force: boolean = false) => {
      try {
        setLoading(true);
        setError(null);

        // Clear predictions when forcing refresh (manual refresh button)
        if (force) {
          setPredictions([]);
        }

        // Check if there's new data available (unless forced)
        if (!force) {
          const hasNewData = await checkForNewData();
          if (!hasNewData && predictions.length > 0) {
            console.log("No new forecast data available, skipping fetch");
            setLoading(false);
            return;
          }
        }

        // Fetch full predictions from BigQuery
        const predictionsResponse = await fetch(`${API_BASE_URL}/forecasts/latest`);

        if (!predictionsResponse.ok) {
          if (predictionsResponse.status === 404) {
            // No predictions yet - this is expected on first run
            setPredictions([]);
            setError("Waiting for first automated forecast run...");
            return;
          }
          throw new Error(`Failed to fetch predictions: ${predictionsResponse.status}`);
        }

        const predictionsData = await predictionsResponse.json();

        // Store the run_timestamp from the first prediction
        if (predictionsData.length > 0 && predictionsData[0].run_timestamp) {
          lastLoadedTimestamp.current = predictionsData[0].run_timestamp;
        }

        setPredictions(predictionsData);
        setLastFetchTime(new Date());
        console.log(`Loaded ${predictionsData.length} predictions`);
      } catch (err) {
        console.error("Error fetching predictions:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch predictions");
      } finally {
        setLoading(false);
      }
    },
    [predictions.length, checkForNewData]
  );

  const fetchStatus = useCallback(async () => {
    try {
      const statusResponse = await fetch(`${API_BASE_URL}/forecasts/status`);

      if (!statusResponse.ok) {
        throw new Error(`Failed to fetch status: ${statusResponse.status}`);
      }

      const statusData = await statusResponse.json();
      setStatus(statusData);
    } catch (err) {
      console.error("Error fetching status:", err);
      // Don't set error state for status failures - predictions are more important
    }
  }, []);

  const refresh = useCallback(async () => {
    await Promise.all([
      fetchPredictions(true), // Force fetch on manual refresh
      fetchStatus(),
    ]);
  }, [fetchPredictions, fetchStatus]);

  const checkAndUpdate = useCallback(async () => {
    await Promise.all([
      fetchPredictions(false), // Smart check (only fetch if new data)
      fetchStatus(),
    ]);
  }, [fetchPredictions, fetchStatus]);

  // Initial fetch on mount
  useEffect(() => {
    if (enabled) {
      refresh();
    }
  }, [enabled]); // Only run on mount or when enabled changes

  // Set up periodic auto-refresh (smart polling - checks timestamp first)
  useEffect(() => {
    if (!enabled || refreshInterval <= 0) {
      return;
    }

    const intervalId = setInterval(() => {
      checkAndUpdate();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [enabled, refreshInterval, checkAndUpdate]);

  return {
    predictions,
    status,
    loading,
    error,
    lastFetchTime,
    refresh, // Manual refresh (force fetch)
  };
}
