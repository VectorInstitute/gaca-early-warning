import { useState, useCallback } from "react";
import type { Prediction } from "../types";
import { API_BASE_URL } from "../constants";

interface UsePredictionsOptions {
  numHours?: number;
}

/**
 * Custom hook to fetch and manage temperature predictions
 * Returns predictions data and a manual fetch function
 */
export function usePredictions(options: UsePredictionsOptions = {}) {
  const { numHours = 24 } = options;

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ num_hours: numHours }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch predictions");
      }

      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to load predictions";
      setError(errorMessage);
      console.error("Error fetching predictions:", err);
    } finally {
      setLoading(false);
    }
  }, [numHours]);

  return {
    predictions,
    loading,
    error,
    fetchPredictions,
  };
}
