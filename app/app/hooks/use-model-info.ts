import { useState, useEffect } from "react";
import type { ModelInfo } from "../types";
import { API_BASE_URL } from "../constants";

/**
 * Custom hook to fetch and manage model information
 * Automatically fetches on mount
 */
export function useModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (!response.ok) {
          throw new Error("Failed to fetch model info");
        }
        const data = await response.json();
        setModelInfo(data);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch model info";
        setError(errorMessage);
        console.error("Error fetching model info:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  return { modelInfo, loading, error };
}
