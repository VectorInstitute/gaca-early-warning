import { useState, useCallback, useRef, useEffect } from "react";
import type { Prediction } from "../types";

export interface ProgressUpdate {
  step: string;
  status: "in_progress" | "completed";
  timestamp: string;
}

interface UsePredictionsOptions {
  onProgress?: (update: ProgressUpdate) => void;
}

const WS_URL = "ws://localhost:8000/ws/predict";

/**
 * Custom hook to manage WebSocket-based predictions with real-time progress
 */
export function useWebSocketPredictions(options: UsePredictionsOptions = {}) {
  const { onProgress } = options;

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const fetchPredictions = useCallback(() => {
    setLoading(true);
    setError(null);
    setPredictions([]);

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Create new WebSocket connection
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      // WebSocket connected successfully
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case "progress":
            if (onProgress) {
              onProgress({
                step: message.step,
                status: message.status,
                timestamp: message.timestamp,
              });
            }
            break;

          case "complete":
            setPredictions(message.data);
            setLoading(false);
            ws.close();
            break;

          case "error":
            setError(message.message || "Inference failed");
            setLoading(false);
            ws.close();
            break;

          default:
            console.warn("Unknown message type:", message.type);
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
        setError("Failed to parse server response");
        setLoading(false);
        ws.close();
      }
    };

    ws.onerror = (event) => {
      console.error("WebSocket error:", event);
      setError("Connection error");
      setLoading(false);
    };

    ws.onclose = (event) => {
      if (loading && !error) {
        console.warn("WebSocket closed unexpectedly:", event.code, event.reason);
        setError("Connection closed unexpectedly");
        setLoading(false);
      }
    };
  }, [loading, error, onProgress]);

  return {
    predictions,
    loading,
    error,
    fetchPredictions,
  };
}
