import { useMemo } from "react";
import type {
  Prediction,
  ModelInfo,
  SelectedNode,
  TimeSeriesDataPoint,
} from "../types";

interface UseTimeSeriesDataOptions {
  predictions: Prediction[];
  modelInfo: ModelInfo | null;
  selectedNode: SelectedNode | null;
}

/**
 * Custom hook to create time series data for chart
 * Shows either selected node data or regional average
 */
export function useTimeSeriesData({
  predictions,
  modelInfo,
  selectedNode,
}: UseTimeSeriesDataOptions): TimeSeriesDataPoint[] {
  return useMemo(() => {
    if (!modelInfo?.prediction_horizons || predictions.length === 0) {
      return [];
    }

    return modelInfo.prediction_horizons
      .map((horizon) => {
        const horizonPreds = predictions.filter((p) => p.horizon_hours === horizon);

        if (horizonPreds.length === 0) {
          return null;
        }

        let temp: number;

        if (selectedNode) {
          // Find the prediction for the selected node
          const nodePred = horizonPreds.find(
            (p) =>
              Math.abs(p.lat - selectedNode.lat) < 0.001 &&
              Math.abs(p.lon - selectedNode.lon) < 0.001
          );
          temp = nodePred ? nodePred.predicted_temp : horizonPreds[0].predicted_temp;
        } else {
          // Average across all nodes
          const sum = horizonPreds.reduce((acc, p) => acc + p.predicted_temp, 0);
          temp = sum / horizonPreds.length;
        }

        return {
          horizon: `${horizon}h`,
          temperature: temp,
        };
      })
      .filter((d): d is TimeSeriesDataPoint => d !== null);
  }, [modelInfo, predictions, selectedNode]);
}
