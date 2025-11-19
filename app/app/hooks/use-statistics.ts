import { useMemo } from "react";
import type { Prediction, Statistics, SelectedNode } from "../types";

interface UseStatisticsOptions {
  predictions: Prediction[];
  selectedHorizon: number;
  selectedNode: SelectedNode | null;
}

/**
 * Custom hook to calculate statistics for predictions
 * Returns spatial statistics (current horizon), global statistics (all horizons),
 * and display statistics (either temporal for selected node or spatial for current horizon)
 */
export function useStatistics({
  predictions,
  selectedHorizon,
  selectedNode,
}: UseStatisticsOptions) {
  // Filter predictions for selected horizon
  const currentPredictions = useMemo(
    () => predictions.filter((p) => p.horizon_hours === selectedHorizon),
    [predictions, selectedHorizon]
  );

  // Calculate global statistics across all horizons (for consistent color scaling)
  const globalStats = useMemo<Statistics | null>(() => {
    if (predictions.length === 0) return null;

    const temps = predictions.map((p) => p.predicted_temp);
    const sum = temps.reduce((acc, temp) => acc + temp, 0);

    return {
      mean: sum / predictions.length,
      min: Math.min(...temps),
      max: Math.max(...temps),
      count: predictions.length,
    };
  }, [predictions]);

  // Calculate spatial statistics for current horizon (for heatmap coloring)
  const spatialStats = useMemo<Statistics | null>(() => {
    if (currentPredictions.length === 0) return null;

    const temps = currentPredictions.map((p) => p.predicted_temp);
    const sum = temps.reduce((acc, temp) => acc + temp, 0);

    return {
      mean: sum / currentPredictions.length,
      min: Math.min(...temps),
      max: Math.max(...temps),
      count: currentPredictions.length,
    };
  }, [currentPredictions]);

  // Calculate display statistics - either temporal (for selected node) or spatial
  const displayStats = useMemo<Statistics | null>(() => {
    if (selectedNode) {
      // Show temporal statistics for selected location across all horizons
      const nodePredictions = predictions.filter(
        (p) =>
          Math.abs(p.lat - selectedNode.lat) < 0.001 &&
          Math.abs(p.lon - selectedNode.lon) < 0.001
      );

      if (nodePredictions.length === 0) return spatialStats;

      const temps = nodePredictions.map((p) => p.predicted_temp);
      const sum = temps.reduce((acc, temp) => acc + temp, 0);

      return {
        mean: sum / nodePredictions.length,
        min: Math.min(...temps),
        max: Math.max(...temps),
        count: nodePredictions.length,
      };
    }

    // Show spatial statistics for current horizon
    return spatialStats;
  }, [spatialStats, predictions, selectedNode]);

  return {
    currentPredictions,
    spatialStats,
    globalStats,
    displayStats,
  };
}
