import { useMemo } from "react";
import type { ColorRange, Statistics } from "../types";
import { DEFAULT_COLOR_RANGE, COLOR_RANGE_CONFIG } from "../config/color-scales";

/**
 * Custom hook to calculate dynamic color scale range with padding
 * Always based on spatial statistics
 */
export function useColorRange(spatialStats: Statistics | null): ColorRange {
  return useMemo(() => {
    if (!spatialStats) {
      return DEFAULT_COLOR_RANGE;
    }

    const { min: dataMin, max: dataMax } = spatialStats;
    const range = dataMax - dataMin;
    const padding = Math.max(
      range * COLOR_RANGE_CONFIG.paddingMultiplier,
      COLOR_RANGE_CONFIG.minPadding
    );

    const min = dataMin - padding;
    const max = dataMax + padding;

    // Generate evenly spaced stops for smooth gradient
    const stops: number[] = [];
    for (let i = 0; i <= COLOR_RANGE_CONFIG.numStops - 1; i++) {
      stops.push(min + (max - min) * (i / (COLOR_RANGE_CONFIG.numStops - 1)));
    }

    return { min, max, stops };
  }, [spatialStats]);
}
