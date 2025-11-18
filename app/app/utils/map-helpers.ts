import type { ColorRange } from "../types";
import { VIRIDIS_COLORS } from "../config/color-scales";
import { REGION_BOUNDS } from "../constants";

/**
 * Generate Mapbox paint expression for temperature fill color using Viridis scale
 */
export function getTemperatureFillColor(colorRange: ColorRange): any {
  return [
    "interpolate",
    ["linear"],
    ["get", "temperature"],
    colorRange.stops[0],
    VIRIDIS_COLORS[0],
    colorRange.stops[1],
    VIRIDIS_COLORS[1],
    colorRange.stops[2],
    VIRIDIS_COLORS[2],
    colorRange.stops[3],
    VIRIDIS_COLORS[3],
    colorRange.stops[4],
    VIRIDIS_COLORS[4],
    colorRange.stops[5],
    VIRIDIS_COLORS[5],
    colorRange.stops[6],
    VIRIDIS_COLORS[6],
    colorRange.stops[7],
    VIRIDIS_COLORS[7],
    colorRange.stops[8],
    VIRIDIS_COLORS[8],
    colorRange.stops[9],
    VIRIDIS_COLORS[9],
    colorRange.stops[10],
    VIRIDIS_COLORS[10],
  ];
}

/**
 * Create region boundary mask geometry
 */
export function getRegionMaskGeometry() {
  return {
    type: "Feature" as const,
    properties: {},
    geometry: {
      type: "Polygon" as const,
      coordinates: [
        [
          [REGION_BOUNDS.west, REGION_BOUNDS.north],
          [REGION_BOUNDS.east, REGION_BOUNDS.north],
          [REGION_BOUNDS.east, REGION_BOUNDS.south],
          [REGION_BOUNDS.west, REGION_BOUNDS.south],
          [REGION_BOUNDS.west, REGION_BOUNDS.north],
        ],
      ],
    },
  };
}

/**
 * Check if two nodes are approximately equal (within tolerance)
 */
export function nodesEqual(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
  tolerance = 0.001
): boolean {
  return Math.abs(lat1 - lat2) < tolerance && Math.abs(lon1 - lon2) < tolerance;
}
