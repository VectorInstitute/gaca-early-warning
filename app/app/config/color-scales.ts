/**
 * Viridis color scale for temperature visualization
 * Perceptually uniform sequential colormap
 */
export const VIRIDIS_COLORS = {
  0: "#440154", // Dark purple
  1: "#482475",
  2: "#414487",
  3: "#355f8d",
  4: "#2a788e", // Teal
  5: "#21918c",
  6: "#22a884", // Green
  7: "#42b97c",
  8: "#7ad151", // Yellow-green
  9: "#bddf26",
  10: "#fde724", // Bright yellow
} as const;

export const VIRIDIS_LEGEND_COLORS = [
  { color: "#440154", index: 0 },
  { color: "#414487", index: 2 },
  { color: "#2a788e", index: 4 },
  { color: "#22a884", index: 6 },
  { color: "#7ad151", index: 8 },
  { color: "#fde724", index: 10 },
] as const;

/**
 * Map style configurations
 */
export const MAP_STYLES = {
  dark: "mapbox://styles/mapbox/dark-v11",
  light: "mapbox://styles/mapbox/light-v11",
  satellite: "mapbox://styles/mapbox/satellite-streets-v12",
} as const;

/**
 * Map layer paint configurations
 */
export const MAP_LAYER_STYLES = {
  temperatureFill: {
    "fill-opacity": 0.75,
  },
  temperatureOutline: {
    "line-color": "#ffffff",
    "line-width": 0.3,
    "line-opacity": 0.15,
  },
  regionBorder: {
    "line-color": "#64748b",
    "line-width": 2,
    "line-opacity": 0.4,
  },
  selectedCellOutline: {
    "line-color": "#fbbf24",
    "line-width": 3,
    "line-opacity": 1,
  },
} as const;

/**
 * Default color range settings
 */
export const DEFAULT_COLOR_RANGE = {
  min: -30,
  max: 30,
  stops: [-30, -20, -10, -5, 0, 5, 10, 15, 20, 25, 30],
} as const;

/**
 * Color range calculation parameters
 */
export const COLOR_RANGE_CONFIG = {
  numStops: 11,
  paddingMultiplier: 0.15,
  minPadding: 2, // Minimum padding in degrees Celsius
} as const;
