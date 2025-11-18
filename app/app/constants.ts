// Mapbox token - replace with your own token
export const MAPBOX_TOKEN =
  process.env.NEXT_PUBLIC_MAPBOX_TOKEN ||
  "pk.eyJ1IjoiYW1yaXRrcmlzaG5hbiIsImEiOiJjbTN1Y2xtZ3UwN3NrMmlwd3huYWFld2hnIn0.a9pxRh0JOh5aHJ-A_cPX1Q";

// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Region bounds for Southwestern Ontario
export const REGION_BOUNDS = {
  north: 45.0,
  south: 42.0,
  east: -78.0,
  west: -81.0,
  center: { lat: 43.5, lng: -79.5 },
};
