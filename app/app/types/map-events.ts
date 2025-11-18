import type { MapLayerMouseEvent } from "react-map-gl/mapbox";

/**
 * Type definitions for map events
 */
export interface MapFeatureProperties {
  temperature: number;
  lat: number;
  lon: number;
  forecast_time: string;
}

export type MapMouseEvent = MapLayerMouseEvent & {
  features?: Array<{
    properties: MapFeatureProperties;
  }>;
  point: {
    x: number;
    y: number;
  };
};
