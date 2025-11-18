import type { LucideIcon } from "lucide-react";

export interface Prediction {
  forecast_time: string;
  horizon_hours: number;
  lat: number;
  lon: number;
  predicted_temp: number;
}

export interface FeatureMetadata {
  name: string;
  label: string;
  icon: string;
}

export interface ModelInfo {
  model_architecture: string;
  num_nodes: number;
  input_features: string[];
  feature_metadata: FeatureMetadata[];
  prediction_horizons: number[];
  region: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  status: string;
}

export interface Statistics {
  mean: number;
  min: number;
  max: number;
  count: number;
}

export interface HoverInfo {
  temperature: number;
  lat: number;
  lon: number;
  x: number;
  y: number;
}

export interface SelectedNode {
  lat: number;
  lon: number;
}

export interface ColorRange {
  min: number;
  max: number;
  stops: number[];
}

export interface VoronoiFeature {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: {
    temperature: number;
    lat: number;
    lon: number;
    forecast_time: string;
  };
}

export interface VoronoiGeoJSON {
  type: "FeatureCollection";
  features: VoronoiFeature[];
}

export interface TimeSeriesDataPoint {
  horizon: string;
  temperature: number;
}

export interface StatCardProps {
  label: string;
  value: number;
  unit: string;
  color: string;
  icon: LucideIcon;
}

export type { MapFeatureProperties } from "./types/map-events";
