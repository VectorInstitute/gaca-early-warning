import { useMemo } from "react";
import { Delaunay } from "d3-delaunay";
import bboxClip from "@turf/bbox-clip";
import { polygon as turfPolygon } from "@turf/helpers";
import type { Prediction, VoronoiGeoJSON } from "../types";
import { REGION_BOUNDS } from "../constants";

/**
 * Custom hook to create Voronoi polygons for smooth temperature visualization
 * Uses Delaunay triangulation and clips to region boundaries
 */
export function useVoronoiGeoJSON(currentPredictions: Prediction[]): VoronoiGeoJSON {
  return useMemo(() => {
    if (currentPredictions.length === 0) {
      return {
        type: "FeatureCollection" as const,
        features: [],
      };
    }

    // Extract coordinates and temperatures
    const points = currentPredictions.map((p) => [p.lon, p.lat] as [number, number]);
    const temps = currentPredictions.map((p) => p.predicted_temp);

    // Create Delaunay triangulation with tighter bounds to reduce edge artifacts
    const delaunay = Delaunay.from(points);
    const voronoi = delaunay.voronoi([
      REGION_BOUNDS.west - 0.2,
      REGION_BOUNDS.south - 0.2,
      REGION_BOUNDS.east + 0.2,
      REGION_BOUNDS.north + 0.2,
    ]);

    // Define the clipping boundary box
    const bbox: [number, number, number, number] = [
      REGION_BOUNDS.west,
      REGION_BOUNDS.south,
      REGION_BOUNDS.east,
      REGION_BOUNDS.north,
    ];

    // Convert Voronoi cells to GeoJSON polygons and clip to boundary
    const features = [];
    for (let i = 0; i < points.length; i++) {
      const cell = voronoi.cellPolygon(i);
      if (cell) {
        try {
          // Create turf polygon from Voronoi cell
          const cellPolygon = turfPolygon([cell.map(([x, y]) => [x, y])]);

          // Clip to boundary box
          const clipped = bboxClip(cellPolygon, bbox);

          // Only add if the clipped polygon is valid
          if (clipped?.geometry?.coordinates) {
            features.push({
              type: "Feature" as const,
              geometry: {
                type: "Polygon" as const,
                coordinates: clipped.geometry.coordinates as number[][][],
              },
              properties: {
                temperature: temps[i],
                lat: currentPredictions[i].lat,
                lon: currentPredictions[i].lon,
                forecast_time: currentPredictions[i].forecast_time,
              },
            });
          }
        } catch (e) {
          // Skip cells that fail to clip
          console.warn("Failed to clip Voronoi cell:", e);
        }
      }
    }

    return {
      type: "FeatureCollection" as const,
      features,
    };
  }, [currentPredictions]);
}
