"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Map, { Source, Layer, NavigationControl } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import { MapPin, TrendingUp, ChevronDown } from "lucide-react";
import { MAPBOX_TOKEN, REGION_BOUNDS } from "./constants";
import type { MapMouseEvent } from "react-map-gl/mapbox";
import type { HoverInfo, SelectedNode } from "./types";
import {
  useModelInfo,
  useWebSocketPredictions,
  useStatistics,
  useColorRange,
  useVoronoiGeoJSON,
  useTimeSeriesData,
} from "./hooks";
import type { ProgressUpdate } from "./hooks/use-websocket-predictions";
import {
  Header,
  CollapsedSidebar,
  StatCard,
  MapControls,
  MapLegend,
  ForecastInfo,
  HoverTooltip,
  LoadingOverlay,
  ErrorOverlay,
  EmptyState,
  TemperatureChart,
  TemperatureRange,
  ModelInfoPanel,
  LoadingSkeleton,
  EmptyStats,
  HorizonSlider,
} from "./components";
import { MAP_STYLES, MAP_LAYER_STYLES } from "./config/color-scales";
import {
  getTemperatureFillColor,
  getRegionMaskGeometry,
  nodesEqual,
} from "./utils/map-helpers";

export default function Home() {
  // State management
  const [selectedHorizon, setSelectedHorizon] = useState(24);
  const [_mapLoaded, setMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [currentProgress, setCurrentProgress] = useState<ProgressUpdate | null>(null);

  // Data fetching hooks
  const { modelInfo } = useModelInfo();
  const { predictions, loading, error, fetchPredictions } = useWebSocketPredictions({
    onProgress: (update) => setCurrentProgress(update),
  });

  // Data processing hooks
  const { currentPredictions, globalStats, displayStats } = useStatistics({
    predictions,
    selectedHorizon,
    selectedNode,
  });
  const colorRange = useColorRange(globalStats);
  const voronoiGeoJSON = useVoronoiGeoJSON(currentPredictions);
  const timeSeriesData = useTimeSeriesData({ predictions, modelInfo, selectedNode });

  // Event handlers
  const handleMouseMove = (e: MapMouseEvent) => {
    const feature = e.features?.[0];
    if (feature?.properties) {
      setHoverInfo({
        temperature: feature.properties.temperature,
        lat: feature.properties.lat,
        lon: feature.properties.lon,
        x: e.point.x,
        y: e.point.y,
      });
    }
  };

  const handleClick = (e: MapMouseEvent) => {
    const feature = e.features?.[0];
    if (feature?.properties) {
      setSelectedNode({
        lat: feature.properties.lat,
        lon: feature.properties.lon,
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex">
      <CollapsedSidebar />

      <div className="flex-1 ml-16">
        <Header />

        <div className="max-w-[1920px] mx-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-[calc(100vh-120px)]">
            {/* Left Panel - Map & Controls */}
            <motion.div
              className="lg:col-span-2 flex flex-col gap-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {/* Map Visualization */}
              <div className="flex-1 bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden relative">
                {loading && <LoadingOverlay progress={currentProgress} />}
                {error && <ErrorOverlay error={error} />}

                <Map
                  initialViewState={{
                    longitude: REGION_BOUNDS.center.lng,
                    latitude: REGION_BOUNDS.center.lat,
                    zoom: 7,
                  }}
                  style={{ width: "100%", height: "100%" }}
                  mapStyle={MAP_STYLES.dark}
                  mapboxAccessToken={MAPBOX_TOKEN}
                  attributionControl={false}
                  onLoad={() => setMapLoaded(true)}
                  interactiveLayerIds={["temperature-fill"]}
                  cursor={hoverInfo ? "pointer" : "grab"}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={() => setHoverInfo(null)}
                  onClick={handleClick}
                >
                  <NavigationControl position="top-right" />

                  {/* Temperature Voronoi polygons */}
                  {voronoiGeoJSON.features.length > 0 && (
                    <>
                      <Source
                        id="temperature-voronoi"
                        type="geojson"
                        data={voronoiGeoJSON}
                      >
                        <Layer
                          id="temperature-fill"
                          type="fill"
                          paint={{
                            "fill-color": getTemperatureFillColor(colorRange),
                            ...MAP_LAYER_STYLES.temperatureFill,
                          }}
                        />
                        <Layer
                          id="temperature-outline"
                          type="line"
                          paint={MAP_LAYER_STYLES.temperatureOutline}
                        />
                      </Source>

                      {/* Region boundary mask */}
                      <Source
                        id="region-mask"
                        type="geojson"
                        data={getRegionMaskGeometry()}
                      >
                        <Layer
                          id="region-mask-border"
                          type="line"
                          paint={MAP_LAYER_STYLES.regionBorder}
                        />
                      </Source>

                      {/* Highlight selected cell */}
                      {selectedNode && (
                        <Source
                          id="selected-cell"
                          type="geojson"
                          data={{
                            type: "FeatureCollection",
                            features: voronoiGeoJSON.features.filter((f) =>
                              nodesEqual(
                                f.properties.lat,
                                f.properties.lon,
                                selectedNode.lat,
                                selectedNode.lon
                              )
                            ),
                          }}
                        >
                          <Layer
                            id="selected-cell-outline"
                            type="line"
                            paint={MAP_LAYER_STYLES.selectedCellOutline}
                          />
                        </Source>
                      )}
                    </>
                  )}

                  {/* Empty state */}
                  {voronoiGeoJSON.features.length === 0 && !loading && !error && (
                    <EmptyState />
                  )}

                  {/* Run Forecast Button */}
                  <MapControls onRunForecast={fetchPredictions} loading={loading} />

                  {/* Legend */}
                  {voronoiGeoJSON.features.length > 0 && (
                    <MapLegend colorRange={colorRange} />
                  )}

                  {/* Hover Tooltip */}
                  {hoverInfo && <HoverTooltip hoverInfo={hoverInfo} />}
                </Map>
              </div>

              {/* Controls Below Map */}
              {voronoiGeoJSON.features.length > 0 && (
                <div className="flex items-center gap-4">
                  <ForecastInfo
                    forecastTime={currentPredictions[0].forecast_time}
                    selectedHorizon={selectedHorizon}
                  />
                  <HorizonSlider
                    selectedHorizon={selectedHorizon}
                    onHorizonChange={setSelectedHorizon}
                    modelInfo={modelInfo}
                    disabled={loading}
                  />
                </div>
              )}
            </motion.div>

            {/* Right Panel - Statistics */}
            <motion.div
              className="flex flex-col gap-4 overflow-y-auto"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              {/* Stats Cards */}
              {displayStats && (
                <motion.div
                  key={selectedHorizon}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-col gap-4"
                >
                  <div className="grid grid-cols-2 gap-4">
                    <StatCard
                      label="Mean"
                      value={displayStats.mean}
                      unit="°C"
                      color="#f59e0b"
                      icon={TrendingUp}
                    />
                    <StatCard
                      label="Min"
                      value={displayStats.min}
                      unit="°C"
                      color="#3b82f6"
                      icon={ChevronDown}
                    />
                    <StatCard
                      label="Max"
                      value={displayStats.max}
                      unit="°C"
                      color="#ef4444"
                      icon={TrendingUp}
                    />
                    <StatCard
                      label={selectedNode ? "Horizons" : "Nodes"}
                      value={displayStats.count}
                      unit=""
                      color="#10b981"
                      icon={MapPin}
                    />
                  </div>

                  {!selectedNode && (
                    <p className="text-xs text-slate-400 mt-2 text-center">
                      Statistics for {selectedHorizon}h horizon across all nodes
                    </p>
                  )}
                  {selectedNode && (
                    <p className="text-xs text-slate-400 mt-2 text-center">
                      Statistics for selected location across all horizons
                    </p>
                  )}

                  {/* Time Series Chart */}
                  {timeSeriesData.length > 0 && (
                    <TemperatureChart
                      data={timeSeriesData}
                      selectedNode={selectedNode}
                      onClearSelection={() => setSelectedNode(null)}
                    />
                  )}

                  {/* Temperature Range */}
                  {displayStats && (
                    <TemperatureRange
                      stats={displayStats}
                      selectedNode={selectedNode}
                      selectedHorizon={selectedHorizon}
                      colorRange={colorRange}
                    />
                  )}

                  {/* Model Info */}
                  {modelInfo && (
                    <ModelInfoPanel
                      modelInfo={modelInfo}
                      selectedHorizon={selectedHorizon}
                    />
                  )}
                </motion.div>
              )}

              {/* Loading skeleton */}
              {loading && <LoadingSkeleton />}

              {/* Empty state */}
              {!displayStats && !loading && <EmptyStats />}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
