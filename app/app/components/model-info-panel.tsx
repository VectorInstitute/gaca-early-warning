import {
  Calendar,
  Wind,
  Droplets,
  Thermometer,
  Compass,
  Gauge,
  Mountain,
  Circle,
} from "lucide-react";
import type { ModelInfo } from "../types";

interface ModelInfoPanelProps {
  modelInfo: ModelInfo;
  selectedHorizon: number;
}

// Icon mapping for feature metadata
const ICON_MAP: Record<string, typeof Wind> = {
  thermometer: Thermometer,
  droplets: Droplets,
  wind: Wind,
  compass: Compass,
  gauge: Gauge,
  mountain: Mountain,
  circle: Circle,
};

/**
 * Model information panel component
 */
export function ModelInfoPanel({ modelInfo, selectedHorizon }: ModelInfoPanelProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-6">
      <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
        <Calendar className="w-5 h-5 text-blue-400" />
        Model Information
      </h3>
      <div className="space-y-4 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">Architecture</span>
          <span className="text-white font-medium">{modelInfo.model_architecture}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Spatial Nodes</span>
          <span className="text-white font-medium">{modelInfo.num_nodes}</span>
        </div>

        {/* Input Features Section */}
        <div className="pt-2 border-t border-slate-700/50">
          <span className="text-slate-400 block mb-3">Input Features</span>
          <div className="grid grid-cols-2 gap-2">
            {modelInfo.feature_metadata.map((feature) => {
              const Icon = ICON_MAP[feature.icon] || Circle;
              return (
                <div
                  key={feature.name}
                  className="flex items-center gap-2 bg-slate-700/30 rounded-lg px-3 py-2"
                >
                  <Icon className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  <span className="text-xs text-slate-300">{feature.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Available Horizons Section */}
        <div className="pt-2 border-t border-slate-700/50">
          <span className="text-slate-400 block mb-2">Available Horizons</span>
          <div className="flex flex-wrap gap-2">
            {modelInfo.prediction_horizons.map((h) => (
              <span
                key={h}
                className={`px-2 py-1 rounded text-xs font-medium ${
                  h === selectedHorizon
                    ? "bg-blue-600 text-white"
                    : "bg-slate-700 text-slate-300"
                }`}
              >
                {h}h
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
