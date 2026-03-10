import {
  Wind,
  Droplets,
  Thermometer,
  Compass,
  Gauge,
  Mountain,
  Circle,
  ExternalLink,
  Info,
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
export function ModelInfoPanel({ modelInfo }: ModelInfoPanelProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-6 space-y-5">
      {/* About section */}
      <div>
        <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
          <Info className="w-4 h-4 text-[#E6007E] flex-shrink-0" />
          About This Project
        </h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          This is a research proof-of-concept exploring graph neural networks for
          climate risk assessment and early warning. It uses a GCN-GRU architecture to
          deliver high-resolution, short-range temperature forecasts across Southwestern
          Ontario, with predictions generated hourly from live NOAA URMA reanalysis data
          at roughly 2.5&nbsp;km spatial resolution.
        </p>
        <a
          href="https://neurips.cc/virtual/2025/loc/san-diego/poster/126959"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 mt-3 px-3 py-1.5 rounded-lg bg-[#E6007E] hover:bg-[#cc006e] text-white text-xs font-semibold transition-colors"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          View Technical Paper
        </a>
      </div>

      {/* Model details */}
      <div className="pt-4 border-t border-slate-700/50 space-y-3 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">Architecture</span>
          <span className="text-white font-medium">{modelInfo.model_architecture}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Spatial Nodes</span>
          <span className="text-white font-medium">{modelInfo.num_nodes}</span>
        </div>
      </div>

      {/* Input Features as bullet list */}
      <div className="pt-4 border-t border-slate-700/50">
        <span className="text-slate-400 text-sm block mb-2">Input Features</span>
        <ul className="space-y-1.5">
          {modelInfo.feature_metadata.map((feature) => {
            const Icon = ICON_MAP[feature.icon] || Circle;
            return (
              <li
                key={feature.name}
                className="flex items-center gap-2 text-xs text-slate-300"
              >
                <Icon className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                {feature.label}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
