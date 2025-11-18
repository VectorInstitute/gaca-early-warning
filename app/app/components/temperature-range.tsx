import { Gauge } from "lucide-react";
import type { Statistics, SelectedNode, ColorRange } from "../types";
import { VIRIDIS_COLORS } from "../config/color-scales";

interface TemperatureRangeProps {
  stats: Statistics;
  selectedNode: SelectedNode | null;
  selectedHorizon: number;
  colorRange: ColorRange;
}

/**
 * Temperature range visualization component
 */
export function TemperatureRange({
  stats,
  selectedNode,
  selectedHorizon,
  colorRange: _colorRange,
}: TemperatureRangeProps) {
  // Create gradient string using Viridis color scale
  const viridisGradient = Object.values(VIRIDIS_COLORS).join(", ");

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-6">
      <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
        <Gauge className="w-5 h-5 text-amber-500" />
        Temperature Range
      </h3>
      <p className="text-xs text-slate-400 mb-4">
        {selectedNode
          ? "Across all forecast horizons"
          : `Spatial variation at ${selectedHorizon}h horizon`}
      </p>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Minimum</span>
          <span className="text-white font-semibold">{stats.min.toFixed(1)}°C</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-3">
          <div
            className="h-3 rounded-full"
            style={{
              width: "100%",
              background: `linear-gradient(to right, ${viridisGradient})`,
            }}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Maximum</span>
          <span className="text-white font-semibold">{stats.max.toFixed(1)}°C</span>
        </div>
      </div>
    </div>
  );
}
