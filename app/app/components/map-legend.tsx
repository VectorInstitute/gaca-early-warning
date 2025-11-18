import type { ColorRange } from "../types";
import { VIRIDIS_LEGEND_COLORS } from "../config/color-scales";

interface MapLegendProps {
  colorRange: ColorRange;
}

/**
 * Map legend component displaying the dynamic Viridis color scale
 */
export function MapLegend({ colorRange }: MapLegendProps) {
  return (
    <div className="absolute top-4 right-16 bg-slate-900/90 backdrop-blur-md rounded-lg p-2 border border-slate-700/50">
      <div className="text-xs text-slate-400 mb-1.5 font-medium">Temperature (°C)</div>
      <div className="flex items-center gap-1.5">
        {VIRIDIS_LEGEND_COLORS.map(({ color, index }) => {
          const value = colorRange.stops[index];
          return (
            <div key={index} className="flex flex-col gap-0.5 items-center">
              <div className="w-3 h-3 rounded-sm" style={{ background: color }} />
              <span className="text-[10px] text-slate-400">{value.toFixed(0)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
