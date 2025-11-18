import type { HoverInfo } from "../types";

interface HoverTooltipProps {
  hoverInfo: HoverInfo;
}

/**
 * Hover tooltip displaying temperature and coordinates
 */
export function HoverTooltip({ hoverInfo }: HoverTooltipProps) {
  return (
    <div
      className="absolute bg-slate-900/95 backdrop-blur-md rounded-lg px-3 py-2 border border-slate-700/50 pointer-events-none"
      style={{
        left: hoverInfo.x + 10,
        top: hoverInfo.y + 10,
      }}
    >
      <div className="text-xs font-semibold text-white mb-1">
        {hoverInfo.temperature.toFixed(1)}°C
      </div>
      <div className="text-[10px] text-slate-400">
        {hoverInfo.lat.toFixed(3)}°N, {Math.abs(hoverInfo.lon).toFixed(3)}°W
      </div>
    </div>
  );
}
