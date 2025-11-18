import { useState, useEffect, useCallback, useMemo } from "react";
import { Play, Pause } from "lucide-react";
import type { ModelInfo } from "../types";

interface HorizonSliderProps {
  selectedHorizon: number;
  onHorizonChange: (horizon: number) => void;
  modelInfo: ModelInfo | null;
  disabled?: boolean;
}

/**
 * Horizon slider with play/pause functionality for animating through forecast horizons
 */
export function HorizonSlider({
  selectedHorizon,
  onHorizonChange,
  modelInfo,
  disabled = false,
}: HorizonSliderProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const horizons = useMemo(
    () => modelInfo?.prediction_horizons || [1, 6, 12, 18, 24, 36, 48],
    [modelInfo]
  );
  const currentIndex = horizons.indexOf(selectedHorizon);

  // Auto-advance through horizons when playing
  useEffect(() => {
    if (!isPlaying || disabled) return;

    const interval = setInterval(() => {
      const currentIdx = horizons.indexOf(selectedHorizon);
      const nextIdx = (currentIdx + 1) % horizons.length;
      onHorizonChange(horizons[nextIdx]);

      // Stop at the end
      if (nextIdx === 0) {
        setIsPlaying(false);
      }
    }, 800); // 800ms between frames for smooth animation

    return () => clearInterval(interval);
  }, [isPlaying, selectedHorizon, horizons, onHorizonChange, disabled]);

  const handlePlayPause = useCallback(() => {
    if (disabled) return;
    setIsPlaying((prev) => !prev);
  }, [disabled]);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled) return;
      const index = parseInt(e.target.value, 10);
      onHorizonChange(horizons[index]);
      setIsPlaying(false); // Stop playing when manually adjusting
    },
    [horizons, onHorizonChange, disabled]
  );

  return (
    <div className="flex-1">
      <div className="bg-slate-900/95 backdrop-blur-md rounded-xl border border-slate-700/50 px-4 py-2.5 shadow-lg">
        <div className="flex items-center gap-3">
          {/* Play/Pause Button */}
          <button
            onClick={handlePlayPause}
            disabled={disabled}
            className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-r from-[#EB088A] to-[#313CFF] hover:from-[#d00780] hover:to-[#2831e6] disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white flex items-center justify-center transition-all shadow-lg hover:shadow-xl"
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4 ml-0.5" />
            )}
          </button>

          {/* Slider Container */}
          <div className="flex-1 flex items-center gap-3">
            <span className="text-xs font-medium text-slate-400 flex-shrink-0">
              Forecast Horizon
            </span>

            {/* Slider with Markers */}
            <div className="relative flex-1 pb-5">
              <input
                type="range"
                min={0}
                max={horizons.length - 1}
                step={1}
                value={currentIndex}
                onChange={handleSliderChange}
                disabled={disabled}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed slider-thumb"
                style={{
                  background: `linear-gradient(to right,
                    rgb(59, 130, 246) 0%,
                    rgb(59, 130, 246) ${(currentIndex / (horizons.length - 1)) * 100}%,
                    rgb(51, 65, 85) ${(currentIndex / (horizons.length - 1)) * 100}%,
                    rgb(51, 65, 85) 100%)`,
                }}
              />

              {/* Horizon Markers with Clickable Zones */}
              <div className="absolute top-2.5 left-0 right-0">
                {horizons.map((h, idx) => {
                  const percentage = (idx / (horizons.length - 1)) * 100;
                  // Adjust for 20px thumb width to align markers with slider thumb center
                  const thumbOffset = (0.5 - percentage / 100) * 20;
                  return (
                    <button
                      key={h}
                      onClick={() => {
                        if (!disabled) {
                          onHorizonChange(h);
                          setIsPlaying(false);
                        }
                      }}
                      disabled={disabled}
                      className="absolute flex flex-col items-center -translate-x-1/2 cursor-pointer hover:scale-110 transition-transform disabled:cursor-not-allowed group"
                      style={{ left: `calc(${percentage}% + ${thumbOffset}px)` }}
                      aria-label={`Select ${h} hour forecast`}
                    >
                      {/* Expanded clickable area */}
                      <div className="absolute -inset-4" />
                      <div
                        className={`w-0.5 h-1.5 rounded-full transition-colors ${
                          idx <= currentIndex ? "bg-blue-400" : "bg-slate-600"
                        } group-hover:bg-blue-300`}
                      />
                      <span
                        className={`text-[10px] font-medium mt-0.5 transition-colors whitespace-nowrap ${
                          idx === currentIndex ? "text-blue-400" : "text-slate-500"
                        } group-hover:text-blue-300`}
                      >
                        {h}h
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
