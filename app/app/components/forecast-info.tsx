interface ForecastInfoProps {
  forecastTime: string;
  selectedHorizon: number;
}

/**
 * Forecast info badge displaying prediction valid time
 * Note: forecastTime from backend is already the valid time (base_time + horizon)
 */
export function ForecastInfo({ forecastTime, selectedHorizon }: ForecastInfoProps) {
  // forecastTime is already the valid time (base time + horizon hours)
  const validTime = new Date(forecastTime);
  // Calculate base time by subtracting the horizon
  const baseTime = new Date(validTime.getTime() - selectedHorizon * 60 * 60 * 1000);

  const formattedValidTime = validTime.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const formattedBaseTime = baseTime.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="flex-shrink-0 bg-slate-900/95 backdrop-blur-md rounded-xl border border-slate-700/50 px-4 py-2.5 shadow-lg">
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold">
          Prediction Valid For
        </span>
        <span className="text-sm font-bold text-white whitespace-nowrap">
          {formattedValidTime}
        </span>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <span>Data: {formattedBaseTime}</span>
          <span className="text-blue-400 font-medium">(+{selectedHorizon}h)</span>
        </div>
      </div>
    </div>
  );
}
