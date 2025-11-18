import type { StatCardProps } from "../types";

/**
 * Stat card component to display a single statistic with an icon
 */
export function StatCard({ label, value, unit, color, icon: Icon }: StatCardProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400">{label}</span>
        <Icon className="w-4 h-4" style={{ color }} />
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-white">
          {unit ? value.toFixed(1) : value}
        </span>
        {unit && <span className="text-sm text-slate-400">{unit}</span>}
      </div>
    </div>
  );
}
