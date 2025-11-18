import { Activity } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { TimeSeriesDataPoint, SelectedNode } from "../types";

interface TemperatureChartProps {
  data: TimeSeriesDataPoint[];
  selectedNode: SelectedNode | null;
  onClearSelection: () => void;
}

/**
 * Temperature time series chart component
 */
export function TemperatureChart({
  data,
  selectedNode,
  onClearSelection,
}: TemperatureChartProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Activity className="w-5 h-5 text-amber-500" />
          {selectedNode ? "Selected Location" : "Regional Average"}
        </h3>
        {selectedNode && (
          <button
            onClick={onClearSelection}
            className="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1"
          >
            <span>Clear</span>
            <span className="text-lg leading-none">×</span>
          </button>
        )}
      </div>
      {selectedNode ? (
        <p className="text-xs text-slate-400 mb-3">
          {selectedNode.lat.toFixed(3)}°N, {Math.abs(selectedNode.lon).toFixed(3)}°W
        </p>
      ) : (
        <p className="text-xs text-slate-400 mb-3 italic">
          Click on any cell to view specific location
        </p>
      )}
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="horizon" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
            }}
            labelStyle={{ color: "#cbd5e1" }}
            formatter={(value: number) => [`${value.toFixed(1)}°C`, "Avg Temperature"]}
          />
          <Area
            type="monotone"
            dataKey="temperature"
            stroke="#f59e0b"
            fill="url(#colorTemp)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
