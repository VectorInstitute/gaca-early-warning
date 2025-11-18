import { Cloud } from "lucide-react";

interface ErrorOverlayProps {
  error: string;
}

/**
 * Error overlay for the map
 */
export function ErrorOverlay({ error }: ErrorOverlayProps) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm z-20">
      <div className="text-center max-w-md">
        <Cloud className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <p className="text-slate-300 mb-2">Failed to load data</p>
        <p className="text-sm text-slate-400">{error}</p>
      </div>
    </div>
  );
}
