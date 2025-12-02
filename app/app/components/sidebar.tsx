"use client";

import { Home, BarChart3, ScrollText } from "lucide-react";
import { motion } from "framer-motion";
import { usePathname, useRouter } from "next/navigation";
import Image from "next/image";
import { APP_VERSION, APP_NAME } from "../lib/version";

/**
 * Collapsed sidebar navigation component
 */
export function CollapsedSidebar() {
  const pathname = usePathname();
  const router = useRouter();

  const navItems = [
    {
      icon: Home,
      label: "Forecast",
      path: "/",
      gradient: "from-[#EB088A] to-pink-600",
    },
    {
      icon: BarChart3,
      label: "Evaluation",
      path: "/evaluation",
      gradient: "from-[#313CFF] to-blue-600",
    },
    {
      icon: ScrollText,
      label: "Logs",
      path: "/logs",
      gradient: "from-emerald-500 to-teal-600",
    },
  ];

  return (
    <motion.aside
      initial={{ x: -64, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="w-16 flex-shrink-0 bg-slate-900/80 backdrop-blur-md border-r border-slate-700/50 flex flex-col fixed left-0 top-0 bottom-0 z-50"
    >
      {/* Logo Section */}
      <div className="flex flex-col items-center pt-4 pb-3 border-b border-slate-700/30">
        <div className="w-10 h-10 flex items-center justify-center bg-white/90 rounded-md shadow-sm">
          <Image
            src="/vector-logo.webp"
            alt="Vector Institute"
            width={40}
            height={10}
            priority
            className="w-8 h-auto"
          />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col items-center pt-4 gap-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.path;

          return (
            <button
              key={item.path}
              onClick={() => router.push(item.path)}
              className={`
                relative w-12 h-12 flex items-center justify-center rounded-xl
                transition-all duration-200 group
                ${isActive ? "bg-slate-800 shadow-lg" : "hover:bg-slate-800/60"}
              `}
              title={item.label}
              aria-label={item.label}
            >
              {/* Active Indicator */}
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 rounded-xl"
                  style={{
                    background: `linear-gradient(135deg, rgba(235, 8, 138, 0.1), rgba(49, 60, 255, 0.1))`,
                  }}
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}

              {/* Icon */}
              <div className="relative z-10">
                {isActive ? (
                  <div className={`p-2 rounded-lg bg-gradient-to-br ${item.gradient}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                ) : (
                  <Icon className="w-5 h-5 text-slate-400 group-hover:text-white transition-colors" />
                )}
              </div>

              {/* Tooltip */}
              <div className="absolute left-full ml-2 px-3 py-1.5 bg-slate-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap shadow-lg border border-slate-700/50">
                {item.label}
              </div>
            </button>
          );
        })}
      </nav>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Footer */}
      <div className="pb-4 flex flex-col items-center border-t border-slate-700/30 pt-3">
        <div className="text-[10px] text-slate-500 text-center leading-tight px-1">
          <div className="font-semibold">{APP_NAME}</div>
          <div className="text-[8px] mt-0.5">v{APP_VERSION}</div>
        </div>
      </div>
    </motion.aside>
  );
}
