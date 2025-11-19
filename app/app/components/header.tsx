import { motion } from "framer-motion";
import Image from "next/image";

/**
 * Application header component
 */
export function Header() {
  return (
    <motion.header
      className="bg-slate-900/50 backdrop-blur-md border-b border-slate-700/50"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="max-w-[1920px] mx-auto px-4 py-3">
        <div className="flex items-center gap-4">
          <div className="flex-shrink-0 bg-white/90 rounded-md px-2 py-1.5 shadow-sm">
            <Image
              src="/vector-logo.webp"
              alt="Vector Institute"
              width={70}
              height={15}
              priority
              className="hover:opacity-90 transition-opacity"
            />
          </div>
          <div className="h-8 w-1 bg-gradient-to-b from-[#EB088A] to-[#313CFF] rounded-full"></div>
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">
              Global AI Alliance for Climate Action
            </h1>
            <h2 className="text-sm font-semibold text-[#EB088A] mt-0.5">
              Early Warning System
            </h2>
          </div>
        </div>
      </div>
    </motion.header>
  );
}
