import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Enable experimental features for better Cloud Run compatibility
  experimental: {
    serverActions: {
      bodySizeLimit: "2mb",
    },
  },
};

export default nextConfig;
