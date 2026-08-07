import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        // 1. The path you use in your frontend fetching
        source: "/api/backend/:path*",
        // Fallback to localhost if the environment variable isn't set
        destination: `${process.env.BACKEND_URL || "http://127.0.0.1:8000"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
