import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The editor fills the window, so keep the dev indicator out of the toolbar.
  devIndicators: { position: "bottom-left" },
};

export default nextConfig;
