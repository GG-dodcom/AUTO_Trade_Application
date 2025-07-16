import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
	/* config options here */
	reactStrictMode: true,
	webpack(config) {
		config.module.rules.push({
			test: /\.svg$/,
			use: ["@svgr/webpack"],
		});

		// Add alias for @/
		config.resolve.alias["@"] = path.resolve(__dirname, "src");

		return config;
	},
	eslint: {
		// Ensure ESLint runs during builds
		ignoreDuringBuilds: true,
	},
};

export default nextConfig;
