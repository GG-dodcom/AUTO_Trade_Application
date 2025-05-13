import type { Config } from "tailwindcss";

export default {
	content: [
		"./src/components/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/app/**/*.{js,ts,jsx,tsx,mdx}",
	],
	theme: {
		extend: {
			colors: {
				background: "var(--background)",
				foreground: "var(--foreground)",
				buy: "#057A55", // Green for Buy (Upward Trend)
				sell: "#E02424", // Red for Sell (Downward Trend)
			},
		},
	},
	plugins: [],
} satisfies Config;
