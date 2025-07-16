"use client";
import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import ApexCharts to prevent SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

// Define TypeScript types
interface CandleData {
	x: number; // Timestamp
	y: [number, number, number, number]; // Open, High, Low, Close
}

// Sample candlestick data (for demonstration)
const allData: CandleData[] = [
	{ x: new Date("2023-08-01").getTime(), y: [100, 110, 95, 105] },
	{ x: new Date("2023-09-01").getTime(), y: [105, 115, 100, 110] },
	{ x: new Date("2023-10-01").getTime(), y: [110, 120, 105, 115] },
	{ x: new Date("2023-11-01").getTime(), y: [115, 125, 110, 120] },
	{ x: new Date("2023-12-01").getTime(), y: [120, 130, 115, 125] },
	{ x: new Date("2024-01-01").getTime(), y: [125, 135, 120, 130] },
	{ x: new Date("2024-02-01").getTime(), y: [130, 140, 125, 135] },
];

// Timeframe filter function
const filterDataByTimeframe = (timeframe: string): CandleData[] => {
	const now = new Date();
	let cutoffDate: Date;

	switch (timeframe) {
		case "1D":
			cutoffDate = new Date(now.setDate(now.getDate() - 1));
			break;
		case "1W":
			cutoffDate = new Date(now.setDate(now.getDate() - 7));
			break;
		case "1M":
			cutoffDate = new Date(now.setMonth(now.getMonth() - 1));
			break;
		case "6M":
			cutoffDate = new Date(now.setMonth(now.getMonth() - 6));
			break;
		case "1Y":
			cutoffDate = new Date(now.setFullYear(now.getFullYear() - 1));
			break;
		default:
			return allData; // "ALL" returns all data
	}

	return allData.filter((data) => data.x >= cutoffDate.getTime());
};

export default function CandleChart() {
	const [series, setSeries] = useState([
		{ name: "Candlestick Data", data: allData },
	]);

	// Handle timeframe change
	const handleTimeframeChange = (timeframe: string) => {
		const filteredData = filterDataByTimeframe(timeframe);
		setSeries([{ name: "Candlestick Data", data: filteredData }]);
	};

	// Chart options
	const options: ApexCharts.ApexOptions = {
		chart: {
			type: "candlestick",
			height: 500,
			background: "#f4f4f4",
		},
		title: {
			text: "Candlestick Chart (ApexCharts)",
			align: "left",
		},
		xaxis: {
			type: "datetime",
		},
		yaxis: {
			tooltip: {
				enabled: true,
			},
		},
		tooltip: {
			enabled: true,
		},
	};

	return (
		<div className="p-4 bg-white shadow rounded-lg">
			<div className="flex space-x-2 mb-4">
				{["1D", "1W", "1M", "6M", "1Y", "ALL"].map((timeframe) => (
					<button
						key={timeframe}
						onClick={() => handleTimeframeChange(timeframe)}
						className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
					>
						{timeframe}
					</button>
				))}
			</div>
			{series.length > 0 ? (
				<Chart
					options={options}
					series={series}
					type="candlestick"
					height={500}
				/>
			) : (
				<p>Loading chart...</p>
			)}
		</div>
	);
}
