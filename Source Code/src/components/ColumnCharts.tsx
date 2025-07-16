// src/components/ColumnCharts.tsx
import React, { useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

// Define props interface
interface ColumnChartsProps {
	seriesData?: ApexAxisChartSeries;
	categories?: string[];
	height?: number;
	columnWidth?: string;
	colors?: string[] | ((opts: any) => string); // For series-based coloring
	dynamicColors?: { from: number; to: number; color: string }[]; // For value-based coloring
	yLabel?: string;
	yFormatter?: (val: number) => string; // Optional custom formatter for y-axis
}

const ColumnCharts: React.FC<ColumnChartsProps> = ({
	seriesData = [
		{
			name: "Net Profit",
			data: [44, 55, 57, 56, 61, 58, 63, 60, 66],
		},
		{
			name: "Revenue",
			data: [76, 85, 101, 98, 87, 105, 91, 114, 94],
		},
		{
			name: "Free Cash Flow",
			data: [35, 41, 36, 26, 45, 48, 52, 53, 41],
		},
	],
	categories = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
	height = 350,
	columnWidth = "55%",
	colors = ["#00E396", "#FF4560", "#775DD0"],
	dynamicColors,
	yLabel = undefined,
	yFormatter = (val: number) => `${val}`, // Default formatter
}) => {
	const options: ApexCharts.ApexOptions = {
		chart: {
			type: "bar",
			height: height,
			toolbar: {
				show: false, // Show zoom/pan tools
			},
		},
		plotOptions: {
			bar: {
				horizontal: false,
				columnWidth: columnWidth,
				borderRadius: 5,
				borderRadiusApplication: "end",
				...(dynamicColors && {
					colors: {
						ranges: dynamicColors,
					},
				}), // Apply dynamic colors if provided
			},
		},
		colors: dynamicColors
			? undefined
			: Array.isArray(colors)
			? colors
			: undefined, // Ensure colors is an array or undefined
		dataLabels: {
			enabled: false,
		},
		stroke: {
			show: true,
			width: 2,
			colors: ["transparent"],
		},
		xaxis: {
			categories: categories,
			labels: {
				style: {
					// fontSize: "12px",
					// fontFamily: "Arial, sans-serif",
				},
			},
		},
		yaxis: {
			title: {
				text: yLabel,
				style: {
					fontSize: "14px",
					fontWeight: 600,
					fontFamily: "Arial, sans-serif",
				},
			},
			labels: {
				formatter: yFormatter,
			},
		},
		fill: {
			opacity: 1,
		},
		tooltip: {
			custom: ({ series, seriesIndex, dataPointIndex }) => {
				const value = series[seriesIndex][dataPointIndex];
				const category = categories[dataPointIndex] || "Unknown"; // Fallback if undefined
				const seriesName = seriesData[seriesIndex].name;
				// Determine tooltip color: use dynamicColors if provided, else fall back to colors array
				const color = dynamicColors
					? dynamicColors.find(
							(range) => value >= range.from && value < range.to
					  )?.color || "#000000" // Default to black if no range matches
					: Array.isArray(colors)
					? colors[seriesIndex]
					: "#000000"; // Default fallback
				return `
            <div class="apexcharts-tooltip-custom" style="padding: 5px 10px;">
              <span>${category}</span><br/>
            <span style="color: ${color};">●</span> ${seriesName}: ${yFormatter(
					value
				)}
            </div>
          `;
			},
		},
	};

	return (
		<div>
			<div id="chart">
				<Chart
					options={options}
					series={seriesData}
					type="bar"
					height={height}
				/>
			</div>
			<div id="html-dist"></div>
		</div>
	);
};

export default ColumnCharts;
