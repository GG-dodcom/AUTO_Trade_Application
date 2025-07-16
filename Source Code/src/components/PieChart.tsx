"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import PopoverInfo from "./PopoverInfo";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts (for SSR support)
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface Props {
	header?: string;
	infoContent?: React.ReactNode;
	data: {
		series: any[];
		labels: string[]; // Corresponding labels
		colors?: any[]; // Custom colors (optional)
		formatter?: (value: any) => string; // Custom formatting for axis labels
		total?: {
			label: string;
			formatter: (value: any) => string; // Custom formatting for total
		};
	};
}

export const PieChart: React.FC<Props> = ({ header, infoContent, data }) => {
	// Chart Options
	const [chartOptions, setChartOptions] = useState<ApexOptions>({
		chart: { height: 320, type: "donut" },
		colors: data.colors || ["#1C64F2", "#16BDCA", "#FDBA8C", "#E74694"],
		stroke: { colors: ["transparent"] },
		plotOptions: {
			pie: {
				donut: {
					size: "65%",
					labels: {
						show: true,
						total: {
							show: true,
							label: data.total?.label || undefined,
							formatter: data.total?.formatter || (() => ""),
						},
					},
				},
			},
		},
		labels: data.labels,
		legend: { position: "bottom" },
		dataLabels: { enabled: true },
		xaxis: { labels: { formatter: data?.formatter || undefined } },
		yaxis: { labels: { formatter: data?.formatter || undefined } },
	});

	return (
		<div className="max-w-sm w-full bg-white rounded-lg dark:bg-gray-800 ">
			{/* Header */}
			{header && (
				<div className="flex justify-between">
					<div className="flex items-center">
						<h5 className="text-xl font-bold text-gray-900 dark:text-white pe-1">
							{header}
						</h5>
						{infoContent && (
							<PopoverInfo type={"question"} content={infoContent} />
						)}
					</div>
				</div>
			)}

			{/* Donut Chart */}
			<div className="py-4">
				<Chart
					options={chartOptions}
					series={data.series}
					type="donut"
					height={320}
				/>
			</div>
		</div>
	);
};
