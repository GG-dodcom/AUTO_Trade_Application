// src/components/LineChart.tsx
import dynamic from "next/dynamic";

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface LineChartProps {
	height?: number;
	seriesData: number[];
	categories: string[];
	title?: string;
	decimalPlaces?: number;
	lineColor?: string;
	lineStyle?: "smooth" | "straight" | "stepline" | "linestep" | "monotoneCubic";
}

export const LineChart: React.FC<LineChartProps> = ({
	height = 350,
	seriesData,
	categories,
	title = "",
	decimalPlaces = 2, // Default to 2 decimal places
	lineColor = "#00E396", // Default to a green color
	lineStyle = "straight", // Default to straight line
}) => {
	const options: ApexCharts.ApexOptions = {
		chart: {
			height: height,
			width: "100%",
			type: "line",
			zoom: {
				enabled: true, // Enable zoom to see all points
				type: "x", // Zoom along x-axis
				autoScaleYaxis: true,
			},
			toolbar: {
				show: false, // Show zoom/pan tools
				tools: {
					download: false,
					selection: true,
					zoom: true,
					zoomin: true,
					zoomout: true,
					pan: false,
					reset: true,
				},
			},
		},
		dataLabels: {
			enabled: false,
		},
		stroke: {
			curve: lineStyle,
		},
		title: {
			text: title,
			align: "left",
		},
		grid: {
			row: {
				colors: ["#f3f3f3", "transparent"],
				opacity: 0.5,
			},
		},
		xaxis: {
			type: "datetime", // Treat categories as timestamps
			categories: categories, // e.g., ["2025-03-31T10:00:00Z", "2025-03-31T10:01:00Z", ...]
			labels: {
				datetimeUTC: true,
				format: "dd MMM HH:mm", // e.g., "31 Mar 10:00"
			},
		},
		yaxis: {
			labels: {
				formatter: (value: number) => value.toFixed(decimalPlaces),
			},
		},
		tooltip: {
			x: {
				format: "dd MMM yyyy HH:mm", // Show full date and time in tooltip, e.g., "31 Mar 2025 10:00"
			},
			// Optional: Customize the entire tooltip if needed
			custom: ({ series, seriesIndex, dataPointIndex, w }) => {
				const date = new Date(categories[dataPointIndex]);
				const utcDate = date.toLocaleString("en-US", {
					timeZone: "UTC",
					day: "2-digit",
					month: "short",
					year: "numeric",
					hour: "2-digit",
					minute: "2-digit",
					hour12: false,
				});
				const value = series[seriesIndex][dataPointIndex];
				return `
          <div class="apexcharts-tooltip-custom" style="padding: 5px 10px;">
            <span>${utcDate}</span><br/>
            <span>${title.split(" ")[0]}: ${value.toFixed(decimalPlaces)}</span>
          </div>
        `;
			},
		},
	};

	const series = [
		{
			name: title.split(" ")[0],
			data: seriesData,
			color: lineColor,
		},
	];

	return (
		<div id="chart">
			<Chart options={options} series={series} type="line" height={height} />
		</div>
	);
};
