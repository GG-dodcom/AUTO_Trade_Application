"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts to prevent SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

const XAXISRANGE = 10000; // Adjust time range
const INITIAL_DATA_POINTS = 10; // Initial number of data points

const generateInitialData = () => {
	let data = [];
	let baseTime = new Date().getTime();

	for (let i = 0; i < INITIAL_DATA_POINTS; i++) {
		data.push([baseTime - i * 1000, Math.floor(Math.random() * 80) + 10]);
	}

	return data.reverse();
};

const RealTimeChart = () => {
	const [data, setData] = useState(generateInitialData());
	const [lastDate, setLastDate] = useState(new Date().getTime());

	const [chartOptions, setChartOptions] = useState<ApexOptions>({
		chart: {
			id: "realtime",
			height: 350,
			type: "line",
			animations: {
				enabled: true,
				dynamicAnimation: { speed: 1000 },
			},
			toolbar: { show: false },
			zoom: { enabled: true },
		},
		dataLabels: { enabled: false },
		stroke: { curve: "smooth" },
		title: { text: "Dynamic Updating Chart", align: "left" },
		markers: { size: 0 },
		xaxis: { type: "datetime", range: XAXISRANGE },
		yaxis: { max: 100 },
		legend: { show: false },
	});

	const updateSeries = () => {
		const newLastDate = lastDate + 1000;
		setLastDate(newLastDate);

		setData((prevData) => {
			const newData = [
				...prevData,
				[newLastDate, Math.floor(Math.random() * 80) + 10],
			];
			return newData.slice(-INITIAL_DATA_POINTS); // Keep the last N points
		});
	};

	useEffect(() => {
		const interval = setInterval(updateSeries, 1000 * 60);
		return () => clearInterval(interval);
	}, [lastDate]);

	return (
		<div>
			<Chart
				options={chartOptions}
				series={[{ data }]}
				type="line"
				height={350}
			/>
		</div>
	);
};

export default RealTimeChart;
