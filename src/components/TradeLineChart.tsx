// src/components/TradeLineChart.tsx
"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { ApexOptions } from "apexcharts";
import { ChartTrade } from "@/lib/definitions";
import { useChartTrades } from "@/hook/useChartTrades";

// Load ApexCharts dynamically to prevent SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

// Function to transform and validate API data into ChartTrade format
const transformApiTrades = (apiTrades: any[]): ChartTrade[] => {
	return apiTrades
		.map((trade) => {
			const entryTime = trade.entry_time?.replace(" ", "T") + "Z";
			const exitTime = trade.exit_time
				? trade.exit_time.replace(" ", "T") + "Z"
				: null;
			const entryDate = new Date(entryTime);
			const exitDate = exitTime ? new Date(exitTime) : null;

			// Validate all required fields
			if (
				!Number.isFinite(trade.ticket) ||
				isNaN(entryDate.getTime()) ||
				!Number.isFinite(trade.entry_price) ||
				!Number.isFinite(trade.direction) ||
				(exitTime &&
					(isNaN(exitDate!.getTime()) || !Number.isFinite(trade.exit_price)))
			) {
				console.error("Invalid trade data:", trade);
				return null;
			}

			return {
				ticket: trade.ticket,
				entry_time: entryTime,
				exit_time: exitTime,
				entry_price: trade.entry_price,
				exit_price: trade.exit_price ?? null,
				direction: trade.direction,
				profit_loss: trade.profit_loss ?? null,
			};
		})
		.filter((trade): trade is ChartTrade => trade !== null); // Remove invalid trades
};

const TradeLineChart: React.FC = () => {
	const { trades: apiTrades, error } = useChartTrades(); // Fetch trades from API
	const [chartOptions, setChartOptions] = useState<ApexOptions>({
		chart: {
			type: "line",
			height: "100%",
			fontFamily: "Inter, sans-serif",
			dropShadow: { enabled: false },
			toolbar: { show: true },
			zoom: { autoScaleYaxis: true },
		},
		xaxis: { type: "datetime" }, // Required for time-based charts
		yaxis: { show: true },
		series: [], // Ensure series is defined
	});
	const [chartSeries, setChartSeries] = useState<any[]>([]);

	// Inline SVG data URLs for Buy (up triangle) and Sell (down triangle)
	const buyIcon =
		"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 16 16'><polygon points='8,2 14,14 2,14' fill='%234CAF50'/></svg>";
	const sellIcon =
		"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 16 16'><polygon points='8,14 14,2 2,2' fill='%23F44336'/></svg>";

	// Process trades and update chart when trades change
	useEffect(() => {
		if (!apiTrades || !Array.isArray(apiTrades) || apiTrades.length === 0)
			return;

		// Transform API data to ChartTrade format
		const trades = transformApiTrades(apiTrades);

		// Continuous points with metadata for annotations
		const continuousPoints = trades
			.flatMap((trade, tradeIndex) => [
				{
					x: new Date(trade.entry_time).getTime(),
					y: trade.entry_price,
					type: "entry",
					direction: trade.direction,
					tradeIndex,
				},
				...(trade.exit_time && trade.exit_price
					? [
							{
								x: new Date(trade.exit_time).getTime(),
								y: trade.exit_price,
								type: "exit",
								tradeIndex,
							},
					  ]
					: []),
			])
			.sort((a, b) => a.x - b.x);

		const continuousData = continuousPoints.map((point) => [point.x, point.y]);

		// // Calculate profit/loss and assign colors for trade connections
		// const tradeSeries = trades.map((trade, index) => {
		// 	let lineColor = "#888888"; // Default: gray for open trades (B/E or no exit)
		// 	if (trade.exit_price !== null) {
		// 		const profitLoss =
		// 			trade.direction === 0
		// 				? trade.exit_price - trade.entry_price
		// 				: trade.entry_price - trade.exit_price;
		// 		lineColor =
		// 			profitLoss > 0 ? "#4CAF50" : profitLoss < 0 ? "#F44336" : "#888888";
		// 	}
		// 	return {
		// 		name: `Trade ${index + 1} Connection`,
		// 		data: [
		// 			[new Date(trade.entry_time).getTime(), trade.entry_price],
		// 			...(trade.exit_time && trade.exit_price
		// 				? [[new Date(trade.exit_time).getTime(), trade.exit_price]]
		// 				: []),
		// 		],
		// 		color: lineColor, // Green for profit, red for loss, gray for open/BE
		// 		opacity: 1, // Fully opaque dashed lines
		// 		markers: { size: 0 }, // No markers on dashed lines
		// 	};
		// });

		const newOptions: ApexOptions = {
			chart: {
				height: "100%",
				type: "line",
				fontFamily: "Inter, sans-serif",
				dropShadow: { enabled: false },
				toolbar: { show: true },
				zoom: { autoScaleYaxis: true },
			},
			tooltip: {
				enabled: true,
				// enabledOnSeries: [0], // Only for continuous series (index 0)
				x: { show: true, format: "dd MMM yyyy HH:mm" },
				custom: ({ seriesIndex, dataPointIndex }) => {
					if (seriesIndex === 0) {
						const point = continuousPoints[dataPointIndex];
						const time = new Date(point.x).toLocaleString();
						const price = point.y;
						const label =
							point.type === "entry"
								? point.direction === 0
									? "Buy"
									: "Sell"
								: "Close";
						let profitLoss = null;
						if (point.type === "exit") {
							const trade = trades[point.tradeIndex];
							profitLoss = trade.profit_loss;
						}
						return `
          <div class="apexcharts-tooltip-custom p-2">
            <strong>${label}</strong><br/>
						Ticket: ${trades[point.tradeIndex].ticket}<br/>
            Time: ${time}<br/>
            Price: ${price}<br/>
            ${profitLoss !== null ? `P/L: ${profitLoss.toFixed(2)}` : ""}
          </div>`;
					}
					return "";
				},
			},
			dataLabels: { enabled: false },
			stroke: {
				width: 5, // Only one series (continuous), so single width
				dashArray: 0, // Solid line for continuous
				// width: [5, ...Array(numTrades).fill(2)], // Thicker (5) for continuous, thinner (2) for dashed
				// dashArray: [0, ...Array(numTrades).fill(5)], // Solid for continuous, dashed for trade connections
			},
			grid: {
				show: true,
				padding: { left: 2, right: 2, top: -26 },
			},
			xaxis: {
				type: "datetime",
				min: new Date(trades[0].entry_time).getTime() - 3600000, // 1 hour before first trade
				max:
					new Date(
						trades[trades.length - 1].exit_time ||
							trades[trades.length - 1].entry_time
					).getTime() + 3600000, // 1 hour after last trade
				tickAmount: 5,
				labels: {
					show: true,
					style: {
						fontFamily: "Inter, sans-serif",
						cssClass: "text-xs font-normal text-gray-500 dark:text-gray-400",
					},
				},
				axisBorder: { show: false },
				axisTicks: { show: false },
			},
			yaxis: {
				show: true,
				labels: {
					show: true,
					style: {
						fontFamily: "Inter, sans-serif",
						cssClass: "font-normal text-gray-500 dark:text-gray-400",
					},
					align: "left",
					offsetX: -17,
				},
			},
			legend: { show: false },
			markers: {
				size: 6, // Small markers on the continuous line
				shape: "circle",
				colors: "#000000", // Black circles for all points on the line
			},
			annotations: {
				points: continuousPoints.map((point) => {
					if (point.type === "entry") {
						return {
							x: point.x,
							y: point.y,
							image: {
								path: point.direction === 0 ? buyIcon : sellIcon, // Buy: up triangle, Sell: down triangle
								width: 24,
								height: 24,
								offsetX: 0,
								offsetY: 0,
							},
						};
					} else {
						return {
							x: point.x,
							y: point.y,
							marker: {
								size: 10,
								shape: "circle",
								fillColor: "#000000",
								strokeWidth: 2,
								strokeColor: "#fff",
							},
						};
					}
				}),
			},
		};

		const newSeries = [
			// Continuous line first (solid blue, slightly transparent, rendered beneath)
			{
				name: "Price Movement",
				data: continuousData,
				color: "#dae1f0", // Blue
				opacity: 0.6, // Slightly transparent
			},
			// // Trade connections last (dashed, fully opaque, rendered on top)
			// ...tradeSeries,
		];

		setChartOptions(newOptions);
		setChartSeries(newSeries);
	}, [apiTrades]); // Depend on apiTrades

	const zoomChart = (start: number, end: number) => {
		setChartOptions((prev) => ({
			...prev,
			xaxis: { ...prev.xaxis, min: start, max: end } as any, // Type workaround
		}));
	};

	// Time range functions
	const currentDate = new Date(); // Current date as per system info
	const todayStart = new Date(
		currentDate.getFullYear(),
		currentDate.getMonth(),
		currentDate.getDate()
	);
	const todayEnd = currentDate.getTime();
	const oneMonthAgo = new Date(currentDate);
	oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
	const sixMonthsAgo = new Date(currentDate);
	sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
	const oneYearAgo = new Date(currentDate);
	oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
	const yearStart = new Date(currentDate.getFullYear(), 0, 1); // Jan 1, 2025
	const allStart = apiTrades?.length
		? new Date(apiTrades[0].entry_time).getTime() - 3600000
		: 0;
	const allEnd = apiTrades?.length
		? new Date(
				apiTrades[apiTrades.length - 1].exit_time ||
					apiTrades[apiTrades.length - 1].entry_time
		  ).getTime() + 3600000
		: currentDate.getTime(); // 1 hour after last trade

	// Calculate performance metrics
	const trades = transformApiTrades(apiTrades || []);
	const lastTrade = trades[trades.length - 1] || {
		entry_price: 0,
		exit_price: null,
		profit_loss: 0,
	};
	const totalProfitLoss = trades.reduce(
		(sum: number, trade: ChartTrade) =>
			trade.profit_loss !== null ? sum + trade.profit_loss : sum,
		0
	);

	return (
		<div className="w-full bg-white rounded-lg shadow-sm dark:bg-gray-800 p-4 md:p-6">
			<div className="flex justify-between mb-5">
				<div className="grid gap-4 grid-cols-3">
					<div>
						<h5 className="inline-flex items-center text-gray-500 dark:text-gray-400 font-normal mb-2">
							Last Entry Price
						</h5>
						<p className="text-gray-900 dark:text-white text-2xl font-bold">
							{lastTrade.entry_price}
						</p>
					</div>
					<div>
						<h5 className="inline-flex items-center text-gray-500 dark:text-gray-400 font-normal mb-2">
							Last Exit Price
						</h5>
						<p className="text-gray-900 dark:text-white text-2xl font-bold">
							{lastTrade.exit_price || "N/A"}
						</p>
					</div>
					<div>
						<h5 className="inline-flex items-center text-gray-500 dark:text-gray-400 font-normal mb-2">
							Total P/L
						</h5>
						<p
							className={`text-2xl font-bold ${
								totalProfitLoss >= 0 ? "text-green-600" : "text-red-600"
							} dark:text-white`}
						>
							{totalProfitLoss.toFixed(2)}
						</p>
					</div>
				</div>
				<div className="flex space-x-2 mb-4">
					<button
						onClick={() => zoomChart(todayStart.getTime(), todayEnd)}
						disabled={!trades.length}
					>
						Today
					</button>
					<button
						onClick={() =>
							zoomChart(oneMonthAgo.getTime(), currentDate.getTime())
						}
						disabled={!trades.length}
					>
						1 Month
					</button>
					<button
						onClick={() =>
							zoomChart(sixMonthsAgo.getTime(), currentDate.getTime())
						}
						disabled={!trades.length}
					>
						6 Months
					</button>
					<button
						onClick={() =>
							zoomChart(oneYearAgo.getTime(), currentDate.getTime())
						}
						disabled={!trades.length}
					>
						1 Year
					</button>
					<button
						onClick={() =>
							zoomChart(yearStart.getTime(), currentDate.getTime())
						}
						disabled={!trades.length}
					>
						YTD
					</button>
					<button
						onClick={() => zoomChart(allStart, allEnd)}
						disabled={!trades.length}
					>
						All
					</button>
				</div>
			</div>

			<div className="w-full h-fit">
				{error ? (
					<p className="text-red-500 dark:text-red-400">{error}</p>
				) : chartSeries.length > 0 ? (
					<Chart
						options={chartOptions}
						series={chartSeries}
						type="line"
						height={350}
					/>
				) : (
					<p className="text-gray-500 dark:text-gray-400">
						{trades.length === 0 ? "No trades available" : "Loading chart..."}
					</p>
				)}
			</div>

			{/* Custom Legend */}
			<div className="mt-4 flex flex-wrap gap-4 text-sm text-gray-700 dark:text-gray-300">
				<div className="flex items-center">
					<div className="w-4 h-4 bg-[#dae1f0] mr-2"></div>
					<span>Price Movement</span>
				</div>
				<div className="flex items-center">
					<img src={buyIcon} alt="Buy" className="w-4 h-4 mr-2" />
					<span>Buy Entry</span>
				</div>
				<div className="flex items-center">
					<img src={sellIcon} alt="Sell" className="w-4 h-4 mr-2" />
					<span>Sell Entry</span>
				</div>
				<div className="flex items-center">
					<div className="w-4 h-4 bg-black rounded-full mr-2"></div>
					<span>Trade Exit</span>
				</div>
				{/* <div className="flex items-center">
					<div className="w-4 h-1 border-t-2 border-dashed border-[#4CAF50] mr-2"></div>
					<span>Profitable Trade</span>
				</div>
				<div className="flex items-center">
					<div className="w-4 h-1 border-t-2 border-dashed border-[#F44336] mr-2"></div>
					<span>Loss Trade</span>
				</div>
				<div className="flex items-center">
					<div className="w-4 h-1 border-t-2 border-dashed border-[#888888] mr-2"></div>
					<span>Open/Breakeven Trade</span>
				</div> */}
			</div>
		</div>
	);
};

export default TradeLineChart;
