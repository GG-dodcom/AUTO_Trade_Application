// src/components/TradePerformancePieChart.tsx
"use client";

import { useState, useEffect } from "react";
import { PieChart } from "./PieChart";
import { useUser } from "@/hook/useUser";
import { showNotifyToast } from "./NotificationToast";
import { TradePerformance } from "@/lib/definitions";
import { ApiRoute } from "@/app/api";
import { useTrades } from "@/hook/useTrades";

interface TradePerformancePieChartProps {
	startDate?: string; // e.g., "2025-01-01"
	endDate?: string; // e.g., "2025-12-31"
}

const TradePerformancePieChart: React.FC<TradePerformancePieChartProps> = ({
	startDate,
	endDate,
}) => {
	const { accountId } = useUser();
	const [tradeData, setTradeData] = useState<TradePerformance | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const { trades } = useTrades();

	useEffect(() => {
		async function fetchTradePerformance() {
			if (accountId == null) {
				setError("AccountId is null, skipping fetch");
				return;
			}

			setLoading(true); // Reset loading state when fetching starts
			setError(null); // Clear previous errors on retry

			try {
				const response = await fetch(
					ApiRoute.getTradePerformance(accountId, startDate, endDate)
				);
				if (!response.ok) {
					if (trades.length === 0) return;
					showNotifyToast(
						`Failed to fetch trade performance: ${response.status} ${response.statusText}`,
						"error",
						"bottom-center",
						1000
					);
					throw new Error(`HTTP error: ${response.status}`);
				}
				const data: TradePerformance = await response.json();
				setTradeData(data);
			} catch (err) {
				if (trades.length === 0) return;
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchTradePerformance();
	}, [accountId, startDate, endDate]);

	if (loading) return <div>Loading...</div>;
	if (error) return <div>Error: {error}</div>;
	if (!tradeData) {
		return (
			<div
				style={{
					height: 350,
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					color: "#666",
				}}
			>
				No trade data available
			</div>
		);
	}

	const chartData = {
		series: [
			tradeData.winningTrades,
			tradeData.losingTrades,
			tradeData.breakEvenTrades,
		],
		labels: ["Winning Trade", "Losing Trade", "B/E"],
		colors: ["#057A55", "#E02424", "#2563eb"], // Green for wins, Red for losses, Blue for B/E
		total: {
			label: "Total Trades",
			formatter: () =>
				String(
					tradeData.winningTrades +
						tradeData.losingTrades +
						tradeData.breakEvenTrades
				),
		},
	};

	return (
		<PieChart
			header="Trade Performance"
			infoContent={
				<div>
					<p>Breakdown of trade outcomes:</p>
					<ul>
						<li>
							<strong>Winning Trade</strong>: Trades with a profit.
						</li>
						<li>
							<strong>Losing Trade</strong>: Trades with a loss.
						</li>
						<li>
							<strong>B/E</strong>: Break-even trades (no profit or loss).
						</li>
					</ul>
				</div>
			}
			data={chartData}
		/>
	);
};

export default TradePerformancePieChart;
