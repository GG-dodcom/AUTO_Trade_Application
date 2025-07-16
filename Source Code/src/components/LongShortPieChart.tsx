// src/components/LongShortPieChart.tsx
"use client";

import { useState, useEffect } from "react";
import { PieChart } from "./PieChart"; // Adjust path as needed
import { useUser } from "@/hook/useUser";
import { showNotifyToast } from "./NotificationToast";
import { ApiRoute } from "@/app/api";
import { useTrades } from "@/hook/useTrades";

interface LongShortTrades {
	longTrades: number;
	shortTrades: number;
}

const LongShortPieChart: React.FC = () => {
	const { accountId } = useUser();
	const [tradeData, setTradeData] = useState<LongShortTrades | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const { trades } = useTrades();

	useEffect(() => {
		async function fetchLongShortTrades() {
			if (accountId == null) {
				setError("AccountId is null, skipping fetch");
				return;
			}

			setLoading(true); // Reset loading state when fetching starts
			setError(null); // Clear previous errors on retry

			try {
				const response = await fetch(ApiRoute.getLongShortTrades(accountId));
				if (!response.ok) {
					if (trades.length === 0) return;
					showNotifyToast(
						`Failed to fetch long/short trades: ${response.status} ${response.statusText}`,
						"error",
						"bottom-center",
						1000
					);
				}
				const data: LongShortTrades = await response.json();
				setTradeData(data);
			} catch (err) {
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchLongShortTrades();
	}, [accountId]);

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
		series: [tradeData.longTrades, tradeData.shortTrades],
		labels: ["Long Trade", "Short Trade"],
		colors: ["#057A55", "#E02424"], // Green for Long, Red for Short
		total: {
			label: "Total Trades",
			formatter: () => String(tradeData.longTrades + tradeData.shortTrades),
		},
	};

	return (
		<PieChart
			header="Long/Short Trades"
			infoContent={
				<div>
					<p>Breakdown of trade directions:</p>
					<ul>
						<li>
							<strong>Long Trade</strong>: Buy trades
						</li>
						<li>
							<strong>Short Trade</strong>: Sell trades
						</li>
					</ul>
				</div>
			}
			data={chartData}
		/>
	);
};

export default LongShortPieChart;
