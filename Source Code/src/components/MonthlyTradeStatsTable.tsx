// src/components/MonthlyTradeStatsTable.tsx
"use client";

import { useState, useEffect } from "react";
import { RowTable } from "./RowTable"; // Adjust path
import { useUser } from "@/hook/useUser";
import { showNotifyToast } from "./NotificationToast";
import { ColumnConfig, MonthlyTradeStats } from "@/lib/definitions";
import { monthTradeStat_Header1Groups } from "@/data/tableColumns";
import { ApiRoute } from "@/app/api";
import { useTrades } from "@/hook/useTrades";

const MonthlyTradeStatsTable: React.FC = () => {
	const { accountId } = useUser();
	const [stats, setStats] = useState<MonthlyTradeStats[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const { trades } = useTrades();

	useEffect(() => {
		async function fetchMonthlyStats() {
			if (accountId == null) {
				setError("AccountId is null, skipping fetch");
				return;
			}

			setLoading(true); // Reset loading state when fetching starts
			setError(null); // Clear previous errors on retry

			try {
				const response = await fetch(ApiRoute.getMonthlyTradeStats(accountId));
				if (!response.ok) {
					if (trades.length === 0) return;
					showNotifyToast(
						`Failed to fetch monthly stats: ${response.status} ${response.statusText}`,
						"error",
						"bottom-center",
						1000
					);
				}
				const data: MonthlyTradeStats[] = await response.json();
				setStats(data);
			} catch (err) {
				if (trades.length === 0) return;
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchMonthlyStats();
	}, [accountId]);

	if (loading) return <div>Loading...</div>;
	if (error) return <div>Error: {error}</div>;

	const columns: ColumnConfig<MonthlyTradeStats>[] = [
		{ label: "Month", key: "month", rowHeader: true, width: "w-24" },
		{
			label: "Profit",
			key: "profit",
			isMoney: true,
			enableNumberColor: true,
			width: "w-32",
		},
		{
			label: "R Multiple",
			key: "rMultiple",
			enableNumberColor: true,
			width: "w-28",
			render: (value: string | number) =>
				typeof value === "number" ? value.toFixed(2) : value,
		},
		{ label: "Winning Trade", key: "winningTrades", width: "w-32" },
		{ label: "Losing Trade", key: "losingTrades", width: "w-32" },
		{ label: "B/E", key: "breakEvenTrades", width: "w-24" },
	];

	return (
		<RowTable
			data={stats}
			columns={columns}
			header1Groups={monthTradeStat_Header1Groups}
			cellTheme="!py-2"
		/>
	);
};

export default MonthlyTradeStatsTable;
