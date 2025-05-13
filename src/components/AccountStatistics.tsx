// src/components/AccountStatistics.tsx
import { useEffect, useState } from "react";
import type { AccountStatistics } from "@/lib/definitions"; // Adjust if AccountStatistics type is elsewhere
import { showNotifyToast } from "./NotificationToast";
import { ColumnarTable } from "./ColumnarTable";
import { useUser } from "@/hook/useUser";
import { ApiRoute } from "@/app/api";

const AccountStatistics: React.FC = () => {
	const { accountId } = useUser();
	const [stats, setStats] = useState<AccountStatistics | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		async function fetchStatistics() {
			if (accountId == null) {
				setError("AccountId is null, skipping fetch");
				return;
			}

			setLoading(true); // Reset loading state when fetching starts
			setError(null); // Clear previous errors on retry

			try {
				const response = await fetch(ApiRoute.getAccountStatistics(accountId));
				if (!response.ok) {
					showNotifyToast(
						`Failed to fetch account statistics: ${response.status} ${response.statusText}`,
						"error",
						"bottom-center",
						1000
					);
				}
				const data: AccountStatistics = await response.json();
				setStats(data);
			} catch (err) {
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchStatistics();
	}, [accountId]);

	if (loading) return <div>Loading...</div>;
	if (error) return <div>Error: {error}</div>;
	if (!stats) return null;

	const columns = [
		{
			label: "Average Win",
			key: "averageWin" as keyof AccountStatistics,
			isMoney: true,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				`$${(value as number).toFixed(2)}`,
		},
		{
			label: "Average Loss",
			key: "averageLoss" as keyof AccountStatistics,
			isMoney: true,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number) => `$${(value as number).toFixed(2)}`,
		},
		{
			label: "Largest Win",
			key: "largestWin" as keyof AccountStatistics,
			isMoney: true,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number) => `$${(value as number).toFixed(2)}`,
		},
		{
			label: "Largest Loss",
			key: "largestLoss" as keyof AccountStatistics,
			isMoney: true,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				`$${(value as number).toFixed(2)}`,
		},
		{
			label: "Max Consec. Wins",
			key: "maxConsecutiveWins" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				(value as number).toFixed(0),
		},
		{
			label: "Max Consec. Loss",
			key: "maxConsecutiveLosses" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				(value as number).toFixed(0),
		},
		{
			label: "Max Drawdown",
			key: "maxDrawdown" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: true,
			isPercentage: true,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				`${(value as number).toFixed(2)}%`,
		},
		{
			label: "Long Trade",
			key: "longTrades" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number) => (value as number).toFixed(0),
		},
		{
			label: "Short Trade",
			key: "shortTrades" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				(value as number).toFixed(0),
		},
		{
			label: "Avg. Time in Trade",
			key: "avgTimeInTrade" as keyof AccountStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number, row: AccountStatistics) =>
				value as string, // Already formatted as "Xh Ym"
		},
	];

	return (
		<div>
			<ColumnarTable
				data={[stats]}
				columns={columns}
				header="ACCOUNT STATISTICS"
				cellTheme="!py-2"
				headerCellTheme="!py-2"
			/>
		</div>
	);
};

export default AccountStatistics;
