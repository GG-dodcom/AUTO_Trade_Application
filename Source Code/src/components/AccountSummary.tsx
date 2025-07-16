// src/components/AccountSummary.tsx
import { useEffect, useState } from "react";
import type { AccountSummary } from "@/lib/definitions";
import { showNotifyToast } from "./NotificationToast";
import { ColumnarTable } from "./ColumnarTable";
import { useUser } from "@/hook/useUser";
import { ApiRoute } from "@/app/api";

const AccountSummary: React.FC = () => {
	const { accountId } = useUser();
	const [summary, setSummary] = useState<AccountSummary | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		async function fetchSummary() {
			if (accountId == null) {
				setError("AccountId is null, skipping fetch");
				return;
			}

			setLoading(true); // Reset loading state when fetching starts
			setError(null); // Clear previous errors on retry

			try {
				const response = await fetch(ApiRoute.getAccountSummary(accountId));
				if (!response.ok) {
					showNotifyToast(
						`Failed to fetch account summary. Status: ${response.status} ${response.statusText}`,
						"error",
						"bottom-center",
						1000
					);
				}
				const data: AccountSummary = await response.json();
				setSummary(data);
			} catch (err) {
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchSummary();
	}, [accountId]);

	if (loading) return <div>Loading...</div>;
	if (error) return <div>Error: {error}</div>;
	if (!summary) return null; // Handle case where summary is null after loading

	// Define columns for ColumnarTable
	const columns = [
		{
			label: "Starting Capital",
			key: "startingCapital" as keyof AccountSummary,
			isMoney: true,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: number) => `$${value.toFixed(2)}`,
		},
		{
			label: "Account Balance",
			key: "accountBalance" as keyof AccountSummary,
			isMoney: true,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: number) => `$${value.toFixed(2)}`,
		},
		{
			label: "Profit / Loss",
			key: "profitLoss" as keyof AccountSummary,
			isMoney: true,
			enableNumberColor: true, // Red/green based on positive/negative
			isPercentage: false,
			colorTheme: null,
			render: (value: number) => `$${value.toFixed(2)}`,
		},
		{
			label: "% Gain / Loss",
			key: "percentGainLoss" as keyof AccountSummary,
			isMoney: false,
			enableNumberColor: true, // Red/green based on positive/negative
			isPercentage: true,
			colorTheme: null,
			render: (value: number) => `${value.toFixed(2)}%`,
		},
		{
			label: "Winning Rate",
			key: "winningRate" as keyof AccountSummary,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: true,
			colorTheme: null,
			render: (value: number) => `${value.toFixed(2)}%`,
		},
		{
			label: "Total R Multiple",
			key: "totalRMultiple" as keyof AccountSummary,
			isMoney: false,
			enableNumberColor: true, // Red/green based on positive/negative
			isPercentage: false,
			colorTheme: null,
			render: (value: number) => value.toFixed(4),
		},
		{
			label: "Expectancy per Trade",
			key: "expectancyPerTrade" as keyof AccountSummary,
			isMoney: false,
			enableNumberColor: true, // Red/green based on positive/negative
			isPercentage: false,
			colorTheme: null,
			render: (value: number) => value.toFixed(4),
		},
	];

	return (
		<div>
			<ColumnarTable
				data={[summary]} // Pass summary as a single-item array since ColumnarTable expects an array
				columns={columns}
				header="ACCOUNT SUMMARY"
				headerCellTheme="!py-2"
				cellTheme="!py-2"
			/>
		</div>
	);
};

export default AccountSummary;
