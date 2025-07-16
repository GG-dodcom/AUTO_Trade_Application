// src/components/MonthlyStatistics.tsx
"use client";
import React, { useState, useEffect, ReactNode } from "react";
import { ColumnarTable } from "./ColumnarTable"; // Adjust the import path as needed
import { useUser } from "@/hook/useUser";
import type { MonthlyStatistics } from "@/lib/definitions"; // Adjust the import path as needed
import { ApiRoute } from "@/app/api";

interface MonthlyStatisticsProps {
	selectedMonth: string;
	onMonthChange: (month: string) => void;
}

const MonthlyStatistics: React.FC<MonthlyStatisticsProps> = ({
	selectedMonth,
	onMonthChange,
}) => {
	const { accountId } = useUser();
	const [tableData, setTableData] = useState<MonthlyStatistics[]>([]);
	const [months, setMonths] = useState<string[]>([]);
	const [loading, setLoading] = useState<boolean>(true);
	const [error, setError] = useState<string | null>(null);

	// Fetch data on mount and when selectedMonth or accountId changes
	useEffect(() => {
		async function fetchMonthlyStatistics() {
			if (!accountId) return;
			setLoading(true);
			setError(null);

			try {
				const response = await fetch(
					ApiRoute.getMonthlyStatistics(
						encodeURIComponent(selectedMonth),
						accountId
					)
				);
				if (!response.ok) {
					throw new Error(
						`Failed to fetch data: ${response.status} ${response.statusText}`
					);
				}
				const data = await response.json();

				// Set the months from the response
				if (data.months) {
					setMonths(data.months);
				}

				// Set the table data
				setTableData([data.statistics]); // Wrap in array as ColumnarTable expects an array
			} catch (err) {
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchMonthlyStatistics();
	}, [selectedMonth, accountId]);

	// Column configuration matching the AccountSummary example
	const columns = [
		{
			label: "Account Starting Balance",
			key: "accountStartingBalance" as keyof MonthlyStatistics,
			isMoney: true,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return `$${numValue.toLocaleString(undefined, {
					minimumFractionDigits: 2,
					maximumFractionDigits: 2,
				})}`;
			},
		},
		{
			label: "Account Ending Balance",
			key: "accountEndingBalance" as keyof MonthlyStatistics,
			isMoney: true,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return `$${numValue.toLocaleString(undefined, {
					minimumFractionDigits: 2,
					maximumFractionDigits: 2,
				})}`;
			},
		},
		{
			label: "Profit / Loss",
			key: "profitLoss" as keyof MonthlyStatistics,
			isMoney: true,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return `$${numValue.toLocaleString(undefined, {
					minimumFractionDigits: 2,
					maximumFractionDigits: 2,
				})}`;
			},
		},
		{
			label: "% Gain / Loss",
			key: "percentGainLoss" as keyof MonthlyStatistics,
			isMoney: false,
			enableNumberColor: true,
			isPercentage: true,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return `${numValue.toFixed(2)}%`;
			},
		},
		{
			label: "Winning Rate",
			key: "winningRate" as keyof MonthlyStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: true,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return `${numValue.toFixed(2)}%`;
			},
		},
		{
			label: "Total R Multiple",
			key: "totalRMultiple" as keyof MonthlyStatistics,
			isMoney: false,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return numValue.toFixed(2);
			},
		},
		{
			label: "Expectancy/Trade",
			key: "expectancyPerTrade" as keyof MonthlyStatistics,
			isMoney: false,
			enableNumberColor: true,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => {
				const numValue = typeof value === "string" ? parseFloat(value) : value;
				return numValue.toFixed(2);
			},
		},
		{
			label: "Avg. Time in Trade",
			key: "avgTimeInTrade" as keyof MonthlyStatistics,
			isMoney: false,
			enableNumberColor: false,
			isPercentage: false,
			colorTheme: null,
			render: (value: string | number): ReactNode => value,
		},
	];

	// Custom header with dropdown
	const customHeader = (
		<div className="flex justify-center items-center">
			<select
				value={selectedMonth}
				onChange={(e) => onMonthChange(e.target.value)}
				className="bg-gray-50 border-none shadow text-gray-700 font-bold text-sm p-2 rounded w-full text-center focus:outline-none"
			>
				{months.map((month) => (
					<option key={month} value={month}>
						{month}
					</option>
				))}
			</select>
		</div>
	);

	if (loading) return <div className="text-center p-4">Loading...</div>;
	if (error)
		return <div className="text-center p-4 text-red-600">Error: {error}</div>;

	return (
		<ColumnarTable
			data={tableData}
			columns={columns}
			header={customHeader as any}
			headerTheme="!p-0"
			headerCellTheme="!py-2"
			cellTheme="!py-2 hover:bg-[#e8f5e8] transition-colors duration-150"
		/>
	);
};

export default MonthlyStatistics;
