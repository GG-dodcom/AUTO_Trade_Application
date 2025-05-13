"use client";
import BlockContainer from "@/components/BlockContainer";
import BlockItem from "@/components/BlockItem";
import { LineChart } from "@/components/LineChart";
import MonthlyStatistics from "@/components/MonthlyStatistics";
import { Pagination } from "@/components/Pagination";
import { RowTableWithDetail } from "@/components/RowTableWithDetail";
import TradePerformancePieChart from "@/components/TradePerformancePieChart";
import { JournalColumns, journalHeader1Groups } from "@/data/tableColumns";
import { useTrades } from "@/hook/useTrades";
import { useState, useMemo, useEffect } from "react";
import { useUser } from "@/hook/useUser";
import { ApiRoute } from "@/app/api";
import { debounce } from "@/lib/lib";

// Date utility functions
const startOfDay = (date: Date): Date => {
	const newDate = new Date(date);
	newDate.setUTCHours(0, 0, 0, 0);
	return newDate;
};

const endOfDay = (date: Date): Date => {
	const newDate = new Date(date);
	newDate.setUTCHours(23, 59, 59, 999);
	return newDate;
};

// Helper function to calculate the start and end of the selected month
const getMonthRange = (
	monthString: string
): { startDate: Date; endDate: Date } => {
	const [monthName, year] = monthString.split(" ");
	const monthNames = [
		"January",
		"February",
		"March",
		"April",
		"May",
		"June",
		"July",
		"August",
		"September",
		"October",
		"November",
		"December",
	];
	const monthIndex = monthNames.indexOf(monthName);
	const yearNum = parseInt(year, 10);

	if (monthIndex === -1 || isNaN(yearNum)) {
		throw new Error(`Invalid month format: ${monthString}`);
	}

	const startDate = new Date(Date.UTC(yearNum, monthIndex, 1));
	const endDate = new Date(Date.UTC(yearNum, monthIndex + 1, 0));

	return {
		startDate: startOfDay(startDate), // 2025-04-01T00:00:00.000Z
		endDate: endOfDay(endDate), // 2025-04-30T23:59:59.999Z
	};
};

const MonthlyReport: React.FC = () => {
	const { accountId } = useUser();
	const [currentPage, setCurrentPage] = useState(1);
	const [chartKey, setChartKey] = useState(0); // State to force chart re-render

	// Get current month and year for default selectedMonth
	const getCurrentMonthYear = (): string => {
		const date = new Date();
		const monthNames = [
			"January",
			"February",
			"March",
			"April",
			"May",
			"June",
			"July",
			"August",
			"September",
			"October",
			"November",
			"December",
		];
		const month = monthNames[date.getMonth()];
		const year = date.getFullYear();
		return `${month} ${year}`;
	};

	const [selectedMonth, setSelectedMonth] = useState<string>(
		getCurrentMonthYear()
	);
	const [balanceData, setBalanceData] = useState<
		{ entry_time: string; account_balance: number }[]
	>([]);
	const [pnlData, setPnlData] = useState<
		{ entry_time: string; cumulative_pnl: number }[]
	>([]);
	const [drawdownData, setDrawdownData] = useState<
		{ entry_time: string; drawdown: number }[]
	>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	// Calculate the date range based on selectedMonth
	const { startDate, endDate } = useMemo(
		() => getMonthRange(selectedMonth),
		[selectedMonth]
	);

	// console.log("Start of Month:", startDate.toISOString()); // e.g., 2025-04-01T00:00:00.000Z
	// console.log("End of Month:", endDate.toISOString()); // e.g., 2025-04-30T23:59:59.999Z

	// Fetch trades for the selected month's date range
	const { trades } = useTrades(startDate, endDate);

	// Auto resize with debounce
	useEffect(() => {
		const handleResize = debounce(() => {
			setChartKey((prev) => prev + 1); // Force re-render
			window.dispatchEvent(new Event("resize")); // Notify ApexCharts
		}, 300); // Reduced to 300ms for responsiveness

		return handleResize;
	}, [selectedMonth]); // Empty dependency array to run once on mount

	// Fetch chart data
	useEffect(() => {
		async function fetchChartData() {
			if (!accountId) return;
			setLoading(true);
			setError(null);

			try {
				// Fetch Account Balance
				const balanceResponse = await fetch(
					ApiRoute.getAccountBalance(
						accountId,
						startDate.toISOString(),
						endDate.toISOString()
					)
				);
				if (!balanceResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch balance data");
				}
				const balanceJson = await balanceResponse.json();
				setBalanceData(balanceJson);

				// Fetch Cumulative P&L
				const pnlResponse = await fetch(
					ApiRoute.getCummulativePnL(
						accountId,
						startDate.toISOString(),
						endDate.toISOString()
					)
				);
				if (!pnlResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch P&L data");
				}
				const pnlJson = await pnlResponse.json();
				setPnlData(pnlJson);

				// Fetch Account Drawdown
				const drawdownResponse = await fetch(
					ApiRoute.getDrawdown(
						accountId,
						startDate.toISOString(),
						endDate.toISOString()
					)
				);
				if (!drawdownResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch drawdown data");
				}
				const drawdownJson = await drawdownResponse.json();
				setDrawdownData(drawdownJson);
			} catch (err) {
				if (trades.length === 0) return;
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}

		fetchChartData();
	}, [accountId, selectedMonth, startDate, endDate]);

	const itemsPerPage = 6;

	// Add "no" field to each trade
	const tradesWithNo = trades.map((trade, index) => ({
		...trade,
		no: index + 1,
	}));
	const totalItems = tradesWithNo.length;
	const startIndex = (currentPage - 1) * itemsPerPage;
	const endIndex = startIndex + itemsPerPage;
	const paginatedData = tradesWithNo.slice(startIndex, endIndex);

	const handlePageChange = (page: number) => {
		setCurrentPage(page);
	};

	// Prepare LineChart data
	const balanceSeries = balanceData.map((d) => d.account_balance);
	const pnlSeries = pnlData.map((d) => d.cumulative_pnl);
	const drawdownSeries = drawdownData.map((d) => d.drawdown);
	const categories = balanceData.map((d) => {
		const date = new Date(d.entry_time);
		return date.toISOString(); // Use full timestamp, e.g., "2025-03-31T10:00:00Z"
	});

	return (
		<div>
			<BlockContainer>
				<BlockItem className="min-w-[340px]">
					<MonthlyStatistics
						selectedMonth={selectedMonth}
						onMonthChange={(month: string) => setSelectedMonth(month)}
					/>
				</BlockItem>

				<BlockItem>
					<TradePerformancePieChart
						startDate={startDate.toISOString()}
						endDate={endDate.toISOString()}
					/>
				</BlockItem>

				{loading ? (
					<div>Loading charts...</div>
				) : error ? (
					<div>Error: {error}</div>
				) : (
					<>
						<BlockItem>
							<LineChart
								key={`balance-${chartKey}`} // Unique key per chart
								height={350}
								title="Account Balance"
								seriesData={balanceSeries}
								categories={categories}
								decimalPlaces={2} // 2 decimals for balance
								lineColor="#057A55" // Green line
								lineStyle="straight" // Straight line
							/>
						</BlockItem>

						<BlockItem>
							<LineChart
								key={`pnl-${chartKey}`} // Unique key per chart
								height={350}
								title="Cumulative P&L"
								seriesData={pnlSeries}
								categories={categories}
								decimalPlaces={2} // 2 decimals for P&L
								lineColor="#FF4560" // Red line
								lineStyle="smooth" // Smooth curve
							/>
						</BlockItem>

						<BlockItem>
							<LineChart
								key={`drawdown-${chartKey}`} // Unique key per chart
								height={350}
								title="Account Drawdown"
								seriesData={drawdownSeries}
								categories={categories}
								decimalPlaces={3} // 3 decimals for drawdown
								lineColor="#4267B2" // Blue line
								lineStyle="smooth" // Step line
							/>
						</BlockItem>
					</>
				)}
			</BlockContainer>
			<BlockContainer>
				<div className="w-full rounded-lg shadow-sm p-4 bg-white">
					<div className="flex justify-between items-center px-4 pb-4">
						<h2 className="text-lg font-semibold">Trading Journal</h2>
					</div>
					<div className="px-4 pb-4">
						{trades.length > 0 ? (
							<RowTableWithDetail
								data={paginatedData}
								columns={JournalColumns}
								header1Groups={journalHeader1Groups}
							/>
						) : (
							<p className="text-gray-500 text-center py-4">
								No trades found for this date range.
							</p>
						)}
						<Pagination
							currentPage={currentPage}
							itemsPerPage={itemsPerPage}
							totalItems={totalItems}
							onPageChange={handlePageChange}
							className="justify-self-end w-[-webkit-fill-available]"
						/>
					</div>
				</div>
			</BlockContainer>
		</div>
	);
};

export default MonthlyReport;
