// src/app/page.tsx
"use client";
import { DropdownMenu } from "@/components/DropdownMenu";
import TradingJournalBlock from "@/components/TradingJournalBlock";
import { useCallback, useEffect, useState } from "react";
import TradeLineChart from "@/components/TradeLineChart";
import BlockContainer from "@/components/BlockContainer";
import BlockItem from "@/components/BlockItem";
import AccountSummary from "@/components/AccountSummary";
import AccountStatistics from "@/components/AccountStatistics";
import TradePerformancePieChart from "@/components/TradePerformancePieChart";
import LongShortPieChart from "@/components/LongShortPieChart";
import MonthlyTradeStatsTable from "@/components/MonthlyTradeStatsTable";
import { LineChart } from "@/components/LineChart";
import { ApiRoute } from "./api";
import { debounce } from "@/lib/lib";
import { useUser } from "@/hook/useUser";
import ColumnCharts from "@/components/ColumnCharts";
import { useTrades } from "@/hook/useTrades";
import TradeNotification from "@/components/TradeNotification";

export default function Home() {
	const { accountId } = useUser();
	const [selectedChart, setSelectedChart] = useState("Trading Performance");
	const [isPieChartOpen, setPieChartOpen] = useState(false);
	const [chartKey, setChartKey] = useState(0); // State to force chart re-render
	const [balanceData, setBalanceData] = useState<
		{ entry_time: string; account_balance: number }[]
	>([]);
	const [pnlData, setPnlData] = useState<
		{ entry_time: string; cumulative_pnl: number }[]
	>([]);
	const [drawdownData, setDrawdownData] = useState<
		{ entry_time: string; drawdown: number }[]
	>([]);
	const [loadingBalance, setLoadingBalance] = useState(false);
	const [errorBalance, setErrorBalance] = useState<string | null>(null);
	const [loadingPnl, setLoadingPnl] = useState(false);
	const [errorPnl, setErrorPnl] = useState<string | null>(null);
	const [loadingDrawdown, setLoadingDrawdown] = useState(false);
	const [errorDrawdown, setErrorDrawdown] = useState<string | null>(null);
	const [seriesMonthlyPerformanceData, setSeriesMonthlyPerformanceData] =
		useState<ApexAxisChartSeries>([]);
	const [categoriesMonthlyPerformance, setCategoriesMonthlyPerformance] =
		useState<string[]>([]);
	const [seriesMonthlyNetPnL, setSeriesMonthlyNetPnL] =
		useState<ApexAxisChartSeries>([]);
	const [categoriesMonthlyNetPnL, setCategoriesMonthlyNetPnL] = useState<
		string[]
	>([]);
	const [seriesMonthlyRMultiple, setSeriesMonthlyRMultiple] =
		useState<ApexAxisChartSeries>([]);
	const [categoriesMonthlyRMultiple, setCategoriesMonthlyRMultiple] = useState<
		string[]
	>([]);
	const [loadingPnL, setLoadingPnL] = useState(true);
	const [loadingRMultiple, setLoadingRMultiple] = useState(true);
	const [notificationAction, setNotificationAction] = useState<
		"Buy" | "Sell" | "No Action"
	>("No Action");

	// Auto resize with debounce
	useEffect(() => {
		const handleResize = debounce(() => {
			setChartKey((prev) => prev + 1); // Force re-render
			window.dispatchEvent(new Event("resize")); // Notify ApexCharts
		}, 300); // Reduced to 300ms for responsiveness

		return handleResize;
	}, [accountId]); // Empty dependency array to run once on mount

	// Fetch chart data
	useEffect(() => {
		// Fetch Account Balance
		async function fetchBalanceData() {
			if (!accountId) return;
			setLoadingBalance(true);
			setErrorBalance(null);

			try {
				const balanceResponse = await fetch(
					ApiRoute.getAccountBalance(accountId)
				);
				if (!balanceResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch balance data");
				}
				const balanceJson = await balanceResponse.json();
				setBalanceData(balanceJson);
			} catch (err) {
				if (trades.length === 0) return;
				setErrorBalance((err as Error).message);
			} finally {
				setLoadingBalance(false);
			}
		}

		// Fetch Cumulative P&L
		async function fetchPnlData() {
			if (!accountId) return;
			setLoadingPnl(true);
			setErrorPnl(null);

			try {
				const pnlResponse = await fetch(ApiRoute.getCummulativePnL(accountId));
				if (!pnlResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch P&L data");
				}
				const pnlJson = await pnlResponse.json();
				setPnlData(pnlJson);
			} catch (err) {
				if (trades.length === 0) return;
				setErrorPnl((err as Error).message);
			} finally {
				setLoadingPnl(false);
			}
		}

		// Fetch Account Drawdown
		async function fetchDrawdownData() {
			if (!accountId) return;
			setLoadingDrawdown(true);
			setErrorDrawdown(null);

			try {
				const drawdownResponse = await fetch(ApiRoute.getDrawdown(accountId));
				if (!drawdownResponse.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch drawdown data");
				}
				const drawdownJson = await drawdownResponse.json();
				setDrawdownData(drawdownJson);
			} catch (err) {
				if (trades.length === 0) return;
				setErrorDrawdown((err as Error).message);
			} finally {
				setLoadingDrawdown(false);
			}
		}

		async function fetchMonthlyPerformance() {
			if (!accountId) return;
			try {
				const response = await fetch(
					ApiRoute.getMonthlyTradePerformance(accountId)
				);
				if (!response.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch trade data");
				}
				const data = await response.json();
				setSeriesMonthlyPerformanceData(data.series); // Directly use series with colors
				setCategoriesMonthlyPerformance(data.month.split(",")); // Split months into array
			} catch (error) {
				if (trades.length === 0) return;
				console.error("Error fetching monthly performance:", error);
			}
		}

		async function fetchMonthlyNetPnL() {
			if (!accountId) return;
			setLoadingPnL(true);
			try {
				const response = await fetch(
					`/api/monthly-net-pnl?accountId=${accountId}`
				);
				if (!response.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch monthly net P&L data");
				}
				const data = await response.json();
				setSeriesMonthlyNetPnL(data.series);
				setCategoriesMonthlyNetPnL(data.month.split(","));
			} catch (error) {
				if (trades.length === 0) return;
				console.error("Error fetching monthly net P&L:", error);
			} finally {
				setLoadingPnL(false);
			}
		}

		async function fetchMonthlyRMultiple() {
			if (!accountId) return;
			setLoadingRMultiple(true);
			try {
				const response = await fetch(
					`/api/monthly-r-multiple?accountId=${accountId}`
				);
				if (!response.ok) {
					if (trades.length === 0) return;
					throw new Error("Failed to fetch monthly R-Multiple data");
				}
				const data = await response.json();
				setSeriesMonthlyRMultiple(data.series);
				setCategoriesMonthlyRMultiple(data.month.split(","));
			} catch (error) {
				if (trades.length === 0) return;
				console.error("Error fetching monthly R-Multiple:", error);
			} finally {
				setLoadingRMultiple(false);
			}
		}

		fetchBalanceData();
		fetchPnlData();
		fetchDrawdownData();
		fetchMonthlyPerformance();
		fetchMonthlyNetPnL();
		fetchMonthlyRMultiple();
	}, [accountId]);

	// Prepare LineChart data
	const balanceSeries = balanceData.map((d) => d.account_balance);
	const pnlSeries = pnlData.map((d) => d.cumulative_pnl);
	const drawdownSeries = drawdownData.map((d) => d.drawdown);
	const categories = balanceData.map(
		(d) => new Date(d.entry_time).toISOString() // Use full timestamp, e.g., "2025-03-31T10:00:00Z"
	);

	const handleNewTradeAction = useCallback((action: "Buy" | "Sell") => {
		// console.log("Setting notificationAction:", action); // Debug
		setNotificationAction(action);
	}, []);

	const { trades, error } = useTrades(
		undefined,
		undefined,
		handleNewTradeAction
	);

	return (
		<div className="flex flex-col min-h-screen">
			<div className="relative flex-1">
				{/* Trade Notification */}
				{notificationAction !== "No Action" && (
					<TradeNotification action={notificationAction} />
				)}

				<BlockContainer>
					<TradeLineChart />
				</BlockContainer>

				<TradingJournalBlock />

				<BlockContainer>
					<BlockItem className="w-full sm:w-1/3">
						<AccountSummary />
					</BlockItem>

					<BlockItem className="w-full sm:w-1/3">
						<AccountStatistics />
					</BlockItem>

					<BlockItem className="w-full sm:w-1/3">
						{/* Dropdown Menu */}
						<DropdownMenu
							header={selectedChart}
							isOpen={isPieChartOpen}
							setIsOpen={setPieChartOpen}
							theme="white"
							size="large"
							dropIcon={true}
							className="w-full"
							content={
								<div className="bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg shadow-lg w-full">
									<div className="grid">
										<button
											onClick={() => {
												setSelectedChart("Trading Performance");
												setPieChartOpen(false); // Close dropdown after selection
											}}
											className="px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-t-lg transition-colors"
										>
											Trading Performance
										</button>
										<button
											onClick={() => {
												setSelectedChart("Long / Short");
												setPieChartOpen(false); // Close dropdown after selection
											}}
											className="px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-b-lg transition-colors"
										>
											Long / Short
										</button>
									</div>
								</div>
							}
						/>

						{/* Pie Chart */}
						<div className="w-ful flex items-center justify-center">
							{selectedChart === "Trading Performance" ? (
								<TradePerformancePieChart />
							) : (
								<LongShortPieChart />
							)}
						</div>
					</BlockItem>

					<BlockItem className="w-full sm:w-full lg:w-full xl:w-2/3">
						<MonthlyTradeStatsTable />
					</BlockItem>

					<BlockItem>
						{loadingBalance ? (
							<div>Loading charts...</div>
						) : errorBalance ? (
							<div>Error: {errorBalance}</div>
						) : (
							<LineChart
								key={`balance1-${chartKey}`} // Unique key per chart
								height={350}
								title="Account Balance"
								seriesData={balanceSeries}
								categories={categories}
								decimalPlaces={2} // 2 decimals for balance
								lineColor="#057A55" // Green line
								lineStyle="straight" // Straight line
							/>
						)}
					</BlockItem>

					<BlockItem>
						{loadingPnl ? (
							<div>Loading charts...</div>
						) : errorPnl ? (
							<div>Error: {errorPnl}</div>
						) : (
							<LineChart
								key={`pnl1-${chartKey}`} // Unique key per chart
								height={350}
								title="Cumulative P&L"
								seriesData={pnlSeries}
								categories={categories}
								decimalPlaces={2} // 2 decimals for P&L
								lineColor="#FF4560" // Red line
								lineStyle="smooth" // Smooth curve
							/>
						)}
					</BlockItem>

					<BlockItem>
						{loadingDrawdown ? (
							<div>Loading charts...</div>
						) : errorDrawdown ? (
							<div>Error: {errorDrawdown}</div>
						) : (
							<LineChart
								key={`drawdown1-${chartKey}`} // Unique key per chart
								height={350}
								title="Account Drawdown"
								seriesData={drawdownSeries}
								categories={categories}
								decimalPlaces={3} // 3 decimals for drawdown
								lineColor="#4267B2" // Blue line
								lineStyle="smooth" // Step line
							/>
						)}
					</BlockItem>

					{/* Monthly Winning & Losing */}
					<BlockItem>
						<ColumnCharts
							seriesData={seriesMonthlyPerformanceData}
							categories={categoriesMonthlyPerformance}
							height={400}
							columnWidth="60%"
							colors={["#057A55", "#E02424"]}
							yLabel="Trade Count"
						/>
					</BlockItem>

					{/* Monthly Performance (P&L) */}
					<BlockItem>
						{loadingPnL ? (
							<div>Loading P&L chart...</div>
						) : (
							<ColumnCharts
								seriesData={seriesMonthlyNetPnL}
								categories={categoriesMonthlyNetPnL}
								height={400}
								columnWidth="60%"
								dynamicColors={[
									{ from: 0, to: Infinity, color: "#057A55" }, // Green for positive
									{ from: -Infinity, to: 0, color: "#E02424" }, // Red for negative
								]}
								yLabel="Net P&L ($)"
								yFormatter={(val: number) => `$${val.toFixed(2)}`} // Currency format
							/>
						)}
					</BlockItem>

					{/* Monthly Performance (R-Multiple) */}
					<BlockItem>
						{loadingRMultiple ? (
							<div>Loading R-Multiple chart...</div>
						) : (
							<ColumnCharts
								seriesData={seriesMonthlyRMultiple}
								categories={categoriesMonthlyRMultiple}
								height={400}
								columnWidth="60%"
								dynamicColors={[
									{ from: 0, to: Infinity, color: "#057A55" }, // Green for positive R-Multiple
									{ from: -Infinity, to: 0, color: "#E02424" }, // Red for negative R-Multiple
								]}
								yLabel="Avg R-Multiple"
								yFormatter={(val: number) => val.toFixed(2)} // 2 decimal places
							/>
						)}
					</BlockItem>
				</BlockContainer>
			</div>
			{/* Footer */}
			<footer className="bg-gray-800 text-white py-6 px-4 mt-8">
				<div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center">
					<div className="mb-4 sm:mb-0">
						<p className="text-sm">
							&copy; {new Date().getFullYear()} AutoTrade. All rights reserved.
						</p>
					</div>
					<div className="flex space-x-6">
						<a
							href="/pages/backtest_report"
							className="text-sm hover:text-gray-300 transition-colors"
						>
							Backtest Report
						</a>
						<a
							href="/pages/settings"
							className="text-sm hover:text-gray-300 transition-colors"
						>
							Setting
						</a>
						<a
							href="/pages/help"
							className="text-sm hover:text-gray-300 transition-colors"
						>
							Help
						</a>
					</div>
				</div>
			</footer>
		</div>
	);
}
