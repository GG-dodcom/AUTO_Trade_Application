// src/components/TradingJournalBlock.tsx
import React, { useState } from "react";
import { RowTableWithDetail } from "@/components/RowTableWithDetail";
import { Pagination } from "@/components/Pagination";
import { DropdownMenu } from "@/components/DropdownMenu";
import { JournalColumns, journalHeader1Groups } from "@/data/tableColumns";
import { Calendar } from "@/assets/icons";
import {
	format,
	startOfDay,
	endOfDay,
	subDays,
	startOfWeek,
	endOfWeek,
	subWeeks,
	startOfMonth,
	endOfMonth,
	subMonths,
} from "date-fns";
import { toZonedTime } from "date-fns-tz";
import { useTrades } from "@/hook/useTrades";
import BlockContainer from "./BlockContainer";

export default function TradingJournalBlock() {
	const TIMEZONE = process.env.NEXT_PUBLIC_TIMEZONE || "Etc/GMT-3"; // Default to GMT+3 if not set
	const [currentPage, setCurrentPage] = useState(1);
	const [dateRange, setDateRange] = useState({
		startDate: startOfDay(toZonedTime(new Date(), TIMEZONE)), // GMT+3
		endDate: endOfDay(toZonedTime(new Date(), TIMEZONE)),
		key: "selection",
	});
	const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);

	const { trades, error } = useTrades(dateRange.startDate, dateRange.endDate);

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

	// Date range button handlers
	const setToday = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		setDateRange({
			startDate: startOfDay(today),
			endDate: endOfDay(today),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const setYesterday = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		const yesterday = subDays(today, 1);
		setDateRange({
			startDate: startOfDay(yesterday),
			endDate: endOfDay(yesterday),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const setThisWeek = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		setDateRange({
			startDate: startOfWeek(today, { weekStartsOn: 1 }), // Monday start
			endDate: endOfWeek(today, { weekStartsOn: 1 }),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const setLastWeek = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		const lastWeek = subWeeks(today, 1);
		setDateRange({
			startDate: startOfWeek(lastWeek, { weekStartsOn: 1 }),
			endDate: endOfWeek(lastWeek, { weekStartsOn: 1 }),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const setThisMonth = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		setDateRange({
			startDate: startOfMonth(today),
			endDate: endOfMonth(today),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const setLastMonth = () => {
		const today = toZonedTime(new Date(), TIMEZONE); // GMT+3
		const lastMonth = subMonths(today, 1);
		setDateRange({
			startDate: startOfMonth(lastMonth),
			endDate: endOfMonth(lastMonth),
			key: "selection",
		});
		setIsDatePickerOpen(false);
		setCurrentPage(1);
	};

	const formatDate = (date: Date) => format(date, "MMM d, yyyy");
	const dateRangeLabel =
		dateRange.startDate && dateRange.endDate
			? `${formatDate(dateRange.startDate)} - ${formatDate(dateRange.endDate)}`
			: "Select Date Range";

	if (error) return <p className="text-red-500 text-center py-4">{error}</p>;

	return (
		<BlockContainer>
			<div className="w-full rounded-lg shadow-sm p-4 bg-white">
				<div className="flex justify-between items-center px-4 pb-4">
					<h2 className="text-lg font-semibold">Trading Journal</h2>
					<DropdownMenu
						leftIcon={<Calendar className="w-4 h-4" />}
						header={dateRangeLabel}
						content={
							<div className="bg-white border rounded-lg shadow-lg w-auto inline-block">
								<div className="grid">
									<button
										onClick={setToday}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										Today
									</button>
									<button
										onClick={setYesterday}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										Yesterday
									</button>
									<button
										onClick={setThisWeek}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										This Week
									</button>
									<button
										onClick={setLastWeek}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										Last Week
									</button>
									<button
										onClick={setThisMonth}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										This Month
									</button>
									<button
										onClick={setLastMonth}
										className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded hover:bg-gray-200 transition-colors"
									>
										Last Month
									</button>
								</div>
							</div>
						}
						isOpen={isDatePickerOpen}
						setIsOpen={setIsDatePickerOpen}
						theme="white"
					/>
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
	);
}
