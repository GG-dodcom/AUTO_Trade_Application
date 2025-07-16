"use client";
import { DropdownMenu } from "@/components/DropdownMenu";
import { useTrades } from "@/hook/useTrades";
import { useState } from "react";
import { Calendar } from "@/assets/icons";
import { StickyRowTable } from "@/components/StickyRowTable";
import { JournalColumns, journalHeader1Groups } from "@/data/tableColumns";
import { Pagination } from "@/components/Pagination";
import BlockContainer from "@/components/BlockContainer";
import { DateRangePicker, RangeKeyDict } from "react-date-range";
import { format, parse, startOfDay, endOfDay } from "date-fns";

export default function TradingJournalPage() {
	const [currentPage, setCurrentPage] = useState(1);
	const [dateRange, setDateRange] = useState({
		startDate: startOfDay(new Date()),
		endDate: endOfDay(new Date()),
		key: "selection",
	});
	const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);

	const { trades, error } = useTrades(dateRange.startDate, dateRange.endDate);

	const itemsPerPage = 30;

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

	const handleDateRangeChange = (ranges: RangeKeyDict) => {
		if (ranges.selection.startDate && ranges.selection.endDate) {
			setDateRange({
				startDate: ranges.selection.startDate,
				endDate: ranges.selection.endDate,
				key: "selection",
			});
			setCurrentPage(1);
		}
	};

	const formatDate = (date: Date) => format(date, "MMM d, yyyy");
	const dateRangeLabel =
		dateRange.startDate && dateRange.endDate
			? `${formatDate(dateRange.startDate)} - ${formatDate(dateRange.endDate)}`
			: "Select Date Range";

	if (error) return <p className="text-red-500 text-center py-4">{error}</p>;

	return (
		<BlockContainer className="bg-white shadow-sm rounded-lg overflow-hidden max-h-screen">
			<div className="sticky top-0 z-20 w-[-webkit-fill-available] bg-white border-b border-gray-200 p-4">
				<div className="grid grid-cols-1 sm:grid-cols-[auto_auto_1fr] items-center gap-4">
					<h1 className="text-2xl font-bold text-gray-900">Trading Journal</h1>
					<DropdownMenu
						leftIcon={<Calendar className="w-4 h-4" />}
						header={dateRangeLabel}
						content={
							<div className="bg-white border rounded-lg shadow-lg p-4 max-w-[600px] inline-block">
								<DateRangePicker
									ranges={[dateRange]}
									onChange={handleDateRangeChange}
									showDateDisplay={false}
									direction="vertical"
									months={1}
									maxDate={new Date()}
									className="w-max"
								/>
							</div>
						}
						isOpen={isDatePickerOpen}
						setIsOpen={setIsDatePickerOpen}
						theme="white"
					/>
					<Pagination
						currentPage={currentPage}
						itemsPerPage={itemsPerPage}
						totalItems={totalItems}
						onPageChange={handlePageChange}
						className="justify-self-end"
					/>
				</div>
			</div>
			<div className="overflow-y-auto max-h-[calc(100vh-200px)] h-[100vh]">
				{trades.length > 0 ? (
					<StickyRowTable
						data={paginatedData}
						columns={JournalColumns}
						header1Groups={journalHeader1Groups}
						className="h-[100%]"
					/>
				) : (
					<p className="text-gray-500 text-center py-4 w-[100vh]">
						No trades found for this date range.
					</p>
				)}
			</div>
		</BlockContainer>
	);
}
