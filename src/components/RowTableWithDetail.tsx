// src/components/RowTableWithDetail.tsx
import React, { useState } from "react";
import Badge from "./Badge";
import { ColumnConfig } from "@/lib/definitions";
import { ChevronDown, ChevronRight } from "@/assets/icons";

interface Header1Config<T> {
	header: string;
	keys: string[];
}

interface TableProps<T> {
	data: T[];
	columns: ColumnConfig<T>[];
	header1Groups?: Header1Config<T>[];
	cellTheme?: string;
	className?: string;
}

export const RowTableWithDetail = <T,>({
	data,
	columns,
	header1Groups,
	cellTheme,
	className,
}: TableProps<T>) => {
	const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

	const borderColumns = (() => {
		if (!header1Groups) return [];
		const firstHeaderSpans = header1Groups.map((group) => group.keys.length);
		const columns = firstHeaderSpans.reduce<number[]>((acc, span) => {
			const colIndex = acc.length > 0 ? acc[acc.length - 1] + span : span;
			acc.push(colIndex);
			return acc;
		}, []);
		return columns.slice(0, -1);
	})();

	const toggleRow = (rowNo: number) => {
		const newExpandedRows = new Set(expandedRows);
		if (newExpandedRows.has(rowNo)) {
			newExpandedRows.delete(rowNo);
		} else {
			newExpandedRows.add(rowNo);
		}
		setExpandedRows(newExpandedRows);
	};

	// Mapping function for impact values
	const getImpactText = (impact: number) => {
		const impactMap: { [key: number]: string } = {
			0: "Low Impact Expected",
			1: "Medium Impact Expected",
			2: "High Impact Expected",
		};
		return impactMap[impact] || "Unknown Impact"; // Fallback for unexpected values
	};

	return (
		<div
			className={`custom-scrollbar relative overflow-x-auto sm:rounded-lg ${className}`}
		>
			<table className="w-full text-sm text-left text-gray-500">
				<thead className="text-xs text-gray-700 uppercase bg-gray-50">
					{header1Groups && (
						<tr>
							<th className="w-12 border-b border-gray-400" />
							{/* Toggle column */}
							{header1Groups.map((group, index) => (
								<th
									key={group.header}
									colSpan={group.keys.length}
									className={`px-4 py-2 text-center border-b border-gray-400 ${
										borderColumns.includes(
											(borderColumns[index - 1] || 0) + group.keys.length
										)
											? "border-r border-gray-400"
											: ""
									}`}
								>
									{group.header}
								</th>
							))}
						</tr>
					)}
					<tr>
						{/* Toggle column */}
						<th className="w-12" />
						{columns.map((col, colIndex) => (
							<th
								key={String(col.key)}
								className={`px-4 py-2 ${
									borderColumns.includes(colIndex + 1)
										? "border-r border-gray-400"
										: ""
								} ${
									col.width || ""
								} whitespace-nowrap overflow-hidden text-ellipsis`}
							>
								{col.label}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{data.map((row: any, rowIndex) => (
						<React.Fragment key={rowIndex}>
							<tr className="bg-white border-b border-t hover:bg-gray-50 transition-colors">
								<td className="px-4 py-2 whitespace-nowrap">
									<button
										onClick={() => toggleRow(row.no)}
										className="focus:outline-none"
									>
										{expandedRows.has(row.no) ? (
											<ChevronDown className="w-5 h-5 text-gray-500" />
										) : (
											<ChevronRight className="w-5 h-5 text-gray-500" />
										)}
									</button>
								</td>
								{columns.map((col, colIndex) => {
									const value = row[col.key];
									const formattedValue =
										col.isMoney && typeof value === "number"
											? `$ ${value.toLocaleString()}`
											: col.isPercentage
											? `${value}%`
											: value;
									const textColor =
										col.enableNumberColor && typeof value === "number"
											? value < 0
												? "text-red-600"
												: value > 0
												? "text-green-600"
												: "text-gray-600"
											: col.colorTheme
											? col.colorTheme(value)
											: "";

									return (
										<td
											key={String(col.key)}
											className={`px-4 py-2 whitespace-nowrap overflow-hidden text-ellipsis ${
												borderColumns.includes(colIndex + 1)
													? "border-r border-gray-400"
													: ""
											} ${col.width || ""} ${
												col.rowHeader ? "text-gray-900 font-medium" : ""
											} ${textColor} ${col.cellTheme && col.cellTheme(value)} ${
												cellTheme || ""
											}`}
										>
											{col.render ? (
												col.render(value, row)
											) : col.badge ? (
												<Badge
													text={value}
													theme={
														col.badgeTheme ? col.badgeTheme(value) : "Gray"
													}
												/>
											) : (
												(formattedValue as React.ReactNode)
											)}
										</td>
									);
								})}
							</tr>
							{expandedRows.has(row.no) && (
								<tr>
									<td colSpan={columns.length + 1} className="px-4 py-2">
										<div className="grid grid-cols-1 gap-4">
											{/* Technical Indicators Table */}
											<div>
												<h3 className="text-sm text-gray-700 mb-2">
													Technical Indicators
												</h3>
												<div className="overflow-x-auto">
													<table className="w-auto divide-y divide-gray-200">
														<thead className="bg-gray-50">
															<tr>
																{row.technical_indicators &&
																row.technical_indicators.length > 0 ? (
																	row.technical_indicators.map((ti: any) => (
																		<th
																			key={ti.indicator_name}
																			className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap"
																		>
																			{ti.indicator_name}
																		</th>
																	))
																) : (
																	<th className="px-4 py-2 text-left text-xs font-medium text-gray-500">
																		No Indicators
																	</th>
																)}
															</tr>
														</thead>
														<tbody className="bg-white divide-y divide-gray-200">
															<tr>
																{row.technical_indicators &&
																row.technical_indicators.length > 0 ? (
																	row.technical_indicators.map((ti: any) => (
																		<td
																			key={ti.indicator_name}
																			className="px-4 py-2 whitespace-nowrap text-sm text-gray-900"
																		>
																			{ti.indicator_value}
																		</td>
																	))
																) : (
																	<td className="px-4 py-2 text-sm text-gray-500">
																		No data available
																	</td>
																)}
															</tr>
														</tbody>
													</table>
												</div>
											</div>
											{/* Economic Calendar Table */}
											<div>
												<h3 className="text-sm text-gray-700 mb-2">
													Economic Calendar
												</h3>
												<table className="w-auto divide-y divide-gray-200">
													<thead className="bg-gray-50">
														<tr>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Event
															</th>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Currency
															</th>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Impact
															</th>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Forecast
															</th>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Previous
															</th>
															<th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
																Actual
															</th>
														</tr>
													</thead>
													<tbody className="bg-white divide-y divide-gray-200">
														{row.economic_calendar &&
														row.economic_calendar.length > 0 ? (
															row.economic_calendar.map(
																(ec: any, index: number) => (
																	<tr key={index}>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{ec.event}
																		</td>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{ec.event_currency}
																		</td>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{getImpactText(ec.impact)}
																		</td>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{ec.actual}
																		</td>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{ec.forecast}
																		</td>
																		<td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
																			{ec.previous}
																		</td>
																	</tr>
																)
															)
														) : (
															<tr>
																<td
																	colSpan={6}
																	className="px-4 py-2 text-sm text-gray-500 text-center"
																>
																	No events available
																</td>
															</tr>
														)}
													</tbody>
												</table>
											</div>
										</div>
									</td>
								</tr>
							)}
						</React.Fragment>
					))}
				</tbody>
			</table>
		</div>
	);
};
