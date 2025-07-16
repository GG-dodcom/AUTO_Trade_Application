import React from "react";
import Badge from "./Badge";
import { ColumnConfig } from "@/lib/definitions";

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

export const RowTable = <T,>({
	data,
	columns,
	header1Groups,
	cellTheme,
	className,
}: TableProps<T>) => {
	const borderColumns = (() => {
		if (!header1Groups) return [];

		// Compute column spans
		const firstHeaderSpans = header1Groups.map((group) => group.keys.length);

		// Find the column indexes where a border-r should be added
		const columns = firstHeaderSpans.reduce<number[]>((acc, span) => {
			const colIndex = acc.length > 0 ? acc[acc.length - 1] + span : span;
			acc.push(colIndex);
			return acc;
		}, []);

		return columns.slice(0, -1); // Remove the last element
	})();

	return (
		<div
			className={`custom-scrollbar relative overflow-x-auto sm:rounded-lg ${className}`}
		>
			<table className="w-full text-sm text-left text-gray-500">
				<thead className="text-xs text-gray-700 uppercase bg-gray-50">
					{/* First Row of Header */}
					{header1Groups && (
						<tr>
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

					{/* Second Row of Header */}
					<tr>
						{columns.map((col, colIndex) => (
							<th
								key={String(col.key)}
								className={`px-4 py-2 
                  ${
										borderColumns.includes(colIndex + 1)
											? "border-r border-gray-400"
											: ""
									}
                  ${col.width || ""}
                  whitespace-nowrap overflow-hidden text-ellipsis`}
							>
								{col.label}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{Array.isArray(data) && data.length > 0 ? (
						data.map((row, rowIndex) => (
							<tr
								key={rowIndex}
								className="bg-white border-b hover:bg-gray-50 transition-colors"
							>
								{columns.map((col, colIndex) => {
									const value = row[col.key];

									// Number formatting: Add $ prefix if configured
									const formattedValue =
										col.isMoney && typeof value === "number"
											? `$ ${value.toLocaleString()}`
											: col.isPercentage
											? `${value}%`
											: value;

									// Color logic: Enable only if `enableNumberColor` is set
									const textColor =
										col.enableNumberColor && typeof value === "number"
											? value < 0
												? "text-red-600"
												: value > 0
												? "text-green-600"
												: "text-gray-600"
											: col.colorTheme
											? col.colorTheme(value) // Custom colors for specific values
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
						))
					) : (
						<tr>
							<td colSpan={columns.length} className="px-4 py-2 text-center">
								No data available
							</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
};
