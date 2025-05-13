import React from "react";
import Badge from "./Badge";

interface ColumnConfig<T> {
	label: string;
	key: keyof T;
	width?: string;
	badge?: boolean;
	badgeTheme?: (
		value: T[keyof T]
	) =>
		| "Gray"
		| "Red"
		| "Yellow"
		| "Green"
		| "Blue"
		| "Indigo"
		| "Purple"
		| "Pink";
	isMoney: boolean; // Will add "$" to the value
	enableNumberColor: boolean; // Enables red/green text color for numbers
	colorTheme: any;
	isPercentage: boolean; // Will format the value as a percentage (e.g., "50%")
	cellTheme?: any;
	render?: (value: T[keyof T], row: T) => React.ReactNode;
}

interface TableProps<T> {
	data: T[];
	columns: ColumnConfig<T>[];
	header?: string;
	headerTheme?: string;
	headerCellTheme?: string;
	cellTheme?: string;
}

export const ColumnarTable = <T,>({
	data,
	columns,
	header,
	headerTheme,
	headerCellTheme,
	cellTheme,
}: TableProps<T>) => {
	return (
		<div className="custom-scrollbar relative overflow-x-auto sm:rounded-lg">
			<table className="w-full text-sm text-left text-gray-500">
				{/* Table Header */}
				{header && (
					<thead>
						<tr className="bg-gray-50 text-gray-700">
							<th
								colSpan={columns.length}
								className={`px-6 py-3 font-bold whitespace-nowrap text-center ${headerTheme}`}
							>
								{header}
							</th>
						</tr>
					</thead>
				)}

				<tbody>
					{columns.map((col, colIndex) => (
						<tr key={String(col.key)} className="bg-white border-b">
							<th
								className={`px-6 py-3 text-gray-700 bg-gray-50 font-bold whitespace-nowrap
									${headerCellTheme || ""}	
								`}
							>
								{col.label}
							</th>
							{data.map((row, rowIndex) => {
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
										key={rowIndex}
										className={`px-6 py-3
											${col.width || ""} 
											whitespace-nowrap overflow-hidden text-ellipsis 
											${textColor} 
											${col.cellTheme && col.cellTheme(value)}
											${cellTheme || ""}`}
									>
										{col.render ? (
											col.render(value, row)
										) : col.badge ? (
											<Badge
												text={value}
												theme={col.badgeTheme ? col.badgeTheme(value) : "Gray"}
											/>
										) : (
											(formattedValue as React.ReactNode)
										)}
									</td>
								);
							})}
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
};
