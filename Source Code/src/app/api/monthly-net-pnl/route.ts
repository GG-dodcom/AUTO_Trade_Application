// src/app/api/monthly-net-pnl/route.ts
import { openDb } from "@/lib/db";
import { monthNames } from "@/lib/lib";
import { NextResponse } from "next/server";

interface MonthlyNetPnLData {
	month: string; // e.g., "2025-03,2025-04"
	series: {
		name: string; // "Net P&L"
		data: number[]; // Net P&L values per month (positive or negative)
	}[];
}

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const accountIdRaw = searchParams.get("accountId");

	if (!accountIdRaw || isNaN(parseInt(accountIdRaw))) {
		return NextResponse.json(
			{ error: "Invalid or missing accountId, must be a number" },
			{ status: 400 }
		);
	}
	const accountId = parseInt(accountIdRaw);

	const db = await openDb();
	try {
		// SQL query to group net P&L by month
		let query = `
      SELECT 
        strftime('%Y-%m', entry_time) as month,
        SUM(profit_loss) as netPnL
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
    `;
		const params: (string | number)[] = [accountId];

		query += " GROUP BY strftime('%Y-%m', entry_time) ORDER BY month ASC";

		// Fetch monthly net P&L data
		const monthlyResults = await db.all(query, params);

		if (!monthlyResults || monthlyResults.length === 0) {
			return NextResponse.json(
				{ error: "No closed trades found for this account" },
				{ status: 404 }
			);
		}

		// Prepare data for the chart
		const categories = monthlyResults.map((row) => {
			const [year, month] = row.month.split("-");
			return `${monthNames[parseInt(month) - 1]} ${year}`;
		});
		const netPnL = monthlyResults.map((row) => row.netPnL || 0);

		const series: ApexAxisChartSeries = [
			{
				name: "Net P&L",
				data: netPnL,
			},
		];

		return NextResponse.json(
			{ series, month: categories.join(",") },
			{ status: 200 }
		);
	} catch (error: any) {
		console.error(
			"Error fetching monthly net P&L:",
			error.message,
			error.stack
		);
		return NextResponse.json(
			{ error: "Internal server error", details: error.message },
			{ status: 500 }
		);
	} finally {
		await db.close();
	}
}
