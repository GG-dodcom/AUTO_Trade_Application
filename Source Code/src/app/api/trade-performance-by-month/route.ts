// src/app/api/trade-performance-by-month/route.ts
import { openDb } from "@/lib/db";
import { monthNames } from "@/lib/lib";
import { NextResponse } from "next/server";

interface MonthlyTradeData {
	month: string; // e.g., "2025-03"
	series: {
		name: string; // "Winning Trades" or "Losing Trades"
		data: number[]; // Array of counts for each month
	}[];
}

// Helper function to format date as "YYYY-MM"
const formatMonth = (date: string) => {
	return new Date(date).toISOString().slice(0, 7); // e.g., "2025-03"
};

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
		// Build the SQL query to group trades by month
		let query = `
      SELECT 
        strftime('%Y-%m', entry_time) as month,
        COUNT(CASE WHEN win_loss = 1 THEN 1 END) as winningTrades,
        COUNT(CASE WHEN win_loss = 0 THEN 1 END) as losingTrades
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
    `;
		const params: (string | number)[] = [accountId];

		query += " GROUP BY strftime('%Y-%m', entry_time) ORDER BY month ASC";

		// Fetch monthly trade data
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
		const winningTrades = monthlyResults.map((row) => row.winningTrades || 0);
		const losingTrades = monthlyResults.map((row) => row.losingTrades || 0);

		const response: MonthlyTradeData = {
			month: categories.join(","), // Comma-separated months for reference
			series: [
				{
					name: "Winning Trades",
					data: winningTrades,
				},
				{
					name: "Losing Trades",
					data: losingTrades,
				},
			],
		};

		return NextResponse.json(response, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching monthly trade performance:",
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
