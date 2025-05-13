// src/app/api/monthly-trade-stats/route.ts
import { openDb } from "@/lib/db";
import { NextResponse } from "next/server";
import { MonthlyTradeStats } from "@/lib/definitions";

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
		const monthlyStats = await db.all(
			`
      SELECT 
        STRFTIME('%m', exit_time) as monthNum,
        SUM(profit_loss) as profit,
        AVG(CASE WHEN win_loss IS NOT NULL THEN r_multiple END) as rMultiple,
        COUNT(CASE WHEN win_loss = 1 THEN 1 END) as winningTrades,
        COUNT(CASE WHEN win_loss = 0 THEN 1 END) as losingTrades,
        COUNT(CASE WHEN profit_loss = 0 THEN 1 END) as breakEvenTrades
      FROM trades 
      WHERE accountId = ? AND status = 'Closed' AND exit_time IS NOT NULL
      GROUP BY STRFTIME('%m', exit_time)
      ORDER BY STRFTIME('%m', exit_time)
    `,
			[accountId]
		);

		if (!monthlyStats || monthlyStats.length === 0) {
			return NextResponse.json(
				{ error: "No closed trades found for this account" },
				{ status: 404 }
			);
		}

		// Map month numbers to names and format data
		const monthNames = [
			"Jan",
			"Feb",
			"Mar",
			"Apr",
			"May",
			"Jun",
			"Jul",
			"Aug",
			"Sep",
			"Oct",
			"Nov",
			"Dec",
		];
		const stats: MonthlyTradeStats[] = monthlyStats.map((row) => ({
			month: monthNames[parseInt(row.monthNum) - 1],
			profit: row.profit || 0,
			rMultiple:
				row.rMultiple !== null ? parseFloat(row.rMultiple.toFixed(1)) : 0,
			winningTrades: row.winningTrades || 0,
			losingTrades: row.losingTrades || 0,
			breakEvenTrades: row.breakEvenTrades || 0,
		}));

		// Calculate totals
		const totalStats: MonthlyTradeStats = {
			month: "Total",
			profit: stats.reduce((sum, row) => sum + row.profit, 0),
			rMultiple: parseFloat(
				stats.reduce((sum, row) => sum + row.rMultiple, 0).toFixed(2)
			),
			winningTrades: stats.reduce((sum, row) => sum + row.winningTrades, 0),
			losingTrades: stats.reduce((sum, row) => sum + row.losingTrades, 0),
			breakEvenTrades: stats.reduce((sum, row) => sum + row.breakEvenTrades, 0),
		};

		// Append totals to the result
		stats.push(totalStats);

		return NextResponse.json(stats, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching monthly trade stats:",
			error.message,
			error.stack
		);
		return NextResponse.json(
			// { error: "Internal server error", details: error.message },
			{ status: 500 }
		);
	} finally {
		await db.close();
	}
}
