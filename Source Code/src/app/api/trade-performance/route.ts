// src/app/api/trade-performance/route.ts
import { openDb } from "@/lib/db";
import { TradePerformance } from "@/lib/definitions";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const accountIdRaw = searchParams.get("accountId");
	const startDate =
		searchParams.get("start") === "null" ? null : searchParams.get("start");
	const endDate =
		searchParams.get("end") === "null" ? null : searchParams.get("end");

	if (!accountIdRaw || isNaN(parseInt(accountIdRaw))) {
		return NextResponse.json(
			{ error: "Invalid or missing accountId, must be a number" },
			{ status: 400 }
		);
	}
	const accountId = parseInt(accountIdRaw);

	const db = await openDb();
	try {
		// Build the WHERE clause dynamically based on date range
		let query = `
      SELECT 
        COUNT(CASE WHEN win_loss = 1 THEN 1 END) as winningTrades,
        COUNT(CASE WHEN win_loss = 0 THEN 1 END) as losingTrades,
        COUNT(CASE WHEN win_loss IS NULL OR profit_loss = 0 THEN 1 END) as breakEvenTrades
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
    `;
		const params: (string | number)[] = [accountId];

		if (startDate) {
			query += " AND entry_time >= ?";
			params.push(startDate);
		}
		if (endDate) {
			query += " AND entry_time <= ?";
			params.push(endDate);
		}

		// Fetch trade performance data
		const tradePerformance = await db.get(query, params);

		if (
			!tradePerformance ||
			(tradePerformance.winningTrades === 0 &&
				tradePerformance.losingTrades === 0 &&
				tradePerformance.breakEvenTrades === 0)
		) {
			return NextResponse.json(
				{ error: "No closed trades found for this account" },
				{ status: 404 }
			);
		}

		const stats: TradePerformance = {
			winningTrades: tradePerformance.winningTrades || 0,
			losingTrades: tradePerformance.losingTrades || 0,
			breakEvenTrades: tradePerformance.breakEvenTrades || 0,
		};

		return NextResponse.json(stats, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching trade performance:",
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
