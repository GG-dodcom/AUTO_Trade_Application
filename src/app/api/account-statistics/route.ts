// src/app/api/account-statistics/route.ts
import { openDb } from "@/lib/db";
import { AccountStatistics } from "@/lib/definitions";
import { NextResponse } from "next/server";

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
		// Fetch trade statistics
		const tradeStats = await db.get(
			`
      SELECT 
        AVG(CASE WHEN win_loss = 1 THEN profit_loss END) as averageWin,
        AVG(CASE WHEN win_loss = 0 THEN profit_loss END) as averageLoss,
        MAX(CASE WHEN win_loss = 1 THEN profit_loss END) as largestWin,
        MIN(CASE WHEN win_loss = 0 THEN profit_loss END) as largestLoss,
        COUNT(CASE WHEN direction = 0 THEN 1 END) as longTrades,
        COUNT(CASE WHEN direction = 1 THEN 1 END) as shortTrades,
        MIN(drawdown) as maxDrawdown,
        AVG(
          CAST(SUBSTR(duration, 1, INSTR(duration, ':') - 1) AS INTEGER) * 60 +
          CAST(SUBSTR(duration, INSTR(duration, ':') + 1) AS INTEGER)
        ) as avgTimeInMinutes
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
    `,
			[accountId]
		);

		if (!tradeStats) {
			return NextResponse.json(
				{ error: "No closed trades found for this account" },
				{ status: 404 }
			);
		}

		// Calculate consecutive wins/losses separately
		const trades = await db.all(
			`
      SELECT win_loss
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
      ORDER BY exit_time ASC
    `,
			[accountId]
		);

		let maxConsecutiveWins = 0;
		let maxConsecutiveLosses = 0;
		let currentWins = 0;
		let currentLosses = 0;

		for (const trade of trades) {
			if (trade.win_loss === 1) {
				currentWins++;
				currentLosses = 0;
				maxConsecutiveWins = Math.max(maxConsecutiveWins, currentWins);
			} else if (trade.win_loss === 0) {
				currentLosses++;
				currentWins = 0;
				maxConsecutiveLosses = Math.max(maxConsecutiveLosses, currentLosses);
			}
		}

		// Convert avgTimeInMinutes to "Xh Ym" format
		const avgTimeInMinutes = tradeStats.avgTimeInMinutes || 0;
		const hours = Math.floor(avgTimeInMinutes / 60);
		const minutes = Math.round(avgTimeInMinutes % 60);
		const avgTimeInTrade = `${hours}h ${minutes}m`;

		const stats: AccountStatistics = {
			averageWin: tradeStats.averageWin || 0,
			averageLoss: tradeStats.averageLoss || 0,
			largestWin: tradeStats.largestWin || 0,
			largestLoss: tradeStats.largestLoss || 0,
			maxConsecutiveWins,
			maxConsecutiveLosses,
			maxDrawdown:
				tradeStats.maxDrawdown < 0
					? tradeStats.maxDrawdown
					: -(tradeStats.maxDrawdown || 0), // Ensure negative
			longTrades: tradeStats.longTrades || 0,
			shortTrades: tradeStats.shortTrades || 0,
			avgTimeInTrade,
		};

		return NextResponse.json(stats, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching account statistics:",
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
