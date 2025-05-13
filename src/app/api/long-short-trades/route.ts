// src/app/api/long-short-trades/route.ts
import { openDb } from "@/lib/db";
import { LongShortTrades } from "@/lib/definitions";
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
		// Fetch long/short trade data
		const tradeData = await db.get(
			`
      SELECT 
        COUNT(CASE WHEN direction = 0 THEN 1 END) as longTrades,
        COUNT(CASE WHEN direction = 1 THEN 1 END) as shortTrades
      FROM trades 
      WHERE accountId = ? AND status = 'Closed'
    `,
			[accountId]
		);

		if (
			!tradeData ||
			(tradeData.longTrades === 0 && tradeData.shortTrades === 0)
		) {
			return NextResponse.json(
				{ error: "No closed trades found for this account" },
				{ status: 404 }
			);
		}

		const stats: LongShortTrades = {
			longTrades: tradeData.longTrades || 0,
			shortTrades: tradeData.shortTrades || 0,
		};

		return NextResponse.json(stats, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching long/short trades:",
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
