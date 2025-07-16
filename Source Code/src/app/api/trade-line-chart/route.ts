import { NextRequest, NextResponse } from "next/server";
import { openDb } from "@/lib/db";
import { ChartTrade } from "@/lib/definitions";

export async function GET(request: NextRequest) {
	const { searchParams } = new URL(request.url);
	const accountId = searchParams.get("accountId");

	let db;
	try {
		db = await openDb();

		if (!accountId) {
			return NextResponse.json({ error: "Missing accountId" }, { status: 400 });
		}

		const trades: ChartTrade[] = await db.all(
			`
      SELECT ticket, entry_time, exit_time, entry_price, exit_price, direction, profit_loss
      FROM trades
      WHERE accountId = ?
      ORDER BY entry_time ASC
      `,
			[accountId]
		);

		return NextResponse.json(trades, { status: 200 });
	} catch (error) {
		console.error("API error:", error);
		return NextResponse.json(
			{ error: "Failed to fetch trades" },
			{ status: 500 }
		);
	} finally {
		if (db) await db.close();
	}
}
