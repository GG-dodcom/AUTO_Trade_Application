// src/app/api/cummulative-pnl/route.ts
import { openDb } from "@/lib/db";
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
		let query = `
      SELECT entry_time, cumulative_pnl
      FROM trades
      WHERE accountId = ? AND status = 'Closed' AND cumulative_pnl IS NOT NULL
    `;
		const params: (string | number)[] = [accountId];

		if (startDate) {
			query += " AND entry_time >= ?";
			params.push(startDate);
		}
		if (endDate) {
			query += " AND entry_time <= ?";
			params.push(endDate); // Include full end day
		}

		query += " ORDER BY entry_time ASC";

		const pnls = await db.all(query, params);

		if (!pnls || pnls.length === 0) {
			return NextResponse.json(
				{
					error:
						"No closed trades with cumulative P&L data found for this account",
				},
				{ status: 404 }
			);
		}

		return NextResponse.json(pnls, { status: 200 });
	} catch (error: any) {
		console.error("Error fetching cumulative P&L:", error.message, error.stack);
		return NextResponse.json(
			{ error: "Internal server error", details: error.message },
			{ status: 500 }
		);
	} finally {
		await db.close();
	}
}
