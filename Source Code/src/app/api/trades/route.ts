import { openDb } from "@/lib/db";
import { Trade } from "@/lib/definitions";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const accountId = searchParams.get("accountId"); // Expect accountId directly
	const startDate = searchParams.get("start_date"); // e.g., "2025-03-30T16:00:00.000Z"
	const endDate = searchParams.get("end_date"); // e.g., "2025-03-31T16:00:00.000Z"

	let db: any;
	try {
		db = await openDb(); // Open database connection

		// Step 1: Validate required parameter
		if (!accountId) {
			return NextResponse.json({ error: "Missing accountId" }, { status: 400 });
		}

		// Step 2: Base query for trades using accountId directly
		let tradeQuery = `SELECT * FROM trades WHERE accountId = ?`;
		const params = [accountId];

		if (startDate && endDate) {
			// Convert ISO strings to SQLite-friendly format if needed (e.g., "2025-03-30 16:00:00")
			const sqliteStartDate = startDate.replace("T", " ").replace("Z", "");
			const sqliteEndDate = endDate.replace("T", " ").replace("Z", "");
			tradeQuery += " AND entry_time BETWEEN ? AND ?";
			params.push(sqliteStartDate, sqliteEndDate);
		}

		tradeQuery += " ORDER BY entry_time DESC";

		// Step 3: Fetch trades
		const trades = await db.all(tradeQuery, params);

		// Step 4: Fetch related technical indicators and economic calendar for each trade
		const result = await Promise.all(
			trades.map(async (trade: Trade) => {
				// Technical Indicators
				const technicalIndicators = await db.all(
					`
						SELECT indicator_name, indicator_value
						FROM technical_indicators
						WHERE trade_ticket = ?
					`,
					[trade.ticket]
				);

				// Economic Calendar
				const economicCalendar = await db.all(
					`
						SELECT event, event_currency, actual, forecast, previous, impact
						FROM economic_calendar
						WHERE trade_ticket = ?
					`,
					[trade.ticket]
				);
				// console.log(
				// 	"Technical Indicator: \n",
				// 	technicalIndicators,
				// 	"Economic Calendar: \n",
				// 	economicCalendar
				// );

				return {
					...trade,
					technical_indicators: technicalIndicators,
					economic_calendar: economicCalendar,
				};
			})
		);

		return NextResponse.json(result);
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
