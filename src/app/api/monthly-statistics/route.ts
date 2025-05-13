// src/app/api/monthly-statistics/route.ts
import { openDb } from "@/lib/db";
import { MonthlyStatistics } from "@/lib/definitions";
import { NextRequest, NextResponse } from "next/server";

// Helper function to parse month string to SQLite date format
function parseMonthToDate(month: string): string {
	const [monthName, year] = month.split(" ");
	const monthNum = new Date(`${monthName} 1, ${year}`).getMonth() + 1;
	return `${year}-${monthNum.toString().padStart(2, "0")}`;
}

// GET handler
export async function GET(request: NextRequest) {
	const { searchParams } = new URL(request.url);
	const selectedMonth = searchParams.get("month");
	const accountId = searchParams.get("accountId");

	if (!accountId) {
		return NextResponse.json(
			{ error: "Missing accountId query parameter" },
			{ status: 400 }
		);
	}

	if (!selectedMonth) {
		return NextResponse.json(
			{ error: "Month parameter is required" },
			{ status: 400 }
		);
	}

	let db;
	try {
		db = await openDb();

		// Get distinct months from trades
		const months = await db.all(
			`
        SELECT DISTINCT
          CASE strftime('%m', entry_time)
              WHEN '01' THEN 'January'
              WHEN '02' THEN 'February'
              WHEN '03' THEN 'March'
              WHEN '04' THEN 'April'
              WHEN '05' THEN 'May'
              WHEN '06' THEN 'June'
              WHEN '07' THEN 'July'
              WHEN '08' THEN 'August'
              WHEN '09' THEN 'September'
              WHEN '10' THEN 'October'
              WHEN '11' THEN 'November'
              WHEN '12' THEN 'December'
              ELSE 'Unknown'
          END || ' ' || strftime('%Y', entry_time) as month
        FROM trades
        WHERE accountId = ? AND entry_time IS NOT NULL
        ORDER BY entry_time DESC;
      `,
			[accountId]
		);
		const monthList = months.map((row: any) => row.month);

		// Calculate statistics for the selected month
		const monthStart = `${parseMonthToDate(selectedMonth)}-01`;
		const monthEnd = new Date(
			new Date(monthStart).setMonth(new Date(monthStart).getMonth() + 1)
		)
			.toISOString()
			.split("T")[0];

		// Query for monthly statistics with PreviousMonth, CurrentMonth, and LastTrade CTEs
		const stats = (await db.get(
			`
      WITH PreviousMonth AS (
        SELECT account_balance as prevMonthEndBalance
        FROM trades
        WHERE accountId = ?
          AND entry_time < ?
          AND status = 'Closed'
        ORDER BY entry_time DESC
        LIMIT 1
      ),
      CurrentMonth AS (
      SELECT 
        SUM(profit_loss) as profitLoss,
          (SUM(profit_loss) / COALESCE((SELECT prevMonthEndBalance FROM PreviousMonth), (SELECT balance FROM account WHERE id = ?))) * 100 as percentGainLoss,
        (SUM(CASE WHEN win_loss = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(CASE WHEN win_loss IS NOT NULL THEN 1 END)) as winningRate,
        SUM(r_multiple) as totalRMultiple,
        AVG(r_multiple) as expectancyPerTrade,
        AVG(
          CASE 
            WHEN duration LIKE '%:%' THEN 
              CAST(SUBSTR(duration, 1, INSTR(duration, ':') - 1) AS REAL) * 60 + 
              CAST(SUBSTR(duration, INSTR(duration, ':') + 1) AS REAL)
            ELSE 0 
          END
        ) as avgTimeInTradeMins
      FROM trades
        WHERE accountId = ?
          AND entry_time >= ? 
          AND entry_time < ?
          AND status = 'Closed'
      ),
      LastTrade AS (
        SELECT account_balance as accountEndingBalance
        FROM trades
        WHERE accountId = ?
          AND entry_time >= ? 
          AND entry_time < ?
          AND status = 'Closed'
        ORDER BY entry_time DESC
        LIMIT 1
      )
      SELECT 
        COALESCE((SELECT prevMonthEndBalance FROM PreviousMonth), (SELECT balance FROM account WHERE id = ?)) as accountStartingBalance,
        (SELECT accountEndingBalance FROM LastTrade) as accountEndingBalance,
        profitLoss,
        percentGainLoss,
        winningRate,
        totalRMultiple,
        expectancyPerTrade,
        avgTimeInTradeMins
      FROM CurrentMonth;
    `,
			[
				accountId, // PreviousMonth: accountId
				monthStart, // PreviousMonth: entry_time < monthStart
				accountId, // CurrentMonth: COALESCE fallback accountId
				accountId, // CurrentMonth: accountId
				monthStart, // CurrentMonth: entry_time >= monthStart
				monthEnd, // CurrentMonth: entry_time < monthEnd
				accountId, // LastTrade: accountId
				monthStart, // LastTrade: entry_time >= monthStart
				monthEnd, // LastTrade: entry_time < monthEnd
				accountId, // Final SELECT: COALESCE fallback accountId
			]
		)) || {
			accountStartingBalance: 0,
			accountEndingBalance: 0,
			profitLoss: 0,
			percentGainLoss: 0,
			winningRate: 0,
			totalRMultiple: 0,
			expectancyPerTrade: 0,
			avgTimeInTradeMins: 0,
		};

		// Convert avgTimeInTrade from minutes to "H:MM" format
		const avgTimeInTradeMins = stats.avgTimeInTradeMins || 0;
		const hours = Math.floor(avgTimeInTradeMins / 60);
		const minutes = Math.round(avgTimeInTradeMins % 60);
		const avgTimeInTrade = `${hours}:${minutes.toString().padStart(2, "0")}`;

		const statistics: MonthlyStatistics = {
			accountStartingBalance: stats.accountStartingBalance || 0,
			accountEndingBalance: stats.accountEndingBalance || 0,
			profitLoss: stats.profitLoss || 0,
			percentGainLoss: stats.percentGainLoss || 0,
			winningRate: stats.winningRate || 0,
			totalRMultiple: stats.totalRMultiple || 0,
			expectancyPerTrade: stats.expectancyPerTrade || 0,
			avgTimeInTrade,
		};

		return NextResponse.json({ statistics, months: monthList });
	} catch (error) {
		console.error("Database error:", error);
		return NextResponse.json(
			{ error: "Internal server error" },
			{ status: 500 }
		);
	} finally {
		if (db) {
			await db.close(); // Ensure the database connection is closed
		}
	}
}
