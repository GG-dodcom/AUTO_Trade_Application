// src/app/api/monthly-r-multiple/route.ts
import { openDb } from "@/lib/db";
import { monthNames } from "@/lib/lib";
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
		const query = `
      SELECT 
        strftime('%Y-%m', exit_time) AS month,
        AVG(r_multiple) AS avg_r_multiple
      FROM trades
      WHERE accountId = ? 
        AND status = 'Closed' 
        AND r_multiple IS NOT NULL
        AND exit_time IS NOT NULL
      GROUP BY strftime('%Y-%m', exit_time)
      ORDER BY month ASC
    `;
		const params = [accountId];

		const results = await db.all(query, params);

		if (!results || results.length === 0) {
			return NextResponse.json(
				{
					error: "No closed trades with R-Multiple data found for this account",
				},
				{ status: 404 }
			);
		}

		const categories = results.map((row) => {
			const [year, month] = row.month.split("-");
			return `${monthNames[parseInt(month) - 1]} ${year}`;
		});
		const rMultiples = results.map((row) => row.avg_r_multiple);

		const series: ApexAxisChartSeries = [
			{
				name: "Avg R-Multiple",
				data: rMultiples,
			},
		];

		return NextResponse.json(
			{
				series,
				month: categories.join(","), // Comma-separated for consistency with other routes
			},
			{ status: 200 }
		);
	} catch (error: any) {
		console.error(
			"Error fetching monthly R-Multiple:",
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
