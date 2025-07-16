// src/app/api/account-summary/route.ts
import { openDb } from "@/lib/db";
import { AccountSummary } from "@/lib/definitions";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const accountId = searchParams.get("accountId");

	if (!accountId) {
		return NextResponse.json(
			{ error: "Missing accountId query parameter" },
			{ status: 400 }
		);
	}

	const db = await openDb();
	try {
		// Get starting capital from account table
		const account = await db.get(`SELECT balance FROM account WHERE id = ?`, [
			accountId,
		]);
		if (!account) {
			return NextResponse.json({ error: "Account not found" }, { status: 404 });
		}

		// Starting capital is initial balance
		const startingCapital = account.balance;

		const latestTrade = await db.get(
			`SELECT account_balance FROM trades WHERE accountId = ? AND status = 'Closed' ORDER BY exit_time DESC LIMIT 1`,
			[accountId]
		);
		const accountBalance = latestTrade?.account_balance ?? account.balance;

		const tradeStats = (await db.get(
			`SELECT 
				COUNT(*) as closedTrades,
				SUM(CASE WHEN win_loss = 1 THEN 1 ELSE 0 END) as winningTrades,
				COALESCE(SUM(r_multiple), 0) as totalRMultiple,
				COALESCE(SUM(profit_loss), 0) as totalProfitLoss
			FROM trades 
			WHERE accountId = ? AND status = 'Closed'`,
			[accountId]
		)) || {
			closedTrades: 0,
			winningTrades: 0,
			totalRMultiple: 0,
			totalProfitLoss: 0,
		};

		const profitLoss = tradeStats.totalProfitLoss;
		const percentGainLoss =
			startingCapital > 0 ? (profitLoss / startingCapital) * 100 : 0;
		const winningRate =
			tradeStats.closedTrades > 0
				? (tradeStats.winningTrades / tradeStats.closedTrades) * 100
				: 0;
		const totalRMultiple = tradeStats.totalRMultiple;
		const expectancyPerTrade =
			tradeStats.closedTrades > 0
				? totalRMultiple / tradeStats.closedTrades
				: 0;

		// Construct response
		const summary: AccountSummary = {
			startingCapital,
			accountBalance,
			profitLoss,
			percentGainLoss,
			winningRate,
			totalRMultiple,
			expectancyPerTrade,
		};

		return NextResponse.json(summary, { status: 200 });
	} catch (error: any) {
		console.error(
			"Error fetching account summary:",
			error.message,
			error.stack
		);
		return NextResponse.json(
			{ error: "Internal server error", message: error.message },
			{ status: 500 }
		);
	} finally {
		await db.close();
	}
}
