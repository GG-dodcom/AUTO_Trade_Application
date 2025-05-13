import { openDb } from "@/lib/db";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const userId = searchParams.get("userId");
	const loginId = searchParams.get("loginId");
	const server = searchParams.get("server");

	let db;
	try {
		db = await openDb();

		if (!userId || !loginId || !server) {
			return NextResponse.json(
				{ error: "Missing userId, loginId, or server" },
				{ status: 400 }
			);
		}

		const account = await db.get(
			"SELECT id FROM account WHERE userId = ? AND loginId = ? AND server = ?",
			[userId, loginId, server]
		);

		if (!account) {
			return NextResponse.json({ error: "Account not found" }, { status: 404 });
		}

		return NextResponse.json({ accountId: account.id });
	} catch (error) {
		console.error("API error:", error);
		return NextResponse.json(
			{ error: "Failed to fetch account ID" },
			{ status: 500 }
		);
	} finally {
		if (db) await db.close();
	}
}
