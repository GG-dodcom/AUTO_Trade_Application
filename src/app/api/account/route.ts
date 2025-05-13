// src/app/api/account/route.ts
import { NextResponse } from "next/server";
import { openDb } from "@/lib/db";
import { randomBytes } from "crypto";
import { ApiRoute } from "..";
import { getAccountInfo } from "@/lib/data";
import { encryptPassword } from "@/lib/lib";

export async function GET(req: Request) {
	try {
		const { searchParams } = new URL(req.url);
		const userId = searchParams.get("userId"); // Get userId from query params

		if (!userId) {
			return NextResponse.json(
				{ message: "User ID is required" },
				{ status: 400 }
			);
		}

		const db = await openDb();
		const account = await db.get(
			"SELECT loginId, password, server, remember FROM account WHERE userId = ?",
			[userId]
		);

		if (!account) {
			return NextResponse.json(
				{ message: "Account not found" },
				{ status: 200 }
			);
		}

		console.log("Fetched account:", account);
		const storedRemember = Boolean(account.remember); // Convert 0/1 to true/false
		return NextResponse.json(
			{
				loginId: account.loginId,
				password: storedRemember ? account.password : "", // Encrypted password from DB
				server: account.server,
				remember: storedRemember,
			},
			{ status: 200 }
		);
	} catch (error) {
		console.error("Error fetching account:", error);
		return NextResponse.json(
			{ message: "Server error, please try again later" },
			{ status: 500 }
		);
	}
}

/**
 * Handles account creation and login
 * - `POST /api/account` → if account exists in database, login the user; otherwise, create a new account
 */
export async function POST(req: Request) {
	try {
		const { userId, loginId, server, password, remember } = await req.json();

		if (!loginId || !server || !password) {
			return NextResponse.json(
				{ message: "Missing credentials" },
				{ status: 400 }
			);
		}

		if (!userId) {
			return NextResponse.json(
				{ message: "User not authenticated" },
				{ status: 401 }
			);
		}

		// Check if the password looks like an encrypted string (AES hex)
		const isEncryptedPassword = /^[0-9a-fA-F]{32,}$/.test(password); // Rough check for hex string
		let encryptedPassword = null;
		if (!isEncryptedPassword) {
			try {
				// Encrypt the password
				encryptedPassword = encryptPassword(password);
			} catch (err) {
				console.error("Decryption failed: ", err);
				return NextResponse.json(
					{ message: "Failed to encrypt password" },
					{ status: 400 }
				);
			}
		} else {
			encryptedPassword = password;
		}

		// Step 1: Attempt to login to MT5 with encrypted password
		const mt5Response = await fetch(ApiRoute.postMT5Login, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				account: loginId,
				server,
				password: encryptedPassword,
			}),
		});
		const mt5Data = await mt5Response.json();

		// Step 2: If MT5 login fails, return error and do not proceed to database operations
		if (!mt5Response.ok || mt5Data.error) {
			return NextResponse.json(
				{
					message:
						mt5Data.error ||
						"Failed to login to MT5. Please enter valid account details.",
				},
				{ status: 400 }
			);
		}

		// Step 3: Fetch MT5 account info after successful login
		let accountInfo = null;
		accountInfo = await getAccountInfo(
			userId,
			loginId,
			encryptedPassword,
			server
		);

		// Step 4: Database operations
		const db = await openDb();
		const existingAcc = await db.get(
			"SELECT * FROM account WHERE userId = ? AND loginId = ? AND server = ?",
			[userId, loginId, server]
		);

		if (existingAcc) {
			// Account exists, update if password differs
			const storedRemember = Boolean(existingAcc.remember); // Current value in DB
			const needsUpdate =
				existingAcc.password !== encryptedPassword ||
				storedRemember !== remember;

			if (needsUpdate) {
				await db.run(
					"UPDATE account SET password = ?, remember = ? WHERE userId = ? AND loginId = ? AND server = ?",
					[encryptedPassword, remember ? 1 : 0, userId, loginId, server]
				);
				return NextResponse.json(
					{
						message:
							existingAcc.password !== encryptedPassword
								? "Password updated and login successful!"
								: "Remember preference updated and login successful!",
						encryptedPassword, // Return the encrypted password
						accountInfo: accountInfo || null, // Include account info if available
					},
					{ status: 200 }
				);
			}
			// Password matches, just return success
			return NextResponse.json(
				{
					message: "Login successful!",
					encryptedPassword: existingAcc.password, // Return the stored encrypted password
					accountInfo: accountInfo || null, // Include account info if available
				},
				{ status: 200 }
			);
		} else {
			// Account doesn’t exist, save it to the database
			await db.run(
				`INSERT INTO account (
					userId, loginId, server, password, remember, trade_mode, leverage, limit_orders, 
            margin_so_mode, trade_allowed, trade_expert, margin_mode, currency_digits, 
            fifo_close, balance, credit, profit, equity, margin, margin_free, margin_level, 
            margin_so_call, margin_so_so, margin_initial, margin_maintenance, assets, 
            liabilities, commission_blocked, name, currency, company
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				[
					userId,
					loginId,
					server,
					encryptedPassword,
					remember ? 1 : 0,
					accountInfo?.trade_mode,
					accountInfo?.leverage,
					accountInfo?.limit_orders,
					accountInfo?.margin_so_mode,
					accountInfo?.trade_allowed,
					accountInfo?.trade_expert,
					accountInfo?.margin_mode,
					accountInfo?.currency_digits,
					accountInfo?.fifo_close,
					accountInfo?.balance,
					accountInfo?.credit,
					accountInfo?.profit,
					accountInfo?.equity,
					accountInfo?.margin,
					accountInfo?.margin_free,
					accountInfo?.margin_level,
					accountInfo?.margin_so_call,
					accountInfo?.margin_so_so,
					accountInfo?.margin_initial,
					accountInfo?.margin_maintenance,
					accountInfo?.assets,
					accountInfo?.liabilities,
					accountInfo?.commission_blocked,
					accountInfo?.name,
					accountInfo?.currency,
					accountInfo?.company,
				]
			);

			return NextResponse.json(
				{
					message: "Account created successfully!",
					encryptedPassword, // Return the newly encrypted password
					accountInfo: accountInfo || null, // Include account info if available
				},
				{ status: 201 }
			);
		}
	} catch (error: any) {
		console.error("Error processing account:", error);
		return NextResponse.json(
			{ message: "Server error, please try again later", error: error.message },
			{ status: 500 }
		);
	}
}
