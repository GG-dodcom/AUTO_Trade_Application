import { NextResponse } from "next/server";
import { openDb } from "@/lib/db";
import bcrypt from "bcrypt";

export async function POST(req: Request) {
	try {
		const { email, password } = await req.json();

		if (!email || !password) {
			return NextResponse.json(
				{ message: "Missing credentials" },
				{ status: 400 }
			);
		}

		const db = await openDb();

		// Check if email already exists
		const existingUser = await db.get("SELECT * FROM users WHERE email = ?", [
			email,
		]);

		if (existingUser) {
			return NextResponse.json(
				{ message: "Email already in use" },
				{ status: 400 }
			);
		}

		const hashedPassword = await bcrypt.hash(password, 10);

		await db.run("INSERT INTO users (email, password) VALUES (?, ?)", [
			email,
			hashedPassword,
		]);

		return NextResponse.json(
			{ message: "Account created successfully! Redirecting to login page..." },
			{ status: 201 }
		);
	} catch (error) {
		return NextResponse.json(
			{ message: "Error registering user", error },
			{ status: 500 }
		);
	}
}
