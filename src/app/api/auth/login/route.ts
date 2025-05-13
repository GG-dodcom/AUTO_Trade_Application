// src/app/api/auth/login/route.ts
import { NextResponse } from "next/server";
import { openDb } from "@/lib/db";
import { generateToken } from "@/lib/jwt";
import bcrypt from "bcrypt";

interface LoginRequest {
	email: string;
	password: string;
}

export async function POST(req: Request) {
	try {
		const { email, password } = (await req.json()) as LoginRequest;

		if (!email || !password) {
			return NextResponse.json(
				{ message: "Please provide both email and password" },
				{ status: 400 }
			);
		}

		const db = await openDb();
		try {
			const user = await db.get("SELECT * FROM users WHERE email = ?", [email]);

			// Check if account exists
			if (!user) {
				return NextResponse.json(
					{ message: "Account does not exist" },
					{ status: 404 } // 404: Not Found is more appropriate here
				);
			}

			// Check password
			if (!(await bcrypt.compare(password, user.password))) {
				return NextResponse.json(
					{ message: "Incorrect password" },
					{ status: 401 }
				);
			}

			// Generate JWT Token
			const token = await generateToken({ userId: user.id, email: user.email });

			// Create response with detailed success message
			const response = NextResponse.json(
				{
					message: "Login successful",
					user: {
						id: user.id,
						email: user.email,
					},
					token: token,
				},
				{ status: 200 }
			); // Explicitly set status for clarity

			// Set HTTP-only secure cookie for token
			response.cookies.set("auth-token", token, {
				httpOnly: true, // Prevents access via JavaScript (XSS protection)
				secure: process.env.NODE_ENV === "production", // Enforce HTTPS in production
				path: "/",
				sameSite: "strict", // Prevents CSRF attacks
				maxAge: 60 * 60, // Expires in 1 hour, Prevents CSRF attacks
			});

			return response;
		} finally {
			await db.close();
		}
	} catch (error) {
		console.error("Login error:", error);
		return NextResponse.json(
			{ message: "An unexpected error occurred. Please try again later." },
			{ status: 500 }
		);
	}
}
