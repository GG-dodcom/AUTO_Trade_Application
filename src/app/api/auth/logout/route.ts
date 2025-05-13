// src/app/api/auth/logout/route.ts
import { NextResponse } from "next/server";

export async function POST() {
	try {
		// Create response with success message
		const response = NextResponse.json(
			{ message: "Logout successful" },
			{ status: 200 }
		);

		// Clear the auth-token cookie by setting it to an empty value and expiring it immediately
		response.cookies.set("auth-token", "", {
			httpOnly: true,
			secure: process.env.NODE_ENV === "production",
			path: "/",
			sameSite: "strict",
			expires: new Date(0), // Expire immediately
		});

		return response;
	} catch (error) {
		console.error("Logout error:", error);
		return NextResponse.json(
			{ message: "Failed to logout due to an unexpected error" },
			{ status: 500 }
		);
	}
}
