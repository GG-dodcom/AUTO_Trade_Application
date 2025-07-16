// src/middleware.ts
import { NextResponse, NextRequest } from "next/server";
import { verifyToken } from "@/lib/jwt";

export async function middleware(req: NextRequest) {
	const token = req.cookies.get("auth-token")?.value; // Get JWT token from cookies
	const publicPaths = ["/auth/login", "/auth/signup"]; // Publicly accessible routes

	// Allow public paths without authentication
	if (publicPaths.includes(req.nextUrl.pathname)) {
		return NextResponse.next();
	}

	// Redirect to login if no token
	if (!token) {
		return NextResponse.redirect(new URL("/auth/login", req.url));
	}

	// Verify the token
	const payload = await verifyToken(token);
	if (!payload) {
		return NextResponse.redirect(new URL("/auth/login", req.url));
	}

	return NextResponse.next(); // Continue processing request
}

// Apply middleware to all routes except static and API routes
export const config = {
	matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"], // Exclude static & API routes
};
