import { SignJWT, jwtVerify, JWTPayload } from "jose";

const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
	throw new Error("JWT_SECRET is not defined in environment variables");
}

if (!JWT_SECRET) {
	throw new Error("JWT_SECRET is not defined!");
}

// Convert the secret key to a Uint8Array for jose
const secretKeyUint8 = new TextEncoder().encode(JWT_SECRET);

// Generate JWT Token
export async function generateToken(payload: JWTPayload): Promise<string> {
	return await new SignJWT(payload)
		.setProtectedHeader({ alg: "HS256" })
		.setExpirationTime("1h")
		.sign(secretKeyUint8);
}

// Verify JWT Token
export async function verifyToken(token: string): Promise<any> {
	try {
		const { payload } = await jwtVerify(token, secretKeyUint8);
		return payload; // Return decoded payload
	} catch (error) {
		console.error("JWT Verification Failed:", error);
		return null; // Invalid token
	}
}
