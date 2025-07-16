import { createCipheriv, createDecipheriv, randomBytes } from "crypto";

// AES encryption settings
const algorithm = "aes-256-cbc";
const key = Buffer.from(
	process.env.ENCRYPTION_KEY || "32-byte-long-key-here",
	"utf8"
); // 32 bytes for AES-256

// Function to encrypt a plaintext password
export function encryptPassword(plaintextPassword: string): string {
	const iv = randomBytes(16); // Generate a new IV for each encryption
	const cipher = createCipheriv(algorithm, key, iv);
	const encryptedPassword =
		iv.toString("hex") +
		cipher.update(plaintextPassword, "utf8", "hex") +
		cipher.final("hex");
	return encryptedPassword;
}

// Debounce function to delay execution
export const debounce = <F extends (...args: any[]) => void>(
	func: F,
	wait: number
): ((...args: Parameters<F>) => void) => {
	let timeout: NodeJS.Timeout | null = null;
	return (...args: Parameters<F>) => {
		if (timeout) clearTimeout(timeout);
		timeout = setTimeout(() => func(...args), wait);
	};
};

// Format months for display (e.g., "Jan 2025")
export const monthNames = [
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec",
];

export function formatSqlDate(date: Date): string {
	const pad2 = (n: number) => n.toString().padStart(2, "0");
	const Y = date.getFullYear();
	const M = pad2(date.getMonth() + 1);
	const D = pad2(date.getDate());
	const h = pad2(date.getHours());
	const m = pad2(date.getMinutes());
	const s = pad2(date.getSeconds());
	return `${Y}-${M}-${D} ${h}:${m}:${s}`;
}
