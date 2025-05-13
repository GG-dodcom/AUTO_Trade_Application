import sqlite3 from "sqlite3";
import { open, Database } from "sqlite";

// Define the return type for better TypeScript support
export async function openDb(): Promise<Database> {
	try {
		return await open({
			filename: process.env.DB_FILE || "./database.db",
			driver: sqlite3.Database,
		});
	} catch (error: any) {
		throw new Error(`Failed to open database: ${error.message}`);
	}
}
