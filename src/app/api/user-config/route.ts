import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET() {
	const configPath = path.join(
		process.cwd(),
		"backend/config/user_config.json"
	);
	const configData = await fs.readFile(configPath, "utf-8");
	return NextResponse.json(JSON.parse(configData));
}

export async function POST(request: Request) {
	const configPath = path.join(
		process.cwd(),
		"backend/config/user_config.json"
	);
	const updatedConfig = await request.json();
	await fs.writeFile(configPath, JSON.stringify(updatedConfig, null, 2));
	return NextResponse.json({ message: "Config updated" });
}
