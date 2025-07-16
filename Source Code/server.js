// Log the very first thing to confirm the script starts
console.log("Starting server.js...");

// Log Node.js version for debugging
console.log(`Running on Node.js version: ${process.version}`);

const Database = require("better-sqlite3");
console.log("Loaded better-sqlite3 module");

const { WebSocketServer } = require("ws");
console.log("Loaded ws module");

const crypto = require("crypto");
console.log("Loaded crypto module");

const path = require("path");
console.log("Loaded path module");

// Determine the correct database path
const dbPath = path.join(__dirname, "database.db");
console.log(`Attempting to connect to database at: ${dbPath}`);

let db;

try {
	db = new Database(dbPath);
	console.log("Database instance created");
	db.prepare("SELECT 1").get();
	console.log("✅ Database connected successfully.");
} catch (error) {
	console.error("❌ Database connection failed:", error.message);
	console.error(error.stack);
	process.exit(1);
}

const wss = new WebSocketServer({ port: 4000 });
console.log("✅ WebSocket server running on ws://localhost:4000");

// Store client subscriptions
const clients = new Map(); // Map<ws, { accountId, startDate, endDate }>

// Store chart client subscriptions for trades line chart
const chartClients = new Map(); // Map<accountId, WebSocket[]>

// Function to fetch filtered trades
const getFilteredTrades = (accountId, startDate, endDate) => {
	let query = "SELECT * FROM trades WHERE accountId = ?";
	let params = [accountId];

	if (startDate && endDate) {
		query += " AND entry_time BETWEEN ? AND ?";
		params.push(startDate, endDate);
	}

	query += " ORDER BY entry_time DESC";

	return db.prepare(query).all(...params);
};

// Function to fetch chart trades (new)
const getChartTrades = (accountId) => {
	return db
		.prepare(
			`
				SELECT ticket, entry_time, exit_time, entry_price, exit_price, direction, profit_loss
				FROM trades
				WHERE accountId = ?
				ORDER BY entry_time ASC
				`
		)
		.all(accountId);
};

// Function to check if a trade is relevant to a client
const isTradeRelevant = (trade, { accountId, startDate, endDate }) => {
	if (trade.accountId !== accountId) return false;
	if (startDate && new Date(trade.entry_time) < new Date(startDate))
		return false;
	if (endDate && new Date(trade.entry_time) > new Date(endDate)) return false;
	return true;
};

// Handle WebSocket connections
wss.on("connection", (ws) => {
	console.log("🔗 Client connected (Port 4000)");

	ws.on("message", (message) => {
		try {
			const data = JSON.parse(message.toString());
			const { accountId, startDate, endDate, type } = data;

			if (!accountId) {
				ws.send(JSON.stringify({ error: "Missing accountId" }));
				return;
			}

			// Handle based on client type
			if (type === "chart") {
				// Chart client subscription
				if (!chartClients.has(accountId)) {
					chartClients.set(accountId, []);
				}
				chartClients.get(accountId).push(ws);

				// Send initial chart trades
				const trades = getChartTrades(accountId);
				ws.send(JSON.stringify({ type: "update", trades }));

				console.log(`✅ Chart client subscribed: accountId=${accountId}`);
			} else {
				// Existing client subscription
				clients.set(ws, { accountId, startDate, endDate });

				// Send initial filtered trades
				const trades = getFilteredTrades(accountId, startDate, endDate);
				ws.send(JSON.stringify({ type: "update", trades }));

				console.log(
					`✅ Client subscribed: accountId=${accountId}, start=${startDate}, end=${endDate}`
				);
			}
		} catch (error) {
			console.error("❌ Error processing message:", error.message);
			ws.send(JSON.stringify({ error: "Invalid message format" }));
		}
	});

	ws.on("close", () => {
		// Clean up existing clients
		if (clients.has(ws)) {
			clients.delete(ws);
			console.log("⚠️ Client disconnected");
		}

		// Clean up chart clients
		for (const [accountId, wsList] of chartClients) {
			chartClients.set(
				accountId,
				wsList.filter((client) => client !== ws)
			);
			if (chartClients.get(accountId).length === 0) {
				chartClients.delete(accountId);
			}
		}
		console.log("⚠️ Client disconnected (Port 4000)");
	});

	ws.on("error", (error) => {
		console.error("❌ WebSocket client error:", error.message);
	});
});

wss.on("error", (error) => {
	console.error("❌ WebSocket server error:", error.message);
	if (error.code === "EADDRINUSE") {
		console.error("Port 4000 is already in use. Exiting...");
		process.exit(1);
	}
});

// Function to calculate SHA-256 hash of the entire `trades` table
const getTableHash = () => {
	const trades = db.prepare("SELECT * FROM trades").all();
	const dataString = JSON.stringify(trades);
	return crypto.createHash("sha256").update(dataString).digest("hex");
};

// Track last table hash for detecting changes
let lastTableHash = getTableHash();

// Track last ticket for chart clients
let lastChartCheck = new Map(); // Map<accountId, lastTicket>

// Periodically check for trade updates (combined polling)
setInterval(() => {
	try {
		const newTableHash = getTableHash();
		if (newTableHash !== lastTableHash) {
			const allTrades = db.prepare("SELECT * FROM trades").all();

			// Notify only relevant clients
			clients.forEach((filters, ws) => {
				if (ws.readyState === 1) {
					const filteredTrades = allTrades.filter((trade) =>
						isTradeRelevant(trade, filters)
					);
					if (filteredTrades.length > 0) {
						ws.send(JSON.stringify({ type: "update", trades: filteredTrades }));
					}
				}
			});

			// Notify chart clients
			for (const accountId of chartClients.keys()) {
				const chartTrades = allTrades
					.filter((trade) => trade.accountId === accountId)
					.map(
						({
							ticket,
							entry_time,
							exit_time,
							entry_price,
							exit_price,
							direction,
						}) => ({
							ticket,
							entry_time,
							exit_time,
							entry_price,
							exit_price,
							direction,
						})
					)
					.sort((a, b) => new Date(a.entry_time) - new Date(b.entry_time));

				const latestTicket = chartTrades.length
					? chartTrades[chartTrades.length - 1].ticket
					: 0;
				const lastTicket = lastChartCheck.get(accountId) || 0;

				if (latestTicket !== lastTicket) {
					const wsList = chartClients.get(accountId) || [];
					wsList.forEach((ws) => {
						if (ws.readyState === 1) {
							ws.send(JSON.stringify({ type: "update", trades: chartTrades }));
						}
					});
					lastChartCheck.set(accountId, latestTicket);
				}
			}

			lastTableHash = newTableHash;
			console.log("✅ Trades updated and sent to clients (Port 4000)");
		}
	} catch (error) {
		console.error("❌ Error checking database:", error.message);
	}
}, 1000); // Polling interval

// Keep the process alive and log periodically to confirm
setInterval(() => {
	console.log("WebSocket server is still running...");
}, 30000);

// Handle uncaught exceptions to prevent silent exits
process.on("uncaughtException", (error) => {
	console.error("Uncaught exception in server.js:", error);
	console.error(error.stack);
	process.exit(1);
});

// Handle unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
	console.error("Unhandled Rejection at:", promise, "reason:", reason);
	console.error(reason.stack);
	process.exit(1);
});
