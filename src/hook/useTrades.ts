// src/hook/useTrades.tsx
"use client"; // Ensure this is client-side

import { useState, useEffect, useRef } from "react";
import { Trade } from "@/lib/definitions";
import { getAccountId } from "@/lib/data";
import { useUser } from "@/hook/useUser";
import { ApiRoute } from "@/app/api";
import { formatSqlDate } from "@/lib/lib";

const API_URL = ApiRoute.getTrades; // "/api/trades"
const WS_URL = ApiRoute.ws_url;

export function useTrades(
	startDate?: Date,
	endDate?: Date,
	onNewTradeAction?: (action: "Buy" | "Sell") => void
) {
	const [trades, setTrades] = useState<Trade[]>([]);
	const [error, setError] = useState<string | null>(null);
	const { userId, account, accServer } = useUser(); // account = loginId, accServer = server
	const [accountId, setAccountId] = useState<string | null>(null);
	const socketRef = useRef<WebSocket | null>(null); // WebSocket reference
	const latestTradeRef = useRef<Trade | null>(null); // Store latest trade for onNewTradeAction

	// Fetch trades based on date range and accountId
	const fetchTrades = async (start?: Date, end?: Date) => {
		try {
			if (!accountId) {
				throw new Error("Account ID not available");
			}

			const url = new URL(API_URL); // Constructs "http://localhost:3000/api/trades"
			url.searchParams.append("accountId", accountId);

			// Add date range if provided
			if (start && end) {
				url.searchParams.append("start_date", formatSqlDate(start));
				url.searchParams.append("end_date", formatSqlDate(end));
			}

			// console.log("Fetching trades from:", url.toString()); // Debug
			const response = await fetch(url.toString(), {
				credentials: "include", // Include cookies if needed
			});
			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(
					errorData.error || `HTTP error! Status: ${response.status}`
				);
			}
			const data: Trade[] = await response.json();
			// console.log("Fetched trades:", data); // Debug
			// Sorting is handled by the backend, but we can keep this as a fallback
			// const sortedTrades = data.sort(
			// 	(a, b) =>
			// 		new Date(b.entry_time).getTime() - new Date(a.entry_time).getTime()
			// );
			setTrades(data);
			setError(null);
		} catch (error: any) {
			console.error("Error fetching trades:", error);
			setError(error.message || "Failed to fetch trades.");
		}
	};

	// Fetch accountId when user details change
	useEffect(() => {
		const setupAccount = async () => {
			if (userId && account && accServer) {
				try {
					const id = await getAccountId(userId, account, accServer);
					setAccountId(id);
				} catch (err: any) {
					setError(err.message || "Failed to fetch account ID");
				}
			}
		};
		setupAccount();
	}, [userId, account, accServer]);

	// Fetch trades when accountId or dates change
	useEffect(() => {
		if (accountId) {
			fetchTrades(startDate, endDate);
		}
	}, [accountId, startDate, endDate]);

	// WebSocket Connection for Realtime Updates
	useEffect(() => {
		if (!accountId) return; // Wait until accountId is available

		const socket = new WebSocket(WS_URL);
		socketRef.current = socket;

		socket.onopen = () => {
			console.log("✅ Connected to WebSocket");
			// Send subscription details to WebSocket server
			socket.send(
				JSON.stringify({
					accountId,
					startDate: startDate ? formatSqlDate(startDate) : undefined,
					endDate: endDate ? formatSqlDate(endDate) : undefined,
				})
			);
		};

		socket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				// console.log("WebSocket message received:", data); // Debug
				if (data.type === "update" && Array.isArray(data.trades)) {
					// Update trades with sorted WebSocket data
					const sortedTrades = data.trades.sort(
						(a: Trade, b: Trade) =>
							new Date(b.entry_time).getTime() -
							new Date(a.entry_time).getTime()
					);
					setTrades(sortedTrades);
					latestTradeRef.current = sortedTrades[0] || null; // Store latest trade
				} else if (data.error) {
					setError(data.error);
				}
			} catch (error) {
				console.error("❌ Error parsing WebSocket message:", error);
			}
		};

		socket.onerror = (error) => {
			console.error("❌ WebSocket error:", error);
			setError("WebSocket connection failed");
		};

		socket.onclose = () => {
			console.log("⚠️ WebSocket closed");
		};

		// Cleanup WebSocket on unmount
		return () => {
			socket.close();
		};
	}, [accountId, startDate, endDate]);

	// Handle onNewTradeAction separately
	useEffect(() => {
		if (latestTradeRef.current && onNewTradeAction) {
			const latestTrade = latestTradeRef.current;
			if (latestTrade.direction === 0 || latestTrade.direction === 1) {
				const action = latestTrade.direction === 0 ? "Buy" : "Sell";
				// console.log("Calling onNewTradeAction:", action); // Debug
				onNewTradeAction(action);
			}
		}
	}, [trades, onNewTradeAction]); // Depend on trades and onNewTradeAction

	return { trades, error, fetchTrades };
}
