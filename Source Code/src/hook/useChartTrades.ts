// hooks/useChartTrades.ts
"use client";

import { useState, useEffect, useRef } from "react";
import { useUser } from "@/hook/useUser";
import { ApiRoute } from "@/app/api";

interface ChartTrade {
	ticket: number;
	entry_time: string;
	exit_time: string | null;
	entry_price: number;
	exit_price: number | null;
	direction: number; // 0 = Buy, 1 = Sell
}

const WS_URL = "ws://localhost:4000"; // Updated to match server port

export function useChartTrades() {
	const { accountId } = useUser();
	const [trades, setTrades] = useState<ChartTrade[]>([]);
	const [error, setError] = useState<string | null>(null);
	const socketRef = useRef<WebSocket | null>(null);

	// Fetch trades
	const fetchTrades = async () => {
		try {
			if (!accountId) {
				return;
			}

			const response = await fetch(ApiRoute.getLineChartTrades(accountId), {
				credentials: "include",
			});
			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(
					errorData.error || `HTTP error! Status: ${response.status}`
				);
			}
			const data: ChartTrade[] = await response.json();
			setTrades(data);
			setError(null);
		} catch (error: any) {
			console.error("Error fetching trades:", error);
			setError(error.message || "Failed to fetch trades");
		}
	};

	// Fetch trades when accountId changes
	useEffect(() => {
		if (accountId) {
			fetchTrades();
		}
	}, [accountId]);

	// WebSocket for real-time updates
	useEffect(() => {
		if (!accountId) return;

		const socket = new WebSocket(WS_URL);
		socketRef.current = socket;

		socket.onopen = () => {
			console.log("✅ Connected to WebSocket");
			// Send subscription message with type "chart"
			socket.send(JSON.stringify({ accountId, type: "chart" }));
		};

		socket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === "update" && Array.isArray(data.trades)) {
					setTrades(data.trades);
					setError(null);
				} else if (data.error) {
					setError(data.error);
				}
			} catch (error) {
				console.error("❌ Error parsing WebSocket message:", error);
				setError("Failed to parse WebSocket message");
			}
		};

		socket.onerror = (error) => {
			console.error("❌ WebSocket error:", error);
			setError("WebSocket connection failed");
		};

		socket.onclose = () => {
			console.log("⚠️ WebSocket closed");
		};

		// Cleanup on unmount or accountId change
		return () => {
			socket.close();
		};
	}, [accountId]);

	return { trades, error, fetchTrades };
}
