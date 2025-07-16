// src/components/RunModel.tsx
"use client";
import { ApiRoute } from "@/app/api";
import React, { useState, useEffect } from "react";
import { io } from "socket.io-client";
import { showNotifyToast } from "./NotificationToast";
import { useUser } from "@/hook/useUser";

interface RunModelProps {
	initialModel?: string;
	onToggle?: (isRunning: boolean) => void;
}

const RunModel: React.FC<RunModelProps> = ({
	initialModel = "XAUUSD Model",
	onToggle,
}) => {
	const [isRunning, setIsRunning] = useState(false);
	const [startTime, setStartTime] = useState<Date | null>(null);
	const [selectedModel, setSelectedModel] = useState(initialModel);
	const [isLoading, setIsLoading] = useState(false);
	const [errorMessage, setErrorMessage] = useState<string | null>(null);
	const [isAlgoTradingEnabled, setIsAlgoTradingEnabled] = useState<
		boolean | null
	>(null); // New state for algo trading status
	const { accountId, account, accServer, accPassword } = useUser();

	// Fetch algo trading status
	const checkAlgoTradingStatus = async () => {
		try {
			const response = await fetch(ApiRoute.algoTradingStatus, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					account: account,
					password: accPassword,
					server: accServer,
				}),
			});

			const data = await response.json();
			if (data.status === "success") {
				setIsAlgoTradingEnabled(data.isAlgoTradingEnabled);
				if (!data.isAlgoTradingEnabled) {
					setErrorMessage(
						"Algo trading is disabled in MT5. Please enable it to start trading."
					);
				}
			} else {
				throw new Error(data.message || "Failed to check algo trading status");
			}
		} catch (error: any) {
			console.error("Error checking algo trading status:", error);
			setErrorMessage(error.message || "Failed to check algo trading status");
			setIsAlgoTradingEnabled(false); // Assume disabled on error
		}
	};

	// Socket.IO connection and initial status checks
	useEffect(() => {
		if (!accountId) return; // Don't run if accountId is not available
		const socket = io(
			process.env.NEXT_PUBLIC_BACKEND_ENDPOINT || "http://localhost:5000",
			{
				transports: ["websocket", "polling"],
			}
		);

		socket.on("connect", () => {
			console.log("Connected to Socket.IO server");
		});

		socket.on("status_update", (data: { isRunning: boolean }) => {
			console.log("Status update received:", data);
			setIsRunning(data.isRunning);
			if (data.isRunning) {
				setStartTime(new Date());
			} else {
				setStartTime(null);
			}
			onToggle?.(data.isRunning);
		});

		socket.on("connect_error", (error) => {
			console.error("Socket.IO connection error:", error);
			setErrorMessage("Failed to connect to server. Please try again.");
		});

		// Check initial statuses on mount
		checkStatus();
		checkAlgoTradingStatus(); // Check algo trading status on mount

		return () => {
			socket.disconnect();
			console.log("Disconnected from Socket.IO server");
		};
	}, [onToggle, accountId]);

	const checkStatus = async () => {
		try {
			const response = await fetch(ApiRoute.getTradingStatus);
			const data = await response.json();
			if (data.status === "success") {
				console.log("checkStatus response:", data);
				setIsRunning(data.isRunning);
				if (data.isRunning && data.timestamp && !startTime) {
					setStartTime(new Date(data.timestamp));
				} else if (!data.isRunning) {
					setStartTime(null);
				}
				onToggle?.(data.isRunning);
			}
		} catch (error) {
			console.error("Error checking status:", error);
			setErrorMessage("Failed to check trading status");
		}
	};

	const toggleRun = async () => {
		if (!isAlgoTradingEnabled) {
			setErrorMessage(
				"Algo trading is disabled in MT5. Please enable it to start trading."
			);
			return;
		}

		setIsLoading(true);
		setErrorMessage(null);

		if (!accountId) {
			setErrorMessage("Missing required account details. Please log in again.");
			setIsLoading(false);
			return;
		}

		try {
			const endpoint = isRunning
				? ApiRoute.postTradingStop
				: ApiRoute.postTradingStart;

			const response = await fetch(endpoint, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					model: selectedModel,
					accountId: accountId,
					account: account,
					password: accPassword,
					server: accServer,
				}),
			});

			const data = await response.json();
			console.log("Response:", data);

			if (!response.ok) {
				throw new Error(data.message || `Server error: ${response.status}`);
			}

			if (data.status !== "success") {
				throw new Error(data.message || "Failed to toggle trading");
			}

			// Status will be updated via SocketIO
		} catch (error: any) {
			console.error("Error toggling trading:", error);
			setErrorMessage(
				error.message || "An error occurred while toggling trading"
			);
			await checkStatus(); // Re-sync status in case of failure
		} finally {
			setIsLoading(false);
		}
	};

	// Toast notifications for running/stopped status
	useEffect(() => {
		if (!accountId) return; // Don't show toast if accountId is not available
		let timeoutId;

		// Delay the toast notification by 500ms
		timeoutId = setTimeout(() => {
			if (isRunning) {
				showNotifyToast("Model is running.", "success", "bottom-center", 1000);
			} else {
				showNotifyToast("Model stopped.", "error", "bottom-center", 1000);
			}
		}, 500);

		// Cleanup: clear the timeout if isRunning changes before the delay completes
		return () => {
			clearTimeout(timeoutId);
		};
	}, [isRunning, accountId]); // Trigger whenever isRunning changes

	// Toast notification for errors
	useEffect(() => {
		if (!accountId) return; // Don't show toast if accountId is not available
		if (errorMessage != null) {
			showNotifyToast(errorMessage, "error", "bottom-center", 1000);
		}
	}, [errorMessage, accountId]);

	return (
		<div className="w-auto mx-auto p-4 bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-lg border border-gray-200">
			{/* Header */}
			<h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">
				Trading Control
			</h2>

			{/* Model Selection */}
			<div className="flex items-center justify-center mb-6">
				<label className="text-sm font-medium text-gray-700 mr-3">Model:</label>
				<select
					className="w-48 p-2 bg-white border border-gray-300 rounded-md text-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed transition-all"
					value={selectedModel}
					onChange={(e) => setSelectedModel(e.target.value)}
					disabled={isRunning || isLoading || !isAlgoTradingEnabled}
				>
					<option value="XAUUSD Model">XAUUSD Model</option>
				</select>
			</div>

			{/* Error Message */}
			{errorMessage && (
				<div className="mb-4 text-red-600 text-sm text-center">
					{errorMessage}
				</div>
			)}

			{/* Control Button */}
			<div className="flex justify-center mb-6">
				<button
					className={`w-20 h-20 rounded-full flex items-center justify-center text-white text-2xl transition-all ${
						!isAlgoTradingEnabled
							? "bg-gray-400 cursor-not-allowed opacity-60"
							: isRunning
							? "bg-yellow-500 hover:bg-yellow-600"
							: "bg-blue-600 hover:bg-blue-700"
					} ${
						isLoading || !isAlgoTradingEnabled
							? "opacity-60 cursor-not-allowed scale-95"
							: "hover:scale-105"
					}`}
					onClick={toggleRun}
					disabled={isLoading || !isAlgoTradingEnabled}
					title={
						!isAlgoTradingEnabled
							? "Algo trading is disabled in MT5"
							: isRunning
							? "Stop Trading"
							: "Start Trading"
					}
				>
					{isLoading ? "..." : isRunning ? "❚❚" : "▶"}
				</button>
			</div>

			{/* Status and Start Info */}
			<div className="grid grid-cols-2 gap-4">
				<div className="bg-gray-100 p-3 rounded-lg shadow-sm">
					<p className="text-xs font-semibold text-gray-600 uppercase">
						Start Time
					</p>
					<p className="text-sm text-gray-800 mt-1">
						{startTime ? startTime.toLocaleString() : "--:--:--"}
					</p>
				</div>
				<div className="bg-gray-100 p-3 rounded-lg shadow-sm">
					<p className="text-xs font-semibold text-gray-600 uppercase">
						Status
					</p>
					<p
						className={`text-sm mt-1 font-medium ${
							isRunning ? "text-green-600" : "text-red-600"
						}`}
					>
						{isRunning ? "Running" : "Stopped"}
					</p>
				</div>
			</div>
		</div>
	);
};

export default RunModel;
