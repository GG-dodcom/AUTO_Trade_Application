"use client";
import { ApiRoute } from "@/app/api";
import { showNotifyToast } from "@/components/NotificationToast";
import { useUser } from "@/hook/useUser";
import React, { useState, useEffect } from "react";

interface SymbolConfig {
	point: number;
	max_spread: number;
	transaction_fee: number;
	over_night_penalty: number;
	stop_loss_max: number;
	profit_taken_max: number;
	max_current_holding: number;
	limit_order: boolean;
	limit_order_expiration: number;
	volume: number;
}

interface UserConfig {
	env: {
		observation_list: string[];
		over_night_cash_penalty: number;
		balance: number;
		asset_col: string;
		time_col: string;
		random_start: boolean;
		log_filename: string;
		title: string;
		description: string;
	};
	symbol: {
		[key: string]: SymbolConfig;
	};
	trading_hour: {
		Sydney: { from: number; to: number };
		Tokyo: { from: number; to: number };
		London: { from: number; to: number };
		"New York": { from: number; to: number };
	};
	ploting: {
		buy: { color: string; shape: string };
		sell: { color: string; shape: string };
	};
}

interface MT5Data {
	login: number;
	server: string;
	name: string;
	company: string;
	leverage: string;
	point: number;
	volume_min: number;
	volume_max: number;
	trade_stops_level: number;
	over_night_penalty_long: number;
	over_night_penalty_short: number;
	trade_allowed: boolean;
}

interface EventMap {
	[key: string]: number; // e.g., {"Manufacturing PMI": 0, ...}
}

interface CurrencyMap {
	[key: string]: number; // e.g., {"CNY": 0, ...}
}

// Mapping of technical indicator names to human-readable descriptions
const indicatorDescriptions: { [key: string]: string } = {
	Open_norm: "Opening Price",
	High_norm: "Highest Price",
	Low_norm: "Lowest Price",
	Close_norm: "Closing Price",
	RSI_norm: "Relative Strength Index (RSI)",
	MACD_norm: "Moving Average Convergence Divergence (MACD)",
	MACD_Histogram_norm: "MACD Histogram",
	MACD_Signal_norm: "MACD Signal Line",
	STOCHk_14_3_3_norm: "Stochastic Oscillator (%K)",
	STOCHd_14_3_3_norm: "Stochastic Oscillator (%D)",
	ADX_norm: "Average Directional Index (ADX)",
	CCI_norm: "Commodity Channel Index (CCI)",
	SMA_norm: "Simple Moving Average (SMA)",
	"BBL_20_2.0_norm": "Bollinger Bands Lower Band",
	"BBM_20_2.0_norm": "Bollinger Bands Middle Band",
	"BBU_20_2.0_norm": "Bollinger Bands Upper Band",
	"BBB_20_2.0_norm": "Bollinger Bands Bandwidth",
	"BBP_20_2.0_norm": "Bollinger Bands Percent",
	"Fib_23.6_norm": "Fibonacci Retracement 23.6%",
	"Fib_38.2_norm": "Fibonacci Retracement 38.2%",
	"Fib_50.0_norm": "Fibonacci Retracement 50.0%",
	"Fib_61.8_norm": "Fibonacci Retracement 61.8%",
	"Fib_100.0_norm": "Fibonacci Retracement 100.0%",
	ATR_norm: "Average True Range (ATR)",
};

const Settings: React.FC = () => {
	const { account, accPassword, accServer } = useUser();
	const [activeTab, setActiveTab] = useState<
		"trade" | "indicators" | "economic"
	>("trade");
	const [config, setConfig] = useState<UserConfig | null>(null);
	const [mt5Data, setMt5Data] = useState<MT5Data | null>(null);
	const [eventMap, setEventMap] = useState<EventMap | null>(null);
	const [currencyMap, setCurrencyMap] = useState<CurrencyMap | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [symbolInput, setSymbolInput] = useState<string>("XAUUSD");
	const [showSymbolModal, setShowSymbolModal] = useState<boolean>(false);
	const [tempSymbol, setTempSymbol] = useState<string>("");

	// Fetch initial config, MT5 data, event map, and currency map
	useEffect(() => {
		async function fetchData() {
			if (!account || !accPassword || !accServer) {
				setError("Please provide MT5 account credentials");
				setLoading(false);
				return;
			}

			setError(null);
			setLoading(true);
			try {
				// Fetch user config
				const configResponse = await fetch(ApiRoute.getUserConfig);
				if (!configResponse.ok) throw new Error("Failed to fetch user config");
				const configData: UserConfig = await configResponse.json();
				const initialSymbol = Object.keys(configData.symbol)[0] || "XAUUSD";
				setConfig(configData);
				setSymbolInput(initialSymbol);

				// Attempt to fetch MT5 data with initial symbol
				await fetchMT5Data(initialSymbol);

				// Fetch event map
				const eventResponse = await fetch(ApiRoute.getEventMap);
				if (!eventResponse.ok) throw new Error("Failed to fetch event map");
				const eventData: EventMap = await eventResponse.json();
				setEventMap(eventData);

				// Fetch currency map
				const currencyResponse = await fetch(ApiRoute.getCurrencyMap);
				if (!currencyResponse.ok)
					throw new Error("Failed to fetch currency map");
				const currencyData: CurrencyMap = await currencyResponse.json();
				setCurrencyMap(currencyData);
			} catch (err) {
				setError((err as Error).message);
			} finally {
				setLoading(false);
			}
		}
		fetchData();
	}, [account, accPassword, accServer]);

	// Handle input changes
	const handleInputChange = (field: keyof SymbolConfig, value: number) => {
		if (!config || !mt5Data) return;
		const symbol = symbolInput;
		const minStopDistance = mt5Data.trade_stops_level * mt5Data.point + 10;

		if (field === "stop_loss_max" && value < minStopDistance) {
			showNotifyToast(
				`Stop Loss Max cannot be less than ${minStopDistance}`,
				"error"
			);
			return;
		}
		if (
			field === "volume" &&
			(value < mt5Data.volume_min || value > mt5Data.volume_max)
		) {
			showNotifyToast(
				`Volume must be between ${mt5Data.volume_min} and ${mt5Data.volume_max}`,
				"error"
			);
			return;
		}

		const updatedConfig = {
			...config,
			symbol: {
				...config.symbol,
				[symbol]: {
					...config.symbol[symbol],
					[field]: value,
				},
			},
		};
		setConfig(updatedConfig);
	};

	// Handle symbol change from main UI
	const handleSymbolChange = async (newSymbol: string) => {
		if (!config) return;
		const oldSymbol = symbolInput;
		const updatedConfig = {
			...config,
			symbol: {
				...config.symbol,
				[newSymbol]: config.symbol[oldSymbol] || {
					point: 100,
					max_spread: 50,
					transaction_fee: 10,
					over_night_penalty: 10,
					stop_loss_max: 300,
					profit_taken_max: 2500,
					max_current_holding: 100,
					limit_order: true,
					limit_order_expiration: 5,
					volume: 0.1,
				},
			},
		};
		if (newSymbol !== oldSymbol) delete updatedConfig.symbol[oldSymbol];
		setConfig(updatedConfig);
		setSymbolInput(newSymbol);
		await fetchMT5Data(newSymbol);
	};

	// Handle symbol submission from modal
	const handleModalSymbolSubmit = async () => {
		const newSymbol = tempSymbol.toUpperCase();
		if (!newSymbol) {
			showNotifyToast("Please enter a symbol", "error");
			return;
		}
		await handleSymbolChange(newSymbol);
		if (mt5Data) setShowSymbolModal(false); // Close modal only if fetch succeeds
	};

	// Fetch MT5 data for a specific symbol
	const fetchMT5Data = async (symbol: string) => {
		if (!account || !accPassword || !accServer) {
			setError("Please provide MT5 account credentials");
			setLoading(false);
			return;
		}

		try {
			const mt5Response = await fetch(
				ApiRoute.getMT5Info(account, accPassword, accServer, symbol)
			);
			if (!mt5Response.ok) throw new Error("Failed to fetch MT5 data");
			const mt5Result = await mt5Response.json();
			if (mt5Result.status !== "success") {
				showNotifyToast(
					`Symbol "${symbol}" not found. Please enter a valid symbol for your broker.`,
					"error"
				);
				setShowSymbolModal(true); // Show modal on symbol failure
				setTempSymbol(symbol); // Pre-fill with the failed symbol
				setMt5Data(null);
				return;
			}
			setMt5Data(mt5Result.data);
		} catch (err) {
			showNotifyToast("Error fetching MT5 data", "error");
			setShowSymbolModal(true);
			setTempSymbol(symbol);
			setMt5Data(null);
		}
	};

	// Save changes to backend
	const saveChanges = async () => {
		if (!config || !mt5Data) return;
		try {
			const response = await fetch(ApiRoute.getUserConfig, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(config),
			});
			if (!response.ok) throw new Error("Failed to save settings");
			showNotifyToast("Settings saved successfully!", "success");
		} catch (err) {
			setError((err as Error).message);
		}
	};

	if (loading) {
		return (
			<div className="flex items-center justify-center bg-gray-100">
				<div className="text-center">
					<div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500 mx-auto"></div>
					<p className="mt-4 text-lg font-semibold text-gray-700">
						Loading Settings...
					</p>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="flex items-center justify-center bg-gray-100">
				<div className="bg-red-50 p-6 rounded-lg shadow-lg text-center">
					<p className="text-lg font-semibold text-red-600">Error</p>
					<p className="mt-2 text-sm text-red-500">{error}</p>
				</div>
			</div>
		);
	}

	const symbol = symbolInput;
	const minStopDistance = mt5Data
		? mt5Data.trade_stops_level * mt5Data.point + 10
		: 0;

	return (
		<div className="bg-gradient-to-br from-gray-50 to-gray-200 p-6">
			<div className="max-w-5xl mx-auto">
				{/* Symbol Modal */}
				{showSymbolModal && (
					<div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
						<div className="bg-white p-6 rounded-xl shadow-2xl max-w-md w-full transform transition-all duration-300 scale-100">
							<h2 className="text-xl font-bold text-gray-800 mb-4">
								Set Trading Symbol
							</h2>
							<p className="text-sm text-gray-600 mb-4">
								The default symbol was not found in your MT5 account. Please
								enter a valid symbol (e.g., XAUUSD, XAUUSD.sml).
							</p>
							<input
								type="text"
								value={tempSymbol}
								onChange={(e) => setTempSymbol(e.target.value)}
								className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
								placeholder="Enter symbol"
							/>
							<button
								onClick={handleModalSymbolSubmit}
								className="mt-4 w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200"
							>
								Submit
							</button>
						</div>
					</div>
				)}

				{/* Main UI */}
				{mt5Data && (
					<div className="bg-white rounded-xl shadow-lg overflow-hidden">
						{/* Tabs */}
						<div className="flex border-b border-gray-200">
							<button
								className={`flex-1 py-4 px-6 text-center font-semibold text-gray-700 transition-colors duration-200 ${
									activeTab === "trade"
										? "bg-blue-50 text-blue-600 border-b-2 border-blue-600"
										: "hover:bg-gray-100"
								}`}
								onClick={() => setActiveTab("trade")}
							>
								Trade Settings
							</button>
							<button
								className={`flex-1 py-4 px-6 text-center font-semibold text-gray-700 transition-colors duration-200 ${
									activeTab === "indicators"
										? "bg-blue-50 text-blue-600 border-b-2 border-blue-600"
										: "hover:bg-gray-100"
								}`}
								onClick={() => setActiveTab("indicators")}
							>
								Technical Indicators
							</button>
							<button
								className={`flex-1 py-4 px-6 text-center font-semibold text-gray-700 transition-colors duration-200 ${
									activeTab === "economic"
										? "bg-blue-50 text-blue-600 border-b-2 border-blue-600"
										: "hover:bg-gray-100"
								}`}
								onClick={() => setActiveTab("economic")}
							>
								Economic Data
							</button>
						</div>

						{/* Tab Content */}
						<div className="p-6">
							{activeTab === "trade" && (
								<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
									{/* Trade Configuration */}
									<div className="bg-gray-50 p-6 rounded-lg shadow-sm">
										<h2 className="text-xl font-bold text-gray-800 mb-4">
											Trade Configuration
										</h2>
										<div className="space-y-5">
											<div>
												<label className="block text-sm font-medium text-gray-700 mb-1">
													Symbol
												</label>
												<input
													type="text"
													value={symbolInput}
													onChange={(e) =>
														handleSymbolChange(e.target.value.toUpperCase())
													}
													className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
												/>
											</div>
											<div>
												<label className="block text-sm font-medium text-gray-700 mb-1">
													Volume (Min: {mt5Data.volume_min}, Max:
													{mt5Data.volume_max})
												</label>
												<input
													type="number"
													step="0.01"
													value={config?.symbol[symbol]?.volume ?? 0.1}
													onChange={(e) =>
														handleInputChange("volume", Number(e.target.value))
													}
													className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
													min={mt5Data.volume_min}
													max={mt5Data.volume_max}
												/>
											</div>
											<div>
												<label className="block text-sm font-medium text-gray-700 mb-1">
													Stop Loss Max (Min: {minStopDistance})
												</label>
												<input
													type="number"
													value={config?.symbol[symbol].stop_loss_max}
													onChange={(e) =>
														handleInputChange(
															"stop_loss_max",
															Number(e.target.value)
														)
													}
													className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
													min={minStopDistance}
												/>
											</div>
											<div>
												<label className="block text-sm font-medium text-gray-700 mb-1">
													Profit Taken Max
												</label>
												<input
													type="number"
													value={config?.symbol[symbol].profit_taken_max}
													onChange={(e) =>
														handleInputChange(
															"profit_taken_max",
															Number(e.target.value)
														)
													}
													className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
													min={0}
												/>
											</div>
											<button
												onClick={saveChanges}
												className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors duration-200"
											>
												Save Changes
											</button>
										</div>
									</div>

									{/* MT5 Configuration */}
									<div className="bg-gray-50 p-6 rounded-lg shadow-sm">
										<h2 className="text-xl font-bold text-gray-800 mb-4">
											MT5 Configuration
											<span className="text-sm font-normal text-gray-500">
												(Read-Only)
											</span>
										</h2>
										<div className="space-y-3 text-sm">
											<div className="flex justify-between">
												<span className="text-gray-600">Account User Name</span>
												<span className="font-medium text-gray-800">
													{mt5Data.name}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Account Login ID</span>
												<span className="font-medium text-gray-800">
													{mt5Data.login}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Account Server</span>
												<span className="font-medium text-gray-800">
													{mt5Data.server}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Broker Company</span>
												<span className="font-medium text-gray-800">
													{mt5Data.company}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Account Leverage</span>
												<span className="font-medium text-gray-800">
													{mt5Data.leverage}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Symbol</span>
												<span className="font-medium text-gray-800">
													{symbol}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Point Value</span>
												<span className="font-medium text-gray-800">
													{mt5Data.point}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Minimum Volume</span>
												<span className="font-medium text-gray-800">
													{mt5Data.volume_min}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">Maximum Volume</span>
												<span className="font-medium text-gray-800">
													{mt5Data.volume_max}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">
													Overnight Fee (Long)
												</span>
												<span className="font-medium text-gray-800">
													${mt5Data.over_night_penalty_long}
												</span>
											</div>
											<div className="flex justify-between">
												<span className="text-gray-600">
													Overnight Fee (Short)
												</span>
												<span className="font-medium text-gray-800">
													${mt5Data.over_night_penalty_short}
												</span>
											</div>
										</div>
									</div>
								</div>
							)}

							{activeTab === "indicators" && (
								<div className="bg-gray-50 p-6 rounded-lg shadow-sm">
									<h2 className="text-xl font-bold text-gray-800 mb-4">
										Trading Indicators
									</h2>
									<p className="text-sm text-gray-600 mb-4">
										These are the market signals used by the trading model to
										decide when to buy or sell:
									</p>
									<div className="max-h-96 overflow-y-auto">
										<ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
											{config?.env.observation_list.map((indicator) => (
												<li key={indicator}>
													{indicatorDescriptions[indicator] || indicator}
												</li>
											))}
										</ul>
									</div>
								</div>
							)}

							{activeTab === "economic" && (
								<div className="bg-gray-50 p-6 rounded-lg shadow-sm">
									<h2 className="text-xl font-bold text-gray-800 mb-4">
										Economic Data
									</h2>
									<p className="text-sm text-gray-600 mb-4">
										These are the economic events and currencies that the
										trading model considers when making decisions:
									</p>
									<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
										<div>
											<h3 className="text-lg font-semibold text-gray-700 mb-3">
												Economic Events
											</h3>
											{eventMap ? (
												<div className="max-h-96 overflow-y-auto">
													<ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
														{Object.keys(eventMap).map((event) => (
															<li key={event}>{event}</li>
														))}
													</ul>
												</div>
											) : (
												<p className="text-sm text-gray-500 italic">
													Loading events...
												</p>
											)}
										</div>
										<div>
											<h3 className="text-lg font-semibold text-gray-700 mb-3">
												Currencies
											</h3>
											{currencyMap ? (
												<div className="max-h-96 overflow-y-auto">
													<ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
														{Object.keys(currencyMap).map((currency) => (
															<li key={currency}>{currency}</li>
														))}
													</ul>
												</div>
											) : (
												<p className="text-sm text-gray-500 italic">
													Loading currencies...
												</p>
											)}
										</div>
									</div>
								</div>
							)}
						</div>
					</div>
				)}
			</div>
		</div>
	);
};

export default Settings;
