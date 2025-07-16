// src/app/api/index.ts
// Use window.location.origin as the base URL in client-side code
const baseUrl =
	typeof window !== "undefined"
		? window.location.origin
		: "http://localhost:3000";

export const ApiRoute = {
	posts: "/api/posts",
	getTrades: `${baseUrl}/api/trades`,
	signup: "/api/auth/signup",
	login: "/api/auth/login",
	logout: "/api/auth/logout",
	getAccount: (userId: number) => `/api/account?userId=${userId}`,
	postAccount: "/api/account",
	getAccountSummary: (accountId: number) =>
		`/api/account-summary?accountId=${accountId}`,
	getAccountStatistics: (accountId: number) =>
		`/api/account-statistics?accountId=${accountId}`,
	getTradePerformance: (
		accountId: number,
		startDate: string | null = null, // Default to null
		endDate: string | null = null // Default to null
	) =>
		`/api/trade-performance?accountId=${accountId}&start=${startDate}&end=${endDate}`,
	getLongShortTrades: (accountId: number) =>
		`/api/long-short-trades?accountId=${accountId}`,
	getMonthlyTradeStats: (accountId: number) =>
		`/api/monthly-trade-stats?accountId=${accountId}`,
	getLineChartTrades: (accountId: number) =>
		`/api/trade-line-chart?accountId=${accountId}`,
	getMonthlyStatistics: (month: string, accountId: number) =>
		`/api/monthly-statistics?month=${month}&accountId=${accountId}`,
	getAccountBalance: (
		accountId: number,
		start: string | null = null, // Default to null
		end: string | null = null // Default to null
	) => `/api/account-balance?accountId=${accountId}&start=${start}&end=${end}`,
	getCummulativePnL: (
		accountId: number,
		start: string | null = null, // Default to null
		end: string | null = null // Default to null
	) => `/api/cumulative-pnl?accountId=${accountId}&start=${start}&end=${end}`,
	getDrawdown: (
		accountId: number,
		start: string | null = null, // Default to null
		end: string | null = null // Default to null
	) => `/api/account-drawdown?accountId=${accountId}&start=${start}&end=${end}`,
	getMonthlyTradePerformance: (accountId: number) =>
		`/api/trade-performance-by-month?accountId=${accountId}`,
	getUserConfig: "/api/user-config",

	ws_url: "ws://localhost:4000",

	postMT5Login: "http://localhost:5000/api/mt5-login",
	getMT5AccountInfo: (
		userId: string,
		loginId: string,
		password: string,
		server: string
	) =>
		`http://localhost:5000/api/mt5-account-info?userId=${userId}&account=${loginId}&password=${password}&server=${server}`,
	getAccountId: "/api/account-id",
	getTradingStatus: "http://localhost:5000/api/trading/status",
	postTradingStart: "http://localhost:5000/api/trading/start",
	postTradingStop: "http://localhost:5000/api/trading/stop",
	getEventMap: "http://localhost:5000/api/event-map",
	getCurrencyMap: "http://localhost:5000/api/currency-map",
	getMT5Info: (
		account: number,
		password: string,
		server: string,
		symbol: string
	) =>
		`http://localhost:5000/api/mt5/info?account=${account}&password=${password}&server=${server}&symbol=${symbol}`,
	algoTradingStatus: "http://localhost:5000/api/algo-trading-status",
} as const;
