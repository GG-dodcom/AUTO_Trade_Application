// src/data/tableColumns.tsx
import { ColumnConfig, Trade } from "@/lib/definitions";

export const monthTradeStatColumns = [
	{ label: "Month", key: "month" },
	{ label: "Profit", key: "profit", isMoney: true },
	{ label: "R Multiple", key: "r_mult" },
	{ label: "Winning Trade", key: "win_trade" },
	{ label: "Losing Trade", key: "loss_trade" },
	{ label: "B/E", key: "breakeven" },
];

export const monthTradeStat_Header1Groups = [
	{
		header: "MONTHLY TRADE STATISTICS",
		keys: [
			"Month",
			"Profit",
			"R Multiple",
			"Winning Trade",
			"Losing Trade",
			"B/E",
		],
	},
];

export const dummyMonthTradeStatData = [
	{
		month: "Jan",
		profit: 1200,
		r_mult: 1.5,
		win_trade: 8,
		loss_trade: 4,
		breakeven: 2,
	},
	{
		month: "Feb",
		profit: 1500,
		r_mult: 2.1,
		win_trade: 10,
		loss_trade: 3,
		breakeven: 1,
	},
	{
		month: "Mar",
		profit: -500,
		r_mult: -0.8,
		win_trade: 5,
		loss_trade: 7,
		breakeven: 2,
	},
	{
		month: "Apr",
		profit: 2300,
		r_mult: 3.2,
		win_trade: 12,
		loss_trade: 4,
		breakeven: 1,
	},
	{
		month: "May",
		profit: 1800,
		r_mult: 2.4,
		win_trade: 9,
		loss_trade: 5,
		breakeven: 3,
	},
	{
		month: "Jun",
		profit: -700,
		r_mult: -1.1,
		win_trade: 6,
		loss_trade: 8,
		breakeven: 1,
	},
	{
		month: "Jul",
		profit: 2500,
		r_mult: 3.5,
		win_trade: 13,
		loss_trade: 3,
		breakeven: 2,
	},
	{
		month: "Aug",
		profit: 1400,
		r_mult: 1.8,
		win_trade: 8,
		loss_trade: 6,
		breakeven: 2,
	},
	{
		month: "Sep",
		profit: 1700,
		r_mult: 2.2,
		win_trade: 10,
		loss_trade: 5,
		breakeven: 1,
	},
	{
		month: "Oct",
		profit: -300,
		r_mult: -0.5,
		win_trade: 6,
		loss_trade: 7,
		breakeven: 3,
	},
	{
		month: "Nov",
		profit: 2000,
		r_mult: 2.7,
		win_trade: 11,
		loss_trade: 4,
		breakeven: 1,
	},
	{
		month: "Dec",
		profit: 2600,
		r_mult: 3.8,
		win_trade: 14,
		loss_trade: 3,
		breakeven: 1,
	},
	{
		month: "Total",
		profit: 2600,
		r_mult: 19.86,
		win_trade: 14,
		loss_trade: 6,
		breakeven: 1,
	},
];

export const accStatColumns = [
	{ label: "Average Win", key: "avg_win", isMoney: true },
	{ label: "Average Loss", key: "avg_loss", isMoney: true },
	{
		label: "Largest Win",
		key: "largest_win",
		isMoney: true,
	},
	{
		label: "Largest Loss",
		key: "largest_loss",
		isMoney: true,
	},
	{ label: "Max Consec. Wins", key: "max_consec_wins" },
	{
		label: "Max Consec. Loss",
		key: "max_consec_loss",
	},
	{
		label: "Max Drowdown",
		key: "max_dd",
		isPercentage: true,
	},
	{ label: "Long Trade", key: "long_trade" },
	{ label: "Short Trade", key: "short_trade" },
	{ label: "Avg. Time in Trade", key: "avg_time_trade" },
];

export const dummyAccStatData = [
	{
		avg_win: 1500,
		avg_loss: -800,
		largest_win: 5000,
		largest_loss: -3000,
		max_consec_wins: 7,
		max_consec_loss: 3,
		max_dd: -12.5,
		long_trade: 60,
		short_trade: 40,
		avg_time_trade: "3h 45m",
	},
];

export const accSumColumns = [
	{ label: "Starting Capital", key: "start_cap", isMoney: true },
	{ label: "Account Balance", key: "acc_bal", isMoney: true },
	{
		label: "Profit / Loss",
		key: "profit_loss",
		isMoney: true,
		enableNumberColor: true,
	},
	{
		label: "% Gain / Loss",
		key: "gain_loss_pct",
		isPercentage: true,
		enableNumberColor: true,
	},
	{ label: "Winning Rate", key: "win_rate", isPercentage: true },
	{ label: "Total R Multiple", key: "total_r_mult", enableNumberColor: true },
	{
		label: "Expectancy per Trade",
		key: "exp_per_trade",
		enableNumberColor: true,
	},
];

export const dummyAccSumData = [
	{
		start_cap: 10000,
		acc_bal: 12500,
		profit_loss: 2500,
		gain_loss_pct: "25",
		win_rate: "60",
		total_r_mult: 5.2,
		exp_per_trade: 1.3,
	},
];

export const JournalColumns: ColumnConfig<Trade>[] = [
	{ label: "No", key: "no" as const },
	{ label: "ORDER ID", key: "ticket" as const, rowHeader: true },
	{ label: "ENTRY DATE / TIME", key: "entry_time" as const },
	{
		label: "DIRECTION",
		key: "direction" as const,
		badge: true,
		render: (
			value: string | number | null | undefined,
			row: { direction: 0 | 1 }
		) => (value === 0 ? "BUY" : value === 1 ? "SELL" : value == null ? "" : ""), // Handle null or undefined values
		badgeTheme: (value: string | number | null | undefined) =>
			value === 0 ? "Green" : value === 1 ? "Red" : "Gray",
	},
	{ label: "LOT SIZE", key: "lot_size" as const },
	{ label: "CURRENCY PAIR", key: "currency_pair" as const },
	{ label: "MODAL", key: "modal" as const }, // Added modal field
	{ label: "TIME FRAME", key: "timeframe" as const },
	{ label: "ENTRY PRICE", key: "entry_price" as const },
	{ label: "STOP LOSS", key: "stop_loss" as const },
	{ label: "TAKE PROFIT", key: "take_profit" as const },
	{ label: "1R (PIPS)", key: "one_r_pips" as const },
	{
		label: "PIPS VALUE",
		key: "pips_value" as const,
		isMoney: true, // Will add "$" to the value
	},
	{ label: "RATE", key: "rate" as const, isMoney: true },
	{ label: "REQUIRED MARGIN", key: "required_margin" as const, isMoney: true },
	{ label: "POSITION SIZE", key: "position_size" as const, isMoney: true },
	{ label: "MAX PROFIT", key: "max_profit" as const, isMoney: true },
	{ label: "MAX LOSS", key: "max_loss" as const, isMoney: true },
	{ label: "RISK % / TRADE", key: "risk_percent" as const, isPercentage: true },
	{ label: "MAX RISK-REWARD", key: "max_risk_reward" as const },
	{ label: "EXIT DATE / TIME", key: "exit_time" as const },
	{ label: "EXIT PRICE", key: "exit_price" as const },
	{ label: "COMMISSION", key: "commission" as const, isMoney: true },
	{ label: "HOUR:MINUTE", key: "duration" as const },
	{ label: "R MULTIPLE", key: "r_multiple" as const, enableNumberColor: true },
	{ label: "PIPS", key: "pips" as const },
	{
		label: "PROFIT / LOSS (INC. FEES)",
		key: "profit_loss" as const,
		isMoney: true,
		enableNumberColor: true,
	},
	{
		label: "CUMULATIVE P&L",
		key: "cumulative_pnl" as const,
		isMoney: true,
		enableNumberColor: true,
	},
	{
		label: "WIN / LOSS",
		key: "win_loss" as const,
		render: (value: string | number | null | undefined) =>
			value === 1 ? "Win" : value === 0 ? "Loss" : value == null ? "Open" : "", // Handle undefined and other cases
		colorTheme: (value: 1 | 0 | null) => {
			const displayValue = value === 1 ? "Win" : value === 0 ? "Loss" : "Open";
			return displayValue === "Win"
				? "text-green-600 text-sm font-semibold"
				: displayValue === "Loss"
				? "text-red-600 text-sm font-semibold"
				: displayValue === "Open"
				? "text-gray-600 text-sm font-semibold"
				: "";
		},
		cellTheme: (value: 1 | 0 | null) => {
			const displayValue = value === 1 ? "Win" : value === 0 ? "Loss" : "Open";
			return displayValue === "Win"
				? "bg-green-50"
				: displayValue === "Loss"
				? "bg-red-50"
				: displayValue === "Open"
				? "bg-gray-50"
				: "";
		},
	},
	{
		label: "STATUS",
		key: "status" as const,
		badge: true,
		badgeTheme: (value: string | number | null | undefined) =>
			value === "Open" ? "Blue" : value === "Closed" ? "Gray" : "Gray",
		colorTheme: (value: "Open" | "Closed") =>
			value === "Open"
				? "text-blue-400"
				: value === "Closed"
				? "text-gray-400"
				: "",
	},
	{ label: "ACCOUNT BALANCE", key: "account_balance" as const, isMoney: true },
	{
		label: "% GAIN / LOSS",
		key: "percent_gain_loss" as const,
		isPercentage: true,
		enableNumberColor: true,
	},
	{
		label: "DRAWDOWN",
		key: "drawdown" as const,
		isPercentage: true,
		enableNumberColor: true,
	},

	// {
	// 	label: "Actions",
	// 	key: "actions",
	// 	width: "w-1/6",
	// 	render: (_: any, row: any) => (
	// 		<a href="#" className="text-blue-600 hover:underline">
	// 			Edit
	// 		</a>
	// 	),
	// },
];

export const journalHeader1Groups = [
	{
		header: "Open Position",
		keys: [
			"no",
			"ticket",
			"entry_time",
			"direction",
			"lot_size",
			"currency_pair",
			"modal",
			"timeframe",
			"entry_price",
			"stop_loss",
			"take_profit",
		],
	},
	{
		header: "Risk Management",
		keys: [
			"one_r_pips",
			"pips_value",
			"rate",
			"required_margin",
			"position_size",
			"max_profit",
			"max_loss",
			"risk_percent",
			"max_risk_reward",
		],
	},
	{ header: "Close Position", keys: ["exit_time", "exit_price", "commission"] },
	{ header: "Duration", keys: ["duration"] },
	{
		header: "Profit & Loss",
		keys: [
			"r_multiple",
			"pips",
			"profit_loss",
			"cumulative_pnl",
			"win_loss",
			"status",
		],
	},
	{
		header: "Account Details",
		keys: ["account_balance", "percent_gain_loss", "drawdown"],
	},
];
