export type Account = {
	id: string;
	loginId: number;
	password: string;
	server: string;
};

export type User = {
	id: number;
	email: string;
	password: string;
};

export type Trade = {
	no?: number;
	ticket: number; // INTEGER - Order ID
	accountId: number; // INTEGER - User Account ID (FK)
	entry_time: string; // TEXT - Entry date and time
	direction: 0 | 1; // INTEGER - 0=Buy, 1=Sell
	lot_size: number; // REAL - Lot size
	currency_pair: string; // TEXT - Symbol (e.g., XAUUSD)
	modal: string; // VARCHAR(50) - Modal name
	timeframe: string; // TEXT - Timeframe (e.g., H1)
	entry_price: number; // REAL - Entry price
	stop_loss: number; // REAL - Stop-loss price
	take_profit: number; // REAL - Take-profit price
	one_r_pips: number; // REAL - 1R in pips (SL distance)
	pips_value: number; // REAL - Value per pip
	rate: number; // REAL - Exchange rate (if applicable)
	required_margin: number; // REAL - Margin required
	position_size: number; // REAL - Position size in units
	max_profit: number; // REAL - Max profit (TP - Entry)
	max_loss: number; // REAL - Max loss (Entry - SL)
	risk_percent: number; // REAL - Risk % per trade
	max_risk_reward: number; // REAL - Max risk-reward ratio
	exit_time?: string | null; // TEXT - Exit date/time (nullable if trade is open)
	exit_price?: number | null; // REAL - Exit price (nullable if trade is open)
	commission: number; // REAL - Commission charged
	duration: string; // TEXT - Duration (e.g., '2:30')
	r_multiple: number; // REAL - R multiple (profit/1R)
	pips: number; // REAL - Pips gained/lost
	profit_loss: number; // REAL - Profit/loss including fees
	cumulative_pnl: number; // REAL - Cumulative P&L
	win_loss: 1 | 0 | null; // INTEGER - 1=Win, 0=Loss, NULL=Open
	status: "Open" | "Closed"; // TEXT - Status (Open, Closed)
	account_balance: number; // REAL - Balance after trade
	percent_gain_loss: number; // REAL - % gain/loss
	drawdown: number; // REAL - Drawdown percentage
};

export type TechnicalIndicators = {
	trade_ticket: number;
	indicator_name: string;
	indicator_value: number;
};

export type EconomicCalendar = {
	trade_ticket: number;
	event: string;
	event_currency: string;
	actaul: number;
	forecast: number;
	previous: number;
	impact: number;
};

export interface ColumnConfig<T> {
	label: string;
	key: keyof T;
	width?: string;
	rowHeader?: boolean;
	badge?: boolean;
	badgeTheme?: (
		value: T[keyof T]
	) =>
		| "Gray"
		| "Red"
		| "Yellow"
		| "Green"
		| "Blue"
		| "Indigo"
		| "Purple"
		| "Pink";
	isMoney?: boolean; // Will add "$" to the value
	enableNumberColor?: boolean; // Enables red/green text color for numbers
	colorTheme?: any;
	isPercentage?: boolean; // Will format the value as a percentage (e.g., "50%")
	cellTheme?: any;
	render?: (value: T[keyof T], row: T) => React.ReactNode;
}

// Define the Account Summary type
export interface AccountSummary {
	startingCapital: number;
	accountBalance: number;
	profitLoss: number;
	percentGainLoss: number;
	winningRate: number;
	totalRMultiple: number;
	expectancyPerTrade: number;
}

// Define the Account Statistics type
export interface AccountStatistics {
	averageWin: number;
	averageLoss: number;
	largestWin: number;
	largestLoss: number;
	maxConsecutiveWins: number;
	maxConsecutiveLosses: number;
	maxDrawdown: number;
	longTrades: number;
	shortTrades: number;
	avgTimeInTrade: string;
}

export interface TradePerformance {
	winningTrades: number;
	losingTrades: number;
	breakEvenTrades: number;
}

export interface LongShortTrades {
	longTrades: number;
	shortTrades: number;
}

export interface MonthlyTradeStats {
	month: string; // e.g., "Jan"
	profit: number;
	rMultiple: number;
	winningTrades: number;
	losingTrades: number;
	breakEvenTrades: number;
}

export interface ChartTrade {
	ticket: number;
	entry_time: string;
	exit_time: string | null;
	entry_price: number;
	exit_price: number | null;
	direction: number; // 0 = Buy, 1 = Sell
	profit_loss: number | null; // REAL - Profit/loss including fees
}

export interface MonthlyStatistics {
	accountStartingBalance: number;
	accountEndingBalance: number;
	profitLoss: number;
	percentGainLoss: number;
	winningRate: number;
	totalRMultiple: number;
	expectancyPerTrade: number;
	avgTimeInTrade: string;
}
