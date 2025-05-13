import patch  # Import the patching module first

from dotenv import load_dotenv
import os
import sqlite3
import MetaTrader5 as mt5
import time
import datetime
from typing import Dict, List, Any
import logging

# Load environment variables from .env file
load_dotenv()

# Get environment variables
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root: {PROJECT_ROOT}")
DB_FILE = os.path.join(PROJECT_ROOT, os.getenv("DB_FILE"))

def init_db():
    """Initialize SQLite database with a comprehensive trades table."""
    try:
        # Connect to the database
        logging.info(f"Connecting to database at {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
    
        # Use executescript to run multiple SQL statements
        cursor.executescript('''
            BEGIN TRANSACTION;

            -- Create the users table
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );

            -- Create the account table
            CREATE TABLE IF NOT EXISTS "account" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT, -- Auto-incrementing column, also a PK
                "userId" INT NOT NULL,
                "server" VARCHAR(255) NOT NULL,
                "loginId" INTEGER NOT NULL,
                "password" VARCHAR(255) NOT NULL,
                "trade_mode" INTEGER,           -- 0: demo, 1: contest, 2: real
                "leverage" INTEGER,
                "limit_orders" INTEGER,
                "margin_so_mode" INTEGER,       -- Stop-out mode (0: percent, 1: absolute value)
                "trade_allowed" INTEGER,        -- 0 or 1 (boolean)
                "trade_expert" INTEGER,         -- 0 or 1 (boolean for expert advisors)
                "margin_mode" INTEGER,          -- Margin calculation mode (0: Forex, 1: CFD, etc.)
                "currency_digits" INTEGER,      -- Digits for currency precision
                "fifo_close" INTEGER,           -- 0 or 1 (boolean for FIFO rule)
                "balance" NUMERIC,
                "credit" NUMERIC,
                "profit" NUMERIC,
                "equity" NUMERIC,
                "margin" NUMERIC,
                "margin_free" NUMERIC,
                "margin_level" NUMERIC,         -- Margin level as percentage
                "margin_so_call" NUMERIC,       -- Margin call level
                "margin_so_so" NUMERIC,         -- Stop-out level
                "margin_initial" NUMERIC,       -- Initial margin requirement
                "margin_maintenance" NUMERIC,   -- Maintenance margin requirement
                "assets" NUMERIC,               -- Total assets
                "liabilities" NUMERIC,          -- Total liabilities
                "commission_blocked" NUMERIC,   -- Blocked commission
                "name" VARCHAR(255),
                "currency" VARCHAR(10),
                "company" VARCHAR(255),         -- Broker company name
                "remember" INTEGER DEFAULT 0,   -- 0 for false, 1 for true
                UNIQUE("userId", "server", "loginId"), -- Enforce uniqueness
                FOREIGN KEY("userId") REFERENCES "users"("id") ON DELETE CASCADE
            );

            -- Create the trades table         
            CREATE TABLE IF NOT EXISTS trades (
                ticket INTEGER PRIMARY KEY,          -- Order ID
                accountId INT,                       -- Foreign key referencing account table
                entry_time TEXT,                     -- Entry date/time
                direction INTEGER,                   -- 0=Buy, 1=Sell
                lot_size REAL,                       -- Lot size
                currency_pair TEXT,                  -- Symbol (e.g., XAUUSD)
                modal VARCHAR(50),                   -- Modal name
                timeframe TEXT,                      -- Timeframe (e.g., H1)
                entry_price REAL,                    -- Entry price
                stop_loss REAL,                      -- Stop loss price
                take_profit REAL,                    -- Take profit price
                one_r_pips REAL,                     -- 1R in pips (SL distance)
                pips_value REAL,                     -- Value per pip
                rate REAL,                           -- Exchange rate (if applicable)
                required_margin REAL,                -- Margin required
                position_size REAL,                  -- Position size in units
                max_profit REAL,                     -- Max profit (TP - Entry)
                max_loss REAL,                       -- Max loss (Entry - SL)
                risk_percent REAL,                   -- Risk % per trade
                max_risk_reward REAL,                -- Max risk-reward ratio
                exit_time TEXT,                      -- Exit date/time
                exit_price REAL,                     -- Exit price
                commission REAL,                     -- Commission charged
                duration TEXT,                       -- Duration (e.g., '2:30')
                r_multiple REAL,                     -- R multiple (profit/1R)
                pips REAL,                           -- Pips gained/lost
                profit_loss REAL,                    -- Profit/loss including fees
                cumulative_pnl REAL,                 -- Cumulative P&L
                win_loss INTEGER,                    -- 1=Win, 0=Loss, NULL=Open
                status TEXT DEFAULT 'Open',          -- Status (Open, Closed)
                account_balance REAL,                -- Balance after trade
                percent_gain_loss REAL,              -- % gain/loss
                drawdown REAL,                       -- Drawdown percentage
                FOREIGN KEY (accountId) REFERENCES account(id) ON DELETE CASCADE
            );
        
            -- Create the technical_indicators table
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique ID for each indicator record
                trade_ticket INTEGER,                  -- Foreign key linking to the trades table
                indicator_name TEXT,                   -- Name of the indicator (e.g., "RSI_14", "SMA_50")
                indicator_value REAL,                  -- Value of the indicator
                FOREIGN KEY (trade_ticket) REFERENCES trades(ticket) ON DELETE CASCADE
            );
            
            -- Create the economic_calendar table
                CREATE TABLE IF NOT EXISTS economic_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,        -- Primary key, auto-incremented
                trade_ticket INTEGER NOT NULL,               -- Foreign key linking to trades table
                event TEXT NOT NULL,                         -- Name of the economic event (e.g., "Non-Farm Payrolls")
                event_currency TEXT NOT NULL,                -- Currency affected by the event (e.g., "USD")
                actual REAL,                                 -- Actual value of the economic indicator
                forecast REAL,                               -- Forecasted value of the economic indicator
                previous REAL,                               -- Previous value of the economic indicator
                impact INTEGER NOT NULL,                     -- Impact level (2: High, 1: Medium, 0: Low)
                FOREIGN KEY (trade_ticket) REFERENCES trades(ticket) ON DELETE CASCADE
            );

            COMMIT;
        ''')

     # Log success
        logging.info("Database initialized successfully")
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
        raise
    finally:
        conn.close()

def save_trade_to_db(accountId, trade_data):
    """Save trade action to SQLite trades table with comprehensive data."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Fetch account balance and equity before trade
            account_info = mt5.account_info()
            if account_info is None:
                raise ValueError("Failed to fetch account info from MT5")
            initial_balance = account_info.balance
            
            # Extract data from trade_data
            ticket = trade_data['ticket']
            symbol = trade_data['symbol']
            direction = trade_data['action']  # 0=Buy, 1=Sell
            lot_size = trade_data['volume']
            entry_price = trade_data['open_price']
            sl = trade_data['sl']
            tp = trade_data['tp']
            entry_time = trade_data['open_time']
            timeframe = 'H1'  # Hardcoded for now; adjust if dynamic
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Failed to fetch symbol info for {symbol}")
            point = symbol_info.point
            pip_value = symbol_info.trade_tick_value  # Value per pip
            
            # Calculate 1R (pips), max profit, max loss
            one_r_pips = abs(entry_price - sl) / point  # Distance to SL in pips
            max_profit = abs(tp - entry_price) * lot_size * pip_value  # In account currency
            max_loss = abs(entry_price - sl) * lot_size * pip_value    # In account currency
            
            # Risk % per trade (assuming 1% risk as default; adjust as needed)
            risk_percent = (max_loss / initial_balance) * 100 if initial_balance > 0 else 0
            
            # Max risk-reward ratio
            max_risk_reward = max_profit / max_loss if max_loss > 0 else 0
            
            # Required margin and position size
            margin = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY if direction == 0 else mt5.ORDER_TYPE_SELL,
                symbol, lot_size, entry_price
            )
            position_size = lot_size * symbol_info.trade_contract_size  # Units
            
            # Insert initial trade data
            cursor.execute('''
                INSERT OR REPLACE INTO trades (
                    ticket, accountId, entry_time, direction, lot_size, currency_pair, timeframe,
                    entry_price, stop_loss, take_profit, one_r_pips, pips_value, rate,
                    required_margin, position_size, max_profit, max_loss, risk_percent,
                    max_risk_reward, account_balance, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket, accountId, entry_time, direction, lot_size, symbol, timeframe,
                entry_price, sl, tp, one_r_pips, pip_value, 1.0,  # Rate=1.0 (adjust if cross-currency)
                margin, position_size, max_profit, max_loss, risk_percent,
                max_risk_reward, initial_balance, 'Open'
            ))
            conn.commit()
            logging.info(f"Trade saved: {ticket}, Account ID: {accountId}, Symbol: {symbol}, Direction: {direction}, Lot Size: {lot_size}")
    except sqlite3.OperationalError as e:
        logging.error(f"Database error in save_trade_to_db: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in save_trade_to_db: {str(e)}")
        raise
    
def update_closed_trade_in_db(ticket, account_id):
    """Update trades with closed trade details from MT5."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Query matches schema: ticket, accountId, status
            cursor.execute("SELECT * FROM trades WHERE ticket = ? AND status = 'Open' AND accountId = ?", 
                                            (ticket, account_id))
            trade = cursor.fetchone()
            if not trade:
                    logging.info(f"No open trade found for ticket {ticket} with accountId {account_id}")
                    return
            
            logging.debug(f"Found open trade: {trade}")
            history_orders = mt5.history_orders_get(ticket=ticket)
            if not history_orders or len(history_orders) == 0:
                    logging.info(f"No history orders found for ticket {ticket}")
                    return
            
            order = history_orders[0]
            if order.state not in [mt5.ORDER_STATE_FILLED, mt5.ORDER_STATE_CANCELED, mt5.ORDER_STATE_EXPIRED]:
                    logging.debug(f"Order {ticket} state {order.state} not final")
                    return
            
            position_id = order.position_id
            deals = mt5.history_deals_get(position=position_id)
            if not deals or len(deals) == 0:
                    now = datetime.datetime.now()
                    from_date = now - datetime.timedelta(days=7)
                    deals = mt5.history_deals_get(from_date, now, position=position_id)
                    if not deals:
                            logging.warning(f"No deals found for position_id {position_id}")
                            return
            
            closing_deal = None
            for deal in deals:
                    if deal.position_id == position_id and deal.entry == 1:  # DEAL_ENTRY_OUT
                            closing_deal = deal
                            break
            if not closing_deal:
                    logging.info(f"Position {position_id} not yet closed")
                    return
            
            exit_price = closing_deal.price
            profit_loss = closing_deal.profit
            commission = closing_deal.commission + closing_deal.fee
            exit_time = datetime.datetime.fromtimestamp(closing_deal.time, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Unpack trade tuple (33 columns)
            (ticket, account_id, entry_time, direction, lot_size, currency_pair, modal, timeframe, entry_price, 
                    stop_loss, take_profit, one_r_pips, pips_value, rate, required_margin, position_size, max_profit, 
                    max_loss, risk_percent, max_risk_reward, exit_time_db, exit_price_db, commission_db, duration_db, 
                    r_multiple_db, pips_db, profit_loss_db, cumulative_pnl_db, win_loss_db, status, account_balance_db, 
                    percent_gain_loss_db, drawdown_db) = trade

            # Duration
            entry_dt = datetime.datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
            exit_dt = datetime.datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S')
            duration_delta = exit_dt - entry_dt
            duration_hours = duration_delta.total_seconds() // 3600
            duration_minutes = (duration_delta.total_seconds() % 3600) // 60
            duration = f"{int(duration_hours)}:{int(duration_minutes):02d}"

            # Pips with safeguard
            symbol_info = mt5.symbol_info(currency_pair)
            point = symbol_info.point if symbol_info and symbol_info.point else 0.01
            pips = ((exit_price - entry_price) / point if direction == 0 else (entry_price - exit_price) / point)

            # Net Profit/Loss
            profit_loss_net = profit_loss - commission
            logging.debug(f"Ticket: {ticket}, Gross P/L: {profit_loss}, Commission: {commission}, Net P/L: {profit_loss_net}")

            # R-Multiple
            r_multiple = profit_loss_net / (one_r_pips * pips_value) if one_r_pips > 0 else 0

            # Cumulative P&L
            cursor.execute("SELECT SUM(profit_loss) FROM trades WHERE profit_loss IS NOT NULL")
            prev_cumulative_pnl = cursor.fetchone()[0] or 0
            cumulative_pnl = prev_cumulative_pnl + profit_loss_net

            # Current Balance
            current_balance = account_balance_db + profit_loss_net

            # Percent Gain/Loss
            percent_gain_loss = (profit_loss_net / account_balance_db) * 100 if account_balance_db > 0 else 0

            # Win/Loss
            win_loss = 1 if profit_loss_net > 0 else 0

            # Drawdown (corrected to max risk %)
            drawdown = (abs(max_loss) / account_balance_db) * 100 if account_balance_db > 0 else 0

            # Update database
            cursor.execute(
            '''
                UPDATE trades
                SET exit_time = ?, exit_price = ?, commission = ?, duration = ?, r_multiple = ?,
                        pips = ?, profit_loss = ?, cumulative_pnl = ?, win_loss = ?, status = ?,
                        account_balance = ?, percent_gain_loss = ?, drawdown = ?
                WHERE ticket = ?
            ''', 
            (
                    exit_time, exit_price, commission, duration, r_multiple, pips, profit_loss_net,
                    cumulative_pnl, win_loss, 'Closed', current_balance, percent_gain_loss, drawdown,
                    ticket
            ))
            logging.info(f"Trade {ticket} closed: Pips={pips}, Profit/Loss={profit_loss_net}, R={r_multiple}")
            conn.commit()
    except Exception as e:
        logging.error(f"Error updating trade {ticket}: {str(e)}")
        raise

def save_trade_indicators_and_events(
    row_data: Dict[str, Any],
    trade_ticket: str
) -> None:
    """
    Save technical indicators and economic calendar data for a trade to the database.
    Only columns without '_norm' suffix are saved.

    Args:
        row_data (Dict[str, Any]): A dictionary containing a single row of data with columns as keys.
        trade_ticket (str): The trade ticket (order ID) as a string.
    """
    # Log the input data for debugging
    logging.debug(f"Starting save_trade_indicators_and_events with trade_ticket: {trade_ticket}")
    logging.debug(f"Input row_data: {row_data}")

    # Validate trade ticket
    if not trade_ticket:
        logging.info("No trade ticket provided, skipping save.")
        return

    try:
        # Log database connection attempt
        logging.debug(f"Connecting to database: {DB_FILE}")
        with sqlite3.connect(DB_FILE) as conn:
            # Enable foreign key support
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()
            logging.debug("Database connection established and foreign keys enabled")

            # Track counts for logging
            indicators_saved = 0
            events_saved = 0
            
            # 1. Save Technical Indicators (exclude columns ending with '_norm')
            logging.debug("Processing technical indicators...")
            for col, value in row_data.items():
                # Skip columns ending with '_norm' and the 'events' column
                if col.endswith('_norm') or col == 'events':
                    logging.debug(f"Skipping column {col} (ends with '_norm' or is 'events')")
                    continue
                
                # Ensure the value is not None and can be converted to a float
                if value is not None:
                    logging.debug(f"Processing column {col} with value {value} (type: {type(value)})")
                    try:
                        indicator_value = float(value)  # Convert to float for REAL type in SQLite
                        indicator_name = col

                        # Log the data being inserted
                        logging.debug(f"Inserting technical indicator: trade_ticket={trade_ticket}, "
                                     f"indicator_name={indicator_name}, indicator_value={indicator_value}")

                        # Insert into technical_indicators table
                        cursor.execute(
                            """
                            INSERT INTO technical_indicators (trade_ticket, indicator_name, indicator_value)
                            VALUES (?, ?, ?)
                            """,
                            (trade_ticket, indicator_name, indicator_value)
                        )
                        indicators_saved += 1
                        logging.debug(f"Successfully saved technical indicator for column {col}")
                    except (ValueError, TypeError) as e:
                        # Skip if the value cannot be converted to float (e.g., 'time', 'symbol')
                        logging.debug(f"Skipping column {col}: Cannot convert value {value} to float. Error: {e}")
                        continue
                else:
                    logging.debug(f"Skipping column {col}: Value is None")

            # 2. Save Economic Calendar Data (from the 'events' column)
            logging.debug("Processing economic calendar data...")
            if 'events' in row_data and isinstance(row_data['events'], list):
                logging.debug(f"Found {len(row_data['events'])} events in row_data['events']")
                for event_data in row_data['events']:
                    # Log the event data being processed
                    logging.debug(f"Processing event data: {event_data}")

                    # Extract required fields, ignoring '_norm' fields
                    event = event_data.get('event')
                    event_currency = event_data.get('currency')
                    actual = event_data.get('actual')
                    forecast = event_data.get('forecast')
                    previous = event_data.get('previous')
                    impact = event_data.get('impact_code')  # Use impact_code for the impact value

                    # Log the extracted fields
                    logging.debug(f"Extracted event fields: event={event}, event_currency={event_currency}, "
                                 f"actual={actual}, forecast={forecast}, previous={previous}, impact={impact}")

                    # Ensure required fields are present (event, event_currency, impact are NOT NULL)
                    if event and event_currency and impact is not None:
                        try:
                            # Convert values to appropriate types
                            actual = float(actual) if actual is not None else None
                            forecast = float(forecast) if forecast is not None else None
                            previous = float(previous) if previous is not None else None
                            impact = int(impact)  # Ensure impact is an integer (0, 1, 2)

                            # Log the data being inserted
                            logging.debug(f"Inserting economic event: trade_ticket={trade_ticket}, event={event}, "
                                         f"event_currency={event_currency}, actual={actual}, forecast={forecast}, "
                                         f"previous={previous}, impact={impact}")

                            # Insert into economic_calendar table
                            cursor.execute(
                                """
                                INSERT INTO economic_calendar (trade_ticket, event, event_currency, actual, forecast, previous, impact)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (trade_ticket, event, event_currency, actual, forecast, previous, impact)
                            )
                            events_saved += 1
                            logging.debug(f"Successfully saved economic event: {event}")
                        except (ValueError, TypeError) as e:
                            logging.info(f"Error processing economic event {event}: {e}")
                            continue
                    else:
                        logging.debug(f"Skipping event: Missing required fields (event={event}, "
                                     f"event_currency={event_currency}, impact={impact})")

            # Log the total counts before committing
            logging.info(f"Saved for trade {trade_ticket}: {indicators_saved} technical indicators, {events_saved} economic events")

            # Commit the transaction
            logging.debug("Committing transaction to database")
            conn.commit()
            logging.debug("Transaction committed successfully")

    except sqlite3.OperationalError as e:
        logging.warning(f"Database error in save_trade_indicators_and_events: {str(e)}")
        raise
    except Exception as e:
        logging.warning(f"Unexpected error in save_trade_indicators_and_events: {str(e)}")
        raise