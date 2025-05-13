import patch  # Import the patching module first

from dotenv import load_dotenv
import os
import pandas_ta as ta
import pickle
import re
import investpy
from jb_news import CJBNews
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import datetime
import time
import logging
import pytz
from json import JSONDecodeError
import sqlite3
from stable_baselines3.common.vec_env import DummyVecEnv
import math
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import threading
print("Threading module patched deploy.py:", "GreenThread" in dir(threading))

from database import save_trade_indicators_and_events, save_trade_to_db, update_closed_trade_in_db
from policy import CustomMultiInputPolicy
from real_time_env import RealTimeTgym
from util.read_config import EnvConfig

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")


# Load environment variables from .env file
load_dotenv()

# Get environment variables
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_FILE = os.path.join(PROJECT_ROOT, os.getenv("DB_FILE"))
CONFIG_FOLDER = os.path.join(PROJECT_ROOT, os.getenv("CONFIG_FOLDER"))
MODAL = os.path.join(PROJECT_ROOT, os.getenv("MODAL"))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY or len(ENCRYPTION_KEY.encode()) != 32:
    raise ValueError("ENCRYPTION_KEY must be a 32-byte string in .env")

def decrypt_password(encrypted_password):
    try:
        # Convert hex string to bytes
        encrypted_bytes = bytes.fromhex(encrypted_password)
        
        # Extract IV (first 16 bytes) and encrypted data (rest)
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create AES cipher with key and IV
        key = ENCRYPTION_KEY.encode('utf-8')  # Convert key to bytes
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Decrypt and remove padding
        padded_plaintext = cipher.decrypt(ciphertext)
        plaintext = unpad(padded_plaintext, AES.block_size).decode('utf-8')
        
        return plaintext
    except Exception as e:
        logging.error(f"Decryption error: {str(e)}")
        raise ValueError(f"Failed to decrypt password: {str(e)}")

def economic_calendars(api_key, offset=3, time_zone="GMT", news_source="forex-factory", start_time=None, end_time=None):
    """
    Combines economic calendar data from jb_news and investpy into a single DataFrame.
    
    Parameters:
    - api_key (str): API key for jb_news.
    - offset (int): Timezone offset for jb_news (e.g., 3 for GMT-3).
    - time_zone (str): Timezone for investpy data (default: "GMT").
    - start_time (datetime): Start of the time range for calendar data.
    - end_time (datetime): End of the time range for calendar data.
    - news_source (str): Source for jb_news data (default: "forex-factory").
    
    Returns:
    - pd.DataFrame: Combined economic calendar with non-NaN actual values.
    """
    # Ensure inputs are correct types
    assert isinstance(offset, int), f"Offset must be an integer, got {type(offset)}"
    assert isinstance(time_zone, str), f"Time zone must be a string, got {type(time_zone)}"
    
    # Ensure start_time and end_time are datetime objects
    if start_time and not isinstance(start_time, datetime.datetime):
        start_time = pd.to_datetime(start_time)
    if end_time and not isinstance(end_time, datetime.datetime):
        end_time = pd.to_datetime(end_time)

    # Set default to_date to current date if not provided or if equal to from_date
    current_date = datetime.datetime.now()
    from_date = start_time.strftime('%d/%m/%Y') if start_time else current_date.strftime('%d/%m/%Y')
    to_date = end_time.strftime('%d/%m/%Y') if end_time else current_date.strftime('%d/%m/%Y')
    
    # Ensure to_date is greater than from_date
    from_dt = datetime.datetime.strptime(from_date, '%d/%m/%Y')
    to_dt = datetime.datetime.strptime(to_date, '%d/%m/%Y')
    if to_dt <= from_dt:
        to_dt = from_dt + datetime.timedelta(days=1)  # Add one day
        to_date = to_dt.strftime('%d/%m/%Y')

    # --- Helper Functions ---
    # Remove parenthetical month suffixes like (Jan), (Feb), etc.
    def clean_event_name(event):
        event = re.sub(r"\s*\([^)]+\)$", "", event).strip()  # Remove month suffixes
        # Replace (YoY) with y/y and (MoM) with m/m
        event = re.sub(r"\(YoY\)", "y/y", event)
        event = re.sub(r"\(MoM\)", "m/m", event)
        return event.strip()
    
    # # Convert 'actual' to float, handling None and strings (e.g., "0.9%", "B", "M", "K")
    # def parse_actual(val):
    #     if pd.isna(val) or val is None:
    #         return float("nan")
    #     if isinstance(val, str):
    #         val = val.strip()
    #         if "%" in val:
    #             return float(val.strip("%")) / 100  # Convert percentage to decimal
    #         if val.endswith("B"):
    #             return float(val[:-1]) * 1e9  # Billion
    #         if val.endswith("M"):
    #             return float(val[:-1]) * 1e6  # Million
    #         if val.endswith("K"):
    #             return float(val[:-1]) * 1e3  # Thousand
    #     try:
    #         return float(val)
    #     except ValueError:
    #         return float("nan")  # Fallback for unparseable strings

    # Remove %, B, M, K without conversion
    def parse_actual(val):
        if pd.isna(val) or val is None:
            return float("nan")
        if isinstance(val, str):
            val = val.strip()
            # Remove %, B, M, K from the string
            val = val.replace("%", "").replace("B", "").replace("M", "").replace("K", "")
            try:
                return float(val)
            except ValueError:
                return float("nan")  # Fallback for unparseable strings
        try:
            return float(val)
        except ValueError:
            return float("nan")  # Fallback for unparseable strings

    # --- Fetch Data from jb_news ---
    jb = CJBNews()
    # api_key = "rO0Yq2kk.pUKyAHDrEdbQRlya6xgFk144MgDaaeUO"
    jb.offset = offset  # GMT-3 = 0, GMT = 3, EST = 7, PST = 10
    jb_data = []
    impact = 'Medium Impact Expected' # default impact is medium
    if jb.calendar(api_key, today=False, news_source=news_source):
        jb_data = [
            {
                "event": event.name,
                "currency": event.currency,
                "date": event.date,  # In GMT
                "actual": event.actual,
                "forecast": event.forecast,
                "previous": event.previous,
                "impact": impact,
                "source": "forex factory"
            }
            for event in jb.calendar_info
        ]
    else:
        print("Failed to fetch forex factory data, proceeding with investpy only")
        # Sample Testing Data (unchanged from your original)
        # jb_data = [
        #     {"event": "Consumer Confidence", "currency": "EUR", "date": "2025-03-21 15:00:00", "actual": 0.0, "forecast": -13.0, "previous": -14.0, "impact": impact, "source": "forex factory"},
        #     {"event": "FOMC Member Williams Speaks", "currency": "USD", "date": "2025-03-21 13:05:00", "actual": 0.0, "forecast": 0.0, "previous": 0.0, "impact": impact, "source": "forex factory"},
        #     {"event": "Core Retail Sales m/m", "currency": "CAD", "date": "2025-03-21 12:30:00", "actual": 0.0, "forecast": -0.1, "previous": 2.7, "impact": impact, "source": "forex factory"},
        #     {"event": "NHPI m/m", "currency": "CAD", "date": "2025-03-21 12:30:00", "actual": 0.0, "forecast": 0.0, "previous": -0.1, "impact": impact, "source": "forex factory"},
        #     {"event": "Retail Sales m/m", "currency": "CAD", "date": "2025-03-21 12:30:00", "actual": 0.0, "forecast": -0.4, "previous": 2.5, "impact": impact, "source": "forex factory"},
        #     {"event": "CBI Industrial Order Expectations", "currency": "GBP", "date": "2025-03-21 11:00:00", "actual": 0.0, "forecast": -30.0, "previous": -28.0, "impact": impact, "source": "forex factory"},
        #     {"event": "Current Account", "currency": "EUR", "date": "2025-03-21 09:00:00", "actual": 0.0, "forecast": 0.0, "previous": 38.4, "impact": impact, "source": "forex factory"},
        #     {"event": "Public Sector Net Borrowing", "currency": "GBP", "date": "2025-03-21 07:00:00", "actual": 0.0, "forecast": 7.0, "previous": -15.4, "impact": impact, "source": "forex factory"},
        #     {"event": "Credit Card Spending y/y", "currency": "NZD", "date": "2025-03-21 02:00:00", "actual": 0.9, "forecast": 0.0, "previous": 1.3, "impact": impact, "source": "forex factory"},
        #     {"event": "GfK Consumer Confidence", "currency": "GBP", "date": "2025-03-21 00:01:00", "actual": -19.0, "forecast": -20.0, "previous": -20.0, "impact": impact, "source": "forex factory"},
        #     {"event": "National Core CPI y/y", "currency": "JPY", "date": "2025-03-20 23:30:00", "actual": 3.0, "forecast": 2.9, "previous": 3.2, "impact": impact, "source": "forex factory"},
        #     {"event": "Trade Balance", "currency": "NZD", "date": "2025-03-20 21:45:00", "actual": 510.0, "forecast": -235.0, "previous": -544.0, "impact": impact, "source": "forex factory"}
        # ]

    # Convert to DataFrame
    jb_df = pd.DataFrame(jb_data)
    if not jb_df.empty:
        jb_df["date"] = pd.to_datetime(jb_df["date"])

    # --- Fetch Data from investpy with Error Handling ---
    try:
        investpy_df = investpy.news.economic_calendar(
            time_zone=time_zone,
            time_filter="time_only",
            from_date=from_date,
            to_date=to_date
        )
    except JSONDecodeError as e:
        logging.info(f"Failed to fetch investpy data due to JSONDecodeError: {e}")
        investpy_df = pd.DataFrame()  # Return empty DataFrame on failure
    except Exception as e:
        logging.info(f"Failed to fetch investpy data due to unexpected error: {e}")
        investpy_df = pd.DataFrame()  # Return empty DataFrame on failure

    if not investpy_df.empty:
        investpy_df = investpy_df.rename(columns={
            "event": "event",
            "currency": "currency",
            "date": "date",
            "time": "time",
            "actual": "actual",
            "forecast": "forecast",
            "previous": "previous",
            "importance": "impact"
        }).assign(source="investing.com")
        # Ensure date is datetime, handle 'All Day' or invalid times
        investpy_df["date"] = pd.to_datetime(
            investpy_df["date"] + " " + investpy_df["time"].replace("All Day", "00:00"), 
            errors="coerce", 
            format="%d/%m/%Y %H:%M"
        )

        # Update impact column values
        investpy_df["impact"] = investpy_df["impact"].replace({
            "low": "Low Impact Expected",
            "medium": "Medium Impact Expected",
            "high": "High Impact Expected"
        })

        # Clean event names
        investpy_df["event"] = investpy_df["event"].apply(clean_event_name)

    # --- Normalize Columns ---
    # Select relevant columns and ensure consistent types
    columns = ["event", "currency", "date", "impact", "actual", "forecast", "previous", "source"]
    if not jb_df.empty:
        jb_df = jb_df[columns]
        jb_df["actual"] = jb_df["actual"].apply(parse_actual)
        jb_df["forecast"] = jb_df["forecast"].apply(parse_actual)
        jb_df["previous"] = jb_df["previous"].apply(parse_actual)
    if not investpy_df.empty:
        investpy_df = investpy_df[columns]
        investpy_df["actual"] = investpy_df["actual"].apply(parse_actual)
        investpy_df["forecast"] = investpy_df["forecast"].apply(parse_actual)
        investpy_df["previous"] = investpy_df["previous"].apply(parse_actual)

    # --- Combine Data ---
    combined_df = pd.DataFrame()
    if jb_df.empty and investpy_df.empty:
        print("Both data sources failed, returning empty DataFrame")
        return combined_df
    elif jb_df.empty:
        combined_df = investpy_df.copy()
    elif investpy_df.empty:
        combined_df = jb_df.copy()
    else:
        # Step 1: Identify matching events by event name and currency
        jb_df["key"] = jb_df["event"].str.lower() + "_" + jb_df["currency"].str.lower()
        investpy_df["key"] = investpy_df["event"].str.lower() + "_" + investpy_df["currency"].str.lower()

        # Step 2: Sort both DataFrames by date (ascending, so oldest first; adjust to descending if needed)
        jb_df = jb_df.sort_values("date")
        investpy_df = investpy_df.sort_values("date")

        # Step 3: Filter based on actual values
        combined_df = pd.DataFrame()
        for key in set(jb_df["key"]).union(investpy_df["key"]):
            jb_match = jb_df[jb_df["key"] == key]
            investpy_match = investpy_df[investpy_df["key"] == key]

            if not jb_match.empty and investpy_match.empty:
                # Only in jb_news
                combined_df = pd.concat([combined_df, jb_match])
            elif jb_match.empty and not investpy_match.empty:
                # Only in investpy
                combined_df = pd.concat([combined_df, investpy_match])
            elif not jb_match.empty and not investpy_match.empty:
                # Both have the event
                jb_actual = jb_match["actual"].iloc[0]
                investpy_actual = investpy_match["actual"].iloc[0]
                
                if pd.isna(jb_actual) and not pd.isna(investpy_actual):
                    # Use investpy if jb_news actual is NaN
                    combined_df = pd.concat([combined_df, investpy_match])
                elif not pd.isna(jb_actual) and pd.isna(investpy_actual):
                    # Use jb_news if investpy actual is NaN
                    combined_df = pd.concat([combined_df, jb_match])
                elif not pd.isna(jb_actual) and not pd.isna(investpy_actual):
                    # Both non-NaN, use investpy
                    combined_df = pd.concat([combined_df, investpy_match])
                else:
                    # Both NaN, use investpy (arbitrary choice)
                    combined_df = pd.concat([combined_df, investpy_match])

    # Remove rows where actual is NaN
    combined_df = combined_df.dropna(subset=["actual"])

    # --- Clean Up ---
    # Ensure date is datetime before strftime
    combined_df.loc[:, "date"] = pd.to_datetime(combined_df["date"], errors="coerce")
    combined_df = combined_df.drop(columns=["key"] if "key" in combined_df.columns else []).reset_index(drop=True)
    combined_df.loc[:, "date"] = combined_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return combined_df

def clean_calendar_data(df, impact_map=None, event_map=None, currency_map=None, event_freq=None):
    """
    Clean and preprocess economic calendar data by standardizing columns, converting dates, mapping impact levels,
    normalizing numeric features, and aggregating events into hourly intervals.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing raw economic calendar data with columns such as 'event',
                         'currency', 'impact', 'actual', 'forecast', 'previous', and 'date'.
    - impact_map (dict, optional): Dictionary mapping impact strings to integer codes. 
                                   Default is {'High Impact Expected': 2, 'Medium Impact Expected': 1, 
                                   'Low Impact Expected': 0}.
    - event_map (dict, optional): Dictionary mapping event names to IDs. If provided, rows with events not in this 
                                  map are filtered out. Default is None (no filtering).
    - currency_map (dict, optional): Dictionary mapping currency codes to IDs. If provided, rows with currencies not 
                                     in this map are filtered out. Default is None (no filtering).
    - event_freq (dict, optional): Dictionary mapping event names to their frequency counts. If provided, an 
                                   'event_freq' column is added with these values; otherwise, it defaults to 0. 
                                   Default is None.

    Returns:
    - pd.DataFrame: A cleaned and aggregated DataFrame with one row per hourly interval, containing a 'time' column 
                    (renamed from 'interval') and an 'events' column with a list of dictionaries. Each dictionary 
                    includes 'event', 'currency', 'impact_code', 'actual_norm', 'forecast_norm', 'previous_norm', 
                    'surprise_norm', and 'event_freq' for events within that hour.
    """
    # Default impact mapping if none provided
    if impact_map is None:
        impact_map = {'High Impact Expected': 2, 'Medium Impact Expected': 1, 'Low Impact Expected': 0}

    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    logging.debug(f"Input columns: {list(cleaned_df.columns)}")

    # Check required columns
    required_cols = ['event', 'currency', 'date', 'actual']
    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(columns=['time', 'events'])

    # Filter rows based on event_map and currency_map (if provided)
    if event_map is not None:
        valid_events = list(event_map.keys())  # Extract valid event names
        cleaned_df = cleaned_df[cleaned_df['event'].isin(valid_events)]
        if cleaned_df.empty:
            logging.warning("No events match event_map, returning empty aggregation")
            return pd.DataFrame(columns=['time', 'events'])
    if currency_map is not None:
        valid_currencies = list(currency_map.keys())  # Extract valid currency codes
        cleaned_df = cleaned_df[cleaned_df['currency'].isin(valid_currencies)]
        if cleaned_df.empty:
            logging.warning("No currencies match currency_map, returning empty aggregation")
            return pd.DataFrame(columns=['time', 'events'])

    # Map impact to integers (if impact column exists)
    if "impact" in cleaned_df.columns:
        cleaned_df["impact_code"] = cleaned_df["impact"].map(impact_map).fillna(0)  # Default to 0 if unmapped
    else:
        cleaned_df["impact_code"] = 0
        logging.debug("No 'impact' column, defaulting impact_code to 0")

    # Drop rows where 'actual' is NaN
    cleaned_df = cleaned_df.dropna(subset=["actual"])
    if cleaned_df.empty:
        logging.warning("No rows with non-NaN 'actual', returning empty aggregation")
        return pd.DataFrame(columns=['time', 'events'])

    # Convert actual, forecast, and previous to numeric
    for col in ['actual', 'forecast', 'previous']:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        else:
            cleaned_df[col] = 0  # Default if missing
            logging.debug(f"No '{col}' column, defaulting to 0")

    # Handle Missing Forecast and Surprise
    cleaned_df['previous'] = cleaned_df['previous'].fillna(cleaned_df['actual'])   # Use actual if previous is NaN
    cleaned_df['forecast'] = cleaned_df['forecast'].fillna(cleaned_df['previous'])  # Use previous as forecast if missing
    cleaned_df['forecast'] = cleaned_df['forecast'].fillna(cleaned_df['actual'])   # Use actual if previous is also NaN
    cleaned_df['surprise'] = (cleaned_df['actual'] - cleaned_df['forecast']).round(2)  # Recalculate surprise
    cleaned_df['surprise'] = cleaned_df['surprise'].fillna(0)  # Default to 0 if still missing

    # Normalize Numeric Features (z-score standardization)
    for col in ['actual', 'forecast', 'previous', 'surprise']:
        mean = cleaned_df[col].mean()
        std = cleaned_df[col].std() or 1  # Avoid division by zero
        cleaned_df[f'{col}_norm'] = (cleaned_df[col] - mean) / std

    # Add event_freq column
    if event_freq is not None:
        cleaned_df['event_freq'] = cleaned_df['event'].map(event_freq).fillna(0)  # Map frequencies, default to 0 if not found
    else:
        cleaned_df['event_freq'] = 0

    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
    if cleaned_df['date'].isna().all():
        logging.error("All 'date' values are NaN, returning empty aggregation")
        return pd.DataFrame(columns=['time', 'events'])

    # Event Window Expansion and Aggregation
    cleaned_df['interval'] = cleaned_df['date'].dt.ceil('1h')
    event_agg = cleaned_df.groupby('interval').apply(
        lambda x: x[['event', 'currency', 'impact', 'actual', 'forecast', 'previous',
                     'impact_code', 'actual_norm', 'forecast_norm',
                     'previous_norm', 'surprise_norm', 'event_freq']].to_dict('records')
    ).reset_index(name='events')
    event_agg.rename(columns={'interval': 'time'}, inplace=True)

    return event_agg

def add_technical_indicators(df):
    # Calculate technical indicators
    # 1. Relative Strength Index (RSI)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # 2. Moving Average Convergence Divergence (MACD)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    macd.columns = ['MACD', 'MACD_Histogram', 'MACD_Signal']
    df = df.join(macd)

    # 3. Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = df.join(bbands)

    # 4. Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df = df.join(stoch)

    # 5. Average Directional Index (ADX)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

    # 6. Commodity Channel Index (CCI)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

    # 7. Parabolic Stop and Reverse (Parabolic SAR)
    df['PSAR'] = ta.psar(df['High'], df['Low'], df['Close'], af=0.02, max_af=0.2)['PSARl_0.02_0.2']

    # 8. Simple Moving Average (SMA)
    df['SMA'] = ta.sma(df['Close'], length=50)

    # 9. Fibonacci Retracement Levels
    # Calculate the high and low over a specified period
    period = 20  # You can adjust this period as needed
    df['High_Max'] = df['High'].rolling(window=period).max()
    df['Low_Min'] = df['Low'].rolling(window=period).min()
    # Calculate Fibonacci levels
    df['Fib_23.6'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.236
    df['Fib_38.2'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.382
    df['Fib_50.0'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.500
    df['Fib_61.8'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.618
    df['Fib_100.0'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 1.000
    # Drop intermediate columns
    df.drop(columns=['High_Max', 'Low_Min'], inplace=True)

    # 10. ATR (Average True Range)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Display the DataFrame with the new indicators
    # print(df.tail())

    return df

# MT5 Initialization
def initialize_mt5(account, password, server):
    mt5.shutdown()  # Reset state
    
    # establish connection to the MetaTrader 5 terminal
    if not mt5.initialize(login=account, password=password, server=server):
        print("MT5 initialize failed, error code =",mt5.last_error())
        return False
    print("MT5 initialized")
    
    if not mt5.login(account, password=password, server=server):
        print(f"Failed to login to MT5 with account {account}")
        return False
    print(f"Logged into MT5 with account {account}")
    return True

def get_mt5_account_info(account, password, server):
    initialize_mt5(account=account, password=password, server=server)
    
    # Ensure MT5 is initialized and logged in
    if not mt5.terminal_info():
        print("MT5 not initialized, cannot fetch account info")
        return False
    
    account_info = mt5.account_info()
    if account_info is not None:
        # Convert account_info to a dictionary for JSON serialization
        account_info_dict = {
            "login": account_info.login,
            "trade_mode": account_info.trade_mode,
            "leverage": account_info.leverage,
            "limit_orders": account_info.limit_orders,
            "margin_so_mode": account_info.margin_so_mode,
            "trade_allowed": account_info.trade_allowed,
            "trade_expert": account_info.trade_expert,
            "margin_mode": account_info.margin_mode,
            "currency_digits": account_info.currency_digits,
            "fifo_close": account_info.fifo_close,
            "balance": account_info.balance,
            "credit": account_info.credit,
            "profit": account_info.profit,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "margin_level": account_info.margin_level,
            "margin_so_call": account_info.margin_so_call,
            "margin_so_so": account_info.margin_so_so,
            "margin_initial": account_info.margin_initial,
            "margin_maintenance": account_info.margin_maintenance,
            "assets": account_info.assets,
            "liabilities": account_info.liabilities,
            "commission_blocked": account_info.commission_blocked,
            "name": account_info.name,
            "server": account_info.server,
            "currency": account_info.currency,
            "company": account_info.company,
        }
        return account_info_dict
    else:
        print("Failed to get account info")
        return False

# Fetch live OHLC data from MT5
def fetch_ohlc_mt5(symbol="XAUUSD.sml", timeframe=mt5.TIMEFRAME_H1, bars=500, retries=3):
    """
    bars+50 to handle NaN values in technical indicators
    """
    for _ in range(retries):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars+50) 
        if rates is None:
            print(f"Failed to fetch rates for {symbol}")
            return None
        else:
            break
    ohlc_df = pd.DataFrame(rates)
    ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], unit='s')
    ohlc_df = ohlc_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
    })
    ohlc_df['symbol'] = symbol
    ohlc_df['weekday'] = ohlc_df['time'].dt.weekday
    ohlc_df['hour'] = ohlc_df['time'].dt.hour
    ohlc_df['hour_sin'] = np.sin(2 * np.pi * ohlc_df['hour'] / 24)  # Sine encoding
    ohlc_df['hour_cos'] = np.cos(2 * np.pi * ohlc_df['hour'] / 24)  # Cosine encoding
    return ohlc_df[['time', 'symbol', 'Open', 'High', 'Low', 'Close', 'weekday', 'hour_sin', 'hour_cos']]

# Process and normalize OHLC data
def process_ohlc(ohlc_df, window_size=500):
    # Calculate Technical Indicators
    ohlc_df = add_technical_indicators(ohlc_df)
    ohlc_df.fillna(method='ffill', inplace=True) # Forward fill to handle NaN values
    ohlc_df.dropna(inplace=True)

    # Columns to normalize
    columns_to_normalize = [
        'Open', 'High', 'Low', 'Close',
        'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',  # Momentum indicators
        'STOCHk_14_3_3', 'STOCHd_14_3_3',  # Stochastic oscillators
        'ADX', 'CCI', 'SMA',  # Trend indicators
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',  # Bollinger Bands
        'Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_100.0',  # Fibonacci levels
        'ATR',  # Measure Market Volatility
    ]
    for col in columns_to_normalize:
        # Calculate rolling mean and standard deviation
        rolling_mean = ohlc_df[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = ohlc_df[col].rolling(window=window_size, min_periods=1).std()

        # Create a new column for the normalized value: {col}_norm
        norm_col = f"{col}_norm"
        ohlc_df[norm_col] = (ohlc_df[col] - rolling_mean) / rolling_std.fillna(1e-6)

    # Fill NaN values in the normalized columns with 0
    ohlc_df.fillna(0, inplace=True)
    return ohlc_df

# Load calendar data and mappings
def fetch_live_calendar(api_key, offset=3, time_zone="GMT", news_source="forex-factory", start_time=None, end_time=None):
    logging.debug(f"config_folder: {CONFIG_FOLDER}")
    with open(f"{CONFIG_FOLDER}event_map.pkl", "rb") as f:
        event_map = pickle.load(f)
    with open(f"{CONFIG_FOLDER}currency_map.pkl", "rb") as f:
        currency_map = pickle.load(f)
    with open(f"{CONFIG_FOLDER}event_freq.pkl", "rb") as f:
        event_freq = pickle.load(f)

    # logging.info(f"Event Map: {event_map}")
    # logging.info(f"Currency Map: {currency_map}")

    calendar_data = economic_calendars(
        api_key, offset=offset, time_zone=time_zone, news_source=news_source, start_time=start_time, end_time=end_time
    )
    logging.debug(f"Calendar data columns: {list(calendar_data.columns)}")
    logging.debug(f"Calendar data sample: {calendar_data.head().to_dict()}")

    if calendar_data.empty:
        logging.warning("Calendar data is empty, returning empty aggregation")
        event_agg = pd.DataFrame(columns=['time', 'events'])
    else:
        try:
            event_agg = clean_calendar_data(
                calendar_data, 
                event_map=event_map, 
                currency_map=currency_map, 
                event_freq=event_freq
            )
        except Exception as e:
            logging.error(f"Exception in clean_calendar_data: {str(e)}", exc_info=True)
            event_agg = pd.DataFrame(columns=['time', 'events'])  # Fallback to empty DataFrame
    
    event_agg['time'] = pd.to_datetime(event_agg['time'])
    # print("Calendar Aggeregation: \n", event_agg.to_string())

    return event_agg, event_map, currency_map, event_freq

# Combine OHLC with calendar data
def combine_data(ohlc_df, calendar_agg):
    df = pd.merge(ohlc_df, calendar_agg, on='time', how='left').fillna({'events': '[]'})
    return df

import MetaTrader5 as mt5

def check_trend(symbol, bars=10, timeframe=mt5.TIMEFRAME_H1):
    """
    Check the trend of the previous 'bars' bars for the given symbol.
    
    Args:
        symbol (str): Trading symbol (e.g., "XAUUSD").
        bars (int): Number of previous bars to analyze (default: 10).
        timeframe (int): MT5 timeframe (default: TIMEFRAME_H1, 1-hour bars).
    
    Returns:
        float: Percentage of bars showing an upward trend (close > open).
    """
    # Ensure MT5 is initialized
    if not mt5.initialize():
        print("MT5 not initialized for trend check")
        return None

    # Fetch the last 'bars' bars of data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < bars:
        print(f"Failed to fetch {bars} bars for {symbol}: {mt5.last_error()}")
        return None

    # Count upward bars (close > open)
    upward_bars = sum(1 for rate in rates if rate['close'] > rate['open'])
    total_bars = len(rates)
    
    # Calculate percentage of upward bars
    upward_percentage = (upward_bars / total_bars) * 100
    return upward_percentage

# Execute trade via MT5
def execute_trade(accountId, symbol, action, lot_size=0.1, config_file="./neo_finrl/env_fx_trading/config/gdbusd-test-1.json"):
    # Ensure MT5 is initialized
    if not mt5.initialize():
        print("MT5 not initialized")
        return None

    # Ensure symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol} in Market Watch")
        return None

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to retrieve symbol info for {symbol}")
        return None

    # Get tick data
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to retrieve tick data for {symbol}")
        return None
    
    # print(f"Symbol info: {symbol_info}")
    # print(f"Tick data: {tick}")

    if not is_market_open(symbol):
        print(f"Market is closed for {symbol}")
        return None
    
    # Load configuration (same as used in tgym)
    cf = EnvConfig(config_file)

    lot_size = cf.symbol(symbol, "volume") 
    
    # Interpret action (float between 0 and 3)
    action_type = math.floor(action)  # 0=Buy, 1=Sell, 2=Nothing
    if action_type not in (0, 1):  # Do nothing for action >= 2
        logging.info(f"No trade executed for {symbol}: action={action} (action_type={action_type}, Nothing)")
        return None

    # Check trend for the previous 10 bars
    trend_percentage = check_trend(symbol, bars=10, timeframe=mt5.TIMEFRAME_H1)
    if trend_percentage is None:
        print(f"Skipping trade due to trend check failure for {symbol}")
        return None

    # Adjust action_type based on trend
    if action_type == 0:  # Intended Buy
        if trend_percentage >= 70:  # 30% or more bars are upward
            final_action_type = 0  # Confirm Buy
            logging.info(f"Predicted action: Buy")
        else:
            final_action_type = 1  # Switch to Sell
            logging.info(f"Predicted action: Sell")
    elif action_type == 1:  # Intended Sell
        if trend_percentage <= 30:  # 70% or more bars are downward (100% - 70% = 30% upward)
            final_action_type = 1  # Confirm Sell
            logging.info(f"Predicted action: Sell")
        else:
            final_action_type = 0  # Switch to Buy
            logging.info(f"Predicted action: Buy")

    # Get price and point value
    price = tick.ask if final_action_type == 0 else tick.bid  # Buy at ask, Sell at bid
    point = symbol_info.point

    # Get SL and TP parameters from config (same as tgym)
    stop_loss_max = cf.symbol(symbol, "stop_loss_max")  # In points
    profit_taken_max = cf.symbol(symbol, "profit_taken_max")  # In points
    
    # Calculate SL and TP in points (same as tgym)
    sl_points = stop_loss_max
    tp_points = math.ceil((action - action_type) * profit_taken_max)

    # Convert to price levels for MT5
    min_stop_distance = symbol_info.trade_stops_level * point  # Broker's minimum stop distance
    sl_distance = max(sl_points * point, min_stop_distance)  # Ensure at least min_stop_distance
    tp_distance = max(tp_points * point, min_stop_distance)  # Ensure at least min_stop_distance

    # Define trade request based on final_action_type
    if final_action_type == 0:  # Buy
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price - sl_distance,  # SL below entry price
            "tp": price + tp_distance,  # TP above entry price
            "deviation": 10,
            "magic": 123456,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
    elif final_action_type == 1:  # Sell
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price + sl_distance,  # SL above entry price
            "tp": price - tp_distance,  # TP below entry price
            "deviation": 10,
            "magic": 123456,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

    print(f"Price: {request['price']}, SL: {request['sl']}, TP: {request['tp']}, Min Distance: {min_stop_distance}")
    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed for {symbol}: {mt5.last_error()}")
        return None
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed for {symbol}: {result.comment} (Retcode: {result.retcode})")
        return None
    else:
        print(f"Trade executed: {symbol}, Action={final_action_type}, Price={price}")
        # Retrieve the deal details using the deal ticket
        deal = mt5.history_deals_get(ticket=result.deal)
        # print("deal: ", deal)
        if deal and len(deal) > 0:
            # Get the deal's execution time as a Unix timestamp
            deal_time = deal[0].time
            # Convert to UTC timezone-aware datetime and format as string
            open_time_utc = datetime.datetime.fromtimestamp(deal_time, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Fallback in case deal details aren’t available
            print("Warning: Could not retrieve deal details. Using current UTC time as fallback.")
            open_time_utc = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # Populate trade_data with the MT5 open time
        trade_data = {
            "ticket": result.order,
            "symbol": symbol,
            "action": final_action_type,
            "volume": result.volume,
            "open_price": result.price,
            "sl": result.request.sl,
            "tp": result.request.tp,
            "open_time": open_time_utc
        }
        save_trade_to_db(accountId, trade_data)
        return {"success": True, "ticket": result, "price": price}

def is_market_open(symbol, timezone="Etc/GMT-3", tick_timeout=300):
    """
    Check if the market for a specific symbol is open based on MT5 data.

    Parameters:
    - symbol (str): The symbol to check (e.g., "XAUUSD").
    - timezone (str): The broker's timezone (e.g., "Etc/GMT-3" for OANDA GMT+3).
    - tick_timeout (int): Maximum seconds since last tick to consider market open (default: 300s = 5min).

    Returns:
    - bool: True if the market is open, False if closed or unknown.
    """
    # Ensure MT5 is initialized
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return False

    # Ensure symbol is available in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol} in Market Watch")
        return False

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found or unavailable")
        return False

    # Check if trading is allowed
    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print(f"Trading disabled for {symbol}")
        return False
    elif symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        print(f"Only closing positions allowed for {symbol}")
        return False

    # Get latest tick to infer server time and market activity
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"No tick data for {symbol}")
        return False

    # Current time in broker's timezone (assumed from tick time)
    tick_time = tick.time  # Server time of last tick in seconds since epoch
    current_time = datetime.datetime.now(pytz.timezone(timezone)).timestamp()  # Local time adjusted to GMT+3
    time_diff = current_time - tick_time

    # If no recent ticks (e.g., > 5 minutes), assume market is closed
    if time_diff > tick_timeout:
        print(f"No recent ticks for {symbol} (last tick: {datetime.datetime.fromtimestamp(tick_time, tz=pytz.timezone(timezone))}, diff: {time_diff:.0f}s)")
        return False

    # Market is open if ticks are recent and trading mode allows
    print(f"Market open check for {symbol}: Last tick at {datetime.datetime.fromtimestamp(tick_time, tz=pytz.timezone(timezone))}, diff: {time_diff:.0f}s")
    return True

# Fetch MT5 data for a given symbol
def get_mt5_info(symbol="XAUUSD"):
    # Ensure symbol is available
    if not mt5.symbol_select(symbol, True):
        error_msg = f"Failed to select symbol {symbol}: {mt5.last_error()}"
        logging.error(error_msg)
        return {"error": error_msg}

    # Get symbol, account, and terminal info
    symbol_info = mt5.symbol_info(symbol)
    account_info = mt5.account_info()
    terminal_info = mt5.terminal_info()
    
    if symbol_info is None:
        error_msg = f"Failed to get symbol info for {symbol}: {mt5.last_error()}"
        logging.error(error_msg)
        return {"error": error_msg}
    if account_info is None:
        error_msg = f"Failed to get account info: {mt5.last_error()}"
        logging.error(error_msg)
        return {"error": error_msg}
    if terminal_info is None:
        error_msg = f"Failed to get terminal info: {mt5.last_error()}"
        logging.error(error_msg)
        return {"error": error_msg}

    # Extract available data
    data = {
        # Account Info
        "login": account_info.login,
        "server": account_info.server,
        "name": account_info.name,
        "company": account_info.company,
        "leverage": account_info.leverage,

        # Symbol Info
        "point": symbol_info.point,  # Pip value (e.g., 0.01 for XAUUSD)
        "volume_min": symbol_info.volume_min,
        "volume_max": symbol_info.volume_max,
        "trade_stops_level": symbol_info.trade_stops_level,
        "over_night_penalty_long": symbol_info.swap_long,  # Swap fee for long positions
        "over_night_penalty_short": symbol_info.swap_short,  # Swap fee for short positions

        # Terminal Info
        "trade_allowed": terminal_info.trade_allowed,
    }
    return data



class RealTimeTrader:
    """
    A class to manage real-time trading with MetaTrader 5, including data fetching, model prediction, and trade execution.
    
    Attributes:
        account_id (str): User account number for MT5.
        login_id (str): Login ID (if separate from account_id).
        password (str): Password for MT5 authentication.
        server (str): MT5 server name.
        symbol (str): Trading symbol (e.g., "XAUUSD").
        api_key (str): API key for fetching calendar data.
        config_file (str): Path to the configuration file.
        stop_event (Event): Threading event to stop the trading loop.
    """
    """
    1. Fetch Live OHLC Data from MetaTrader 5
    2. Real-Time Processing and Normalization
    3. Combine with Calendar Data
    4. Model Decision and Action Execution
    5. SQLite storage in a trading journal.

    config:
    1. user account number, password, server
    2. symbol (broker provided symbol), timezone (timezone used by broker) OANDA: GMT+3
    3. policy_kwargs: initial_balance, point_scale
    """
    def __init__(self, account_id, login_id, password, server, symbol="XAUUSD", stop_event=None):
        """Initialize the RealTimeTrader with user credentials and configuration."""
        self.account_id = account_id
        self.login_id = int(login_id)
        self.password = password
        self.server = server
        self.symbol = symbol
        self.api_key = "rO0Yq2kk.pUKyAHDrEdbQRlya6xgFk144MgDaaeUO"
        self.config_file = f"{CONFIG_FOLDER}user_config.json"
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.env = None
        self.model = None
        self.timeframe = mt5.TIMEFRAME_H1 
        self.time_zone = "GMT +3:00"
        self.offset = 6
        self.MT_timezone = "Etc/GMT-3"
        self.bars = 500
        self.max_retries = 5
        self.trading_thread = None
        self.monitor_thread = None
        logging.info("RealTimeTrader initialized with account: %s, server: %s", account_id, server)

    def reconnect_mt5(self, max_attempts: int = 3, delay: int = 5) -> bool:
        """
        Attempt to reconnect to MT5 if the connection is lost.

        Args:
            max_attempts (int): Maximum number of reconnection attempts.
            delay (int): Delay between attempts in seconds.

        Returns:
            bool: True if reconnection succeeds, False otherwise.
        """
        logging.info("Attempting to reconnect to MT5...")
        for attempt in range(1, max_attempts + 1):
            logging.debug(f"Reconnection attempt {attempt}/{max_attempts}")
            if initialize_mt5(self.login_id, self.password, self.server):
                logging.info("Successfully reconnected to MT5")
                return True
            logging.warning(f"Reconnection attempt {attempt} failed")
            if attempt < max_attempts:
                time.sleep(delay)
        logging.error("Failed to reconnect to MT5 after maximum attempts")
        return False

    def run(self):
        # Import emit_status here to avoid circular import
        from app import emit_status

        logging.info("Starting real_time_trading")
        try:
            print(self.account_id, self.login_id, self.password, self.server, self.symbol)
            # Initialize MT5
            try:
                if not initialize_mt5(account=self.login_id, password=self.password, server=self.server):
                    raise Exception("MT5 initialization failed")
                logging.info("MT5 initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize MT5: {str(e)}")
                self.stop_event.set()
                emit_status()  # Notify frontend
                return

            # Fetch initial 500-bar data from MT5 and sort oldest-to-newest
            ohlc_df = None
            retries = 0
            while ohlc_df is None and not self.stop_event.is_set() and retries < self.max_retries:
                try:
                    ohlc_df = fetch_ohlc_mt5(symbol=self.symbol, timeframe=self.timeframe, bars=self.bars)
                    if ohlc_df is None:
                        logging.info(f"Time: {datetime.datetime.now().time()}, Failed to fetch initial OHLC data, retrying ({retries + 1}/{self.max_retries})...")
                        time.sleep(60)
                except Exception as e:
                    logging.error(f"Error fetching initial OHLC data: {str(e)}")
                    time.sleep(60)
                retries += 1

            if ohlc_df is None:
                logging.error("Failed to fetch initial OHLC data after max retries")
                self.stop_event.set()
                emit_status()  # Notify frontend
                return
            if self.stop_event.is_set():
                logging.info("Trading stopped during initial fetch")
                return
            
            ohlc_df = process_ohlc(ohlc_df).sort_values('time')  # Ensure oldest-to-newest
            # Calculate start and end times from the OHLC data
            start_time = ohlc_df['time'].iloc[0]  # Oldest timestamp (first row)
            end_time = ohlc_df['time'].iloc[-1]   # Newest timestamp (last row)
            # Log the time range for debugging
            logging.info(f"OHLC Data Time Range: Start={start_time}, End={end_time}, Duration={end_time - start_time}")

            # Fetch calendar data
            calendar_agg = None
            calendar_agg, event_map, currency_map, event_freq = fetch_live_calendar(
                self.api_key, 
                time_zone=self.time_zone, 
                offset=self.offset,
                start_time=start_time,  # Pass start time
                end_time=end_time       # Pass end time
            )
            if calendar_agg is None:
                logging.info(f"Time: {datetime.datetime.now().time()}, Failed to fetch calendar data, retrying ({retries + 1}/{self.max_retries})...")
                time.sleep(60)  # Wait before retrying
            
            if self.stop_event.is_set():
                logging.info("Trading stopped during calendar fetch")
                return

            # Combine with calendar data
            try:
                combined_df = combine_data(ohlc_df, calendar_agg).sort_values('time')  # Ensure sorting
                logging.debug(f"Combined DataFrame shape: {combined_df.shape}")
                logging.debug(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
            except Exception as e:
                logging.error(f"Error combining OHLC and calendar data: {str(e)}")
                self.stop_event.set()
                emit_status()  # Notify frontend
                return

            # Initialize environment
            try:
                self.env = RealTimeTgym(df=combined_df, event_map=event_map, currency_map=currency_map, env_config_file=self.config_file)
                env_vec = DummyVecEnv([lambda: self.env])
            except Exception as e:
                logging.error(f"Error initializing environment: {str(e)}")
                self.stop_event.set()
                emit_status()  # Notify frontend
                return

            # Load trained PPO model
            try:
                self.model = PPO.load(
                    MODAL, 
                    env=env_vec, 
                    device="cpu", 
                    custom_objects={
                        "policy": CustomMultiInputPolicy,
                        "policy_kwargs": {
                            "initial_balance": 1000.0,
                            "point_scale": 100.0,
                            "embed_dim": 32,
                            "impact_field_idx": 5,  # 'impact_code' is the 6th field (0-based index)
                            "usd_currency_id": 4    # 'USD' is mapped to ID 4 in currency_map
                        }
                    }
                )
                logging.info("PPO model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading PPO model: {str(e)}")
                self.stop_event.set()
                emit_status()  # Notify frontend
                return

            # # Warm up the model with the 500-bar window
            # try:
            #     window_obs = self.env.get_window_obs()
            #     for i, obs in enumerate(window_obs):  # Process all rows sequentially
            #         action, _states = self.model.predict(obs, deterministic=True)
            #         self.env.current_step = i  # Set the current step explicitly
            #         self.env.step(action)  # Simulate trades for step i
            #     logging.info("Model warm-up completed")
            # except Exception as e:
            #     logging.error(f"Error during model warm-up: {str(e)}")
            #     self.stop_event.set()
            #     emit_status()  # Notify frontend
            #     return
            
            # Real-time loop
            last_time = None
            while not self.stop_event.is_set():
                logging.info(f"Loop iteration started at {datetime.datetime.now().time()}")
                # Check MT5 connection
                if not mt5.terminal_info():
                    logging.error("MT5 connection lost unexpectedly")
                    if self.reconnect_mt5():
                        logging.info("MT5 reconnected, restarting run()")
                        self.run()  # Restart the run method
                        return
                    else:
                        logging.error("Failed to reconnect to MT5, stopping trading")
                        self.stop_event.set()
                        emit_status()  # Notify frontend
                        return

                if not is_market_open(self.symbol, timezone=self.MT_timezone):
                    logging.info(f"Time: {datetime.datetime.now().time()}, Market closed for {self.symbol}, waiting 1 hour...")
                    time.sleep(3600)  # Wait longer during closures (e.g., weekends)
                    continue

                # Fetch latest OHLC data (1-bar)
                try:
                    ohlc_df = fetch_ohlc_mt5(symbol=self.symbol, timeframe=self.timeframe, bars=self.bars)
                    if ohlc_df is None:
                        logging.info(f"Time: {datetime.datetime.now().time()}, Failed to fetch OHLC data, retrying...")
                        time.sleep(60)
                        continue
                    ohlc_df = process_ohlc(ohlc_df)
                    logging.debug(f"Fetched OHLC DataFrame shape: {ohlc_df.shape}")
                except Exception as e:
                    logging.error(f"Error fetching OHLC data: {str(e)}")
                    time.sleep(60)
                    continue

                # Fetch calendar data
                try:
                    calendar_agg, event_map, currency_map, event_freq = fetch_live_calendar(
                        self.api_key, time_zone=self.time_zone, offset=self.offset)
                except Exception as e:
                    logging.error(f"Error fetching calendar data in loop: {str(e)}")
                    time.sleep(60)
                    continue
                
                # Combine with calendar data
                combined_df = combine_data(ohlc_df, calendar_agg)
                current_time = combined_df[self.env.time_col].iloc[-1]

                if last_time == current_time:
                    logging.info(f"Time: {current_time}, No new data, waiting 60 seconds...")
                    time.sleep(60)  # Wait for the next bar (1 minutes)
                    continue
                last_time = current_time

                latest_data = combined_df.tail(1)
                self.env.update_data(latest_data)  # Use only the latest bar
 
                # print("Combined data:\n", combined_df.to_string())
                # print("Latest data:\n", latest_data.to_string(index=False))

                self.env.current_step = len(self.env.dt_datetime) - 1

                # Predict using the latest row
                try:
                    obs = {
                        "ohlc_data": np.array(self.env.cached_ohlc_data[-1], dtype=np.float32),
                        "event_ids": self.env.cached_economic_data[-1]["event_ids"],
                        "currency_ids": self.env.cached_economic_data[-1]["currency_ids"],
                        "economic_numeric": self.env.cached_economic_data[-1]["numeric"],
                        "portfolio_data": np.array(
                            [self.env.balance, self.env.total_equity, self.env.max_draw_down_pct] + 
                            self.env.current_holding + self.env.current_draw_downs,
                            dtype=np.float32
                        ),
                        "weekday": np.array([self.env.cached_time_features[-1][0]], dtype=np.int32),
                        "hour_features": np.array(self.env.cached_time_features[-1][1:], dtype=np.float32)
                    }
                    action, _states = self.model.predict(obs, deterministic=True)
                
                    # logging.info(f"Predicted action: {action}, action[0]: {action[0]}")
                    # print(f"Predicted action: {action}, action[0]: {action[0]}")
                    # Execute trade
                    trade_result = execute_trade(self.account_id, self.symbol, action[0], config_file=self.config_file)
                    logging.info(f"Trade result: {trade_result}")
                    if trade_result is not None:
                        trade_ticket = str(trade_result["ticket"].order)
                        save_trade_indicators_and_events(trade_ticket=trade_ticket, row_data=latest_data)
                    obs, reward, done, info = self.env.step(action)
                    logging.info(f"Time: {current_time}, Action: {action}, Reward: {reward}, Result: {trade_result}, Balance: {self.env.balance}")
                except Exception as e:
                    logging.error(f"Error in prediction or trade execution: {str(e)}")
                    time.sleep(60)
                    continue

                # Wait for the next bar (adjust timing as needed)
                logging.info(f"Completed iteration, sleeping for 1 hour at {datetime.datetime.now().time()}")
                time.sleep(3600 - (datetime.datetime.now().second + 60))  # Sync to next hour
                # time.sleep(60)

        except Exception as e:
            logging.error(f"Unexpected error in real_time_trading: {str(e)}")
            # Check if the error is related to MT5
            if "MetaTrader5" in str(e) or "connection" in str(e).lower():
                logging.error("MT5-related error detected, attempting reconnection")
                if self.reconnect_mt5():
                    logging.info("MT5 reconnected, restarting run()")
                    self.run()
                    return
                else:
                    logging.error("Failed to reconnect to MT5, stopping trading")
                    self.stop_event.set()
                    emit_status()  # Notify frontend
                    return
            else:
                # For non-MT5 errors, attempt to reconnect and restart
                logging.warning("Non-MT5 error detected, attempting to recover...")
                if self.reconnect_mt5():
                    logging.info("MT5 reconnected, restarting run()")
                    self.run()
                    return
                else:
                    logging.error("Failed to reconnect to MT5 after non-MT5 error, stopping trading")
                    self.stop_event.set()
                    emit_status()  # Notify frontend
                    return
        finally:
            logging.info("Real-time trading loop terminated")
            # Ensure MT5 connection is closed
            try:
                mt5.shutdown()
                logging.info("MT5 connection closed")
            except Exception as e:
                logging.error(f"Error closing MT5 connection: {str(e)}")

    def monitor_closed_trades(self):
        """Monitor closed trades in MT5 and update the database."""
        # Import emit_status here to avoid circular import
        from app import emit_status

        if not initialize_mt5(self.login_id, self.password, self.server):
            logging.error("MT5 initialization failed for monitoring")
            self.stop_event.set()
            emit_status()  # Notify frontend
            return
    
        logging.info("Starting trade monitoring thread")
        while not self.stop_event.is_set():
            try:
                conn = sqlite3.connect(DB_FILE)
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT ticket FROM trades WHERE status = 'Open' AND accountId = ?", (self.account_id,))
                    open_tickets = [row[0] for row in cursor.fetchall()]
                    logging.info(f"Open tickets: {open_tickets}")
                    for ticket in open_tickets:
                        update_closed_trade_in_db(ticket, account_id=self.account_id)
                finally:
                    conn.close()  # Always close the connection
            except Exception as e:
                logging.error(f"Error in trade monitoring: {str(e)}")
            time.sleep(60)  # Check every minute
        logging.info("Trade monitoring thread terminated")
        # Ensure MT5 connection is closed
        try:
            mt5.shutdown()
            logging.info("MT5 connection closed in monitor thread")
        except Exception as e:
            logging.error(f"Error closing MT5 connection in monitor thread: {str(e)}")

    def start(self):
        """Start both trading and trade monitoring threads."""
        self.trading_thread = threading.Thread(target=self.run, daemon=True)
        self.monitor_thread = threading.Thread(target=self.monitor_closed_trades, daemon=True)
        self.trading_thread.start()
        self.monitor_thread.start()
        logging.info("Trading and monitoring threads started")

    def stop(self):
        """Stop both threads and clean up resources."""
        logging.info("Stopping RealTimeTrader...")
        self.stop_event.set()
        
        # Wait for threads to terminate
        if self.trading_thread and self.trading_thread.is_alive():
            logging.info("Waiting for trading thread to terminate...")
            self.trading_thread.join(timeout=10)
            if self.trading_thread.is_alive():
                logging.warning("Trading thread did not terminate within timeout")
            else:
                logging.info("Trading thread terminated successfully")
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            logging.info("Waiting for monitor thread to terminate...")
            self.monitor_thread.join(timeout=10)
            if self.monitor_thread.is_alive():
                logging.warning("Monitor thread did not terminate within timeout")
            else:
                logging.info("Monitor thread terminated successfully")

        # Clean up resources
        self.trading_thread = None
        self.monitor_thread = None
        self.env = None
        self.model = None
        try:
            mt5.shutdown()
            logging.info("MT5 connection closed during stop")
        except Exception as e:
            logging.error(f"Error closing MT5 connection during stop: {str(e)}")
        logging.info("RealTimeTrader stopped successfully")

