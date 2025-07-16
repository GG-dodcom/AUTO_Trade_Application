
import pandas as pd
import pandas_ta as ta
import numpy as np
import os


dataset = 'data/dataset/'
config = 'config/'
# --------------------------------------------
# Economic Calendar Data Preprocessing
# --------------------------------------------

# Function to determine if an event is relevant
def is_relevant_event(row):
    # Define relevant currencies
    relevant_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CNY', 'All']

    # Define keywords for event categories
    monetary_policy_keywords = [
        'Rate', 'FOMC', 'MPC', 'BOJ', 'ECB', 'SNB', 'RBA', 'RBNZ', 'BOC',
        'Speaks', 'Statement', 'Minutes', 'Press Conference'
    ]
    economic_indicators_keywords = [
        'PMI', 'CPI', 'PPI', 'Employment', 'Unemployment', 'GDP', 'Retail Sales',
        'Consumer Confidence', 'Trade Balance', 'Industrial Production',
        'Manufacturing Production'
    ]
    market_sentiment_keywords = [
        'Election', 'Summit', 'Meetings', 'Vote', 'Crisis', 'Bailout'
    ]
    commodity_keywords = ['Crude Oil Inventories', 'Gold']

    # Special case for USD: Include all high/medium impact events
    if row['currency'] == 'USD':
        return True  # USD events are always relevant if high/medium impact
    # For other currencies, apply the relevance filter
    if row['currency'] not in relevant_currencies:
        return False

    event = row['event'].lower()
    if any(keyword.lower() in event for keyword in monetary_policy_keywords):
        return True
    if any(keyword.lower() in event for keyword in economic_indicators_keywords):
        return True
    if any(keyword.lower() in event for keyword in market_sentiment_keywords):
        return True
    if any(keyword.lower() in event for keyword in commodity_keywords):
        return True
    return False

# Load the economic calendar data
calendar = pd.read_csv("forexfactory_economic_calendar.csv", low_memory=False)

# Apply the filter
calendar['is_relevant'] = calendar.apply(is_relevant_event, axis=1)

# Filter: Include all USD high/medium impact events + relevant high/medium impact events for other currencies
filtered_calendar = calendar[
    (calendar['is_relevant'] & calendar['impact'].isin(['High Impact Expected', 'Medium Impact Expected'])) |
    ((calendar['currency'] == 'USD') & calendar['impact'].isin(['High Impact Expected', 'Medium Impact Expected']))
]

# Drop the helper column
filtered_calendar = filtered_calendar.drop(columns=['is_relevant'])

# Columns to keep for RL learning
columns_to_keep = ['datetime', 'currency', 'event', 'impact', 'actual', 'forecast', 'previous']

# Remove unused columns
filtered_calendar = filtered_calendar[columns_to_keep]

# Convert datetime to proper format
filtered_calendar['datetime'] = pd.to_datetime(filtered_calendar['datetime'])

# Convert actual, forecast, and previous to numeric (handling empty strings as NaN)
filtered_calendar['actual'] = pd.to_numeric(filtered_calendar['actual'], errors='coerce')
filtered_calendar['forecast'] = pd.to_numeric(filtered_calendar['forecast'], errors='coerce')
filtered_calendar['previous'] = pd.to_numeric(filtered_calendar['previous'], errors='coerce')

# Remove rows where 'actual' is NaN
filtered_calendar = filtered_calendar.dropna(subset=['actual'])

# Count occurrences of each event
event_counts = filtered_calendar['event'].value_counts()

# Set threshold for minimum occurrences (e.g., 5)
min_occurrences = 5

# Filter out events occurring fewer than min_occurrences times
frequent_events = event_counts[event_counts >= min_occurrences].index
filtered_calendar = filtered_calendar[filtered_calendar['event'].isin(frequent_events)]

# Display results
print("Filtered DataFrame:")
print(filtered_calendar)
print("\nShape of Filtered DataFrame:", filtered_calendar.shape)
print("Number of Unique Events:", len(filtered_calendar['event'].unique()))

# --------------------------------------------------------------

# Load and preprocess economic calendar data
calendar_df = filtered_calendar
calendar_df['datetime'] = pd.to_datetime(calendar_df['datetime'])

# Step 1: Handle Missing Forecast and Surprise
calendar_df['previous'] = calendar_df['previous'].fillna(calendar_df['actual'])   # Use actual if previous is NaN
calendar_df['forecast'] = calendar_df['forecast'].fillna(calendar_df['previous'])  # Use previous as forecast if missing
calendar_df['forecast'] = calendar_df['forecast'].fillna(calendar_df['actual'])   # Use actual if previous is also NaN
calendar_df['surprise'] = (calendar_df['actual'] - calendar_df['forecast']).round(2)  # Recalculate surprise
calendar_df['surprise'] = calendar_df['surprise'].fillna(0)  # Default to 0 if still missing

# Step 3: Encode impact_code Features (Integer Encoding)
impact_map = {'High Impact Expected': 2, 'Medium Impact Expected': 1}
calendar_df['impact_code'] = calendar_df['impact'].map(impact_map)

# Step 4: Normalize Numeric Features (z-score standardization)
for col in ['actual', 'forecast', 'previous', 'surprise']:
    mean = calendar_df[col].mean()
    std = calendar_df[col].std()
    calendar_df[f'{col}_norm'] = (calendar_df[col] - mean) / std

# Step 5: Add Event Frequency and Save It
event_freq = calendar_df['event'].value_counts()  # Frequency of each unique event
calendar_df['event_freq'] = calendar_df['event'].map(event_freq)

# Save event_freq to a file
with open(f"{config}event_freq.pkl", "wb") as f:
    pickle.dump(event_freq.to_dict(), f)  # Save as a dictionary for easier retrieval

# Step NEW: Create Event and Currency Mappings
event_map = {e: i for i, e in enumerate(calendar_df['event'].unique())}  # Unique event names to IDs
currency_map = {c: i for i, c in enumerate(calendar_df['currency'].unique())}  # Unique currencies to IDs

# Step 2 & 6: Event Window Expansion and Aggregation
calendar_df['interval'] = calendar_df['datetime'].dt.ceil('1h')
event_agg = calendar_df.groupby('interval').apply(
    lambda x: x[['event', 'currency', 'impact_code', 'actual_norm', 'forecast_norm',
                 'previous_norm', 'surprise_norm', 'event_freq']].to_dict('records')
).reset_index(name='events')
event_agg.rename(columns={'interval': 'time'}, inplace=True)

# Save preprocessed data and mappings
event_agg.to_pickle(f"{dataset}calendar_preprocessed.pkl")
import pickle
with open(f"{config}event_map.pkl", "wb") as f:
    pickle.dump(event_map, f)
with open(f"{config}currency_map.pkl", "wb") as f:
    pickle.dump(currency_map, f)

print("Preprocessed Calendar Dataset:")
print(event_agg.head())
print(f"Event Map (sample): {dict(list(event_map.items())[:5])}")
print("Len of event_map", len(event_map))
print(f"Currency Map: {currency_map}")

# Maximum number of events in 1 hour
hourly_event_counts = calendar_df.groupby('interval').size()
# Find the maximum number of events in one hour
max_events_in_one_hour = hourly_event_counts.max()
print("Maximum number of events happening in one hour:", max_events_in_one_hour)


# --------------------------------------------
# OHLC Data Preprocessing
# --------------------------------------------

# Load OHLC data, concatenate all dataframes into a single dataframe
# Convert tick data into 1h OHLC data

# Directory where the files are stored
directory_path = f'{dataset}XAUUSD'

# Initialize an empty list to store dataframes for each year
all_data = []

for year in range(2013, 2025):
    if year == 2024: year = '2024-09'
    # Construct the file path for each year
    file_path = os.path.join(directory_path, f'{year}XAUUSD-TICK-No Session.csv')
    
    # Load the data if the file exists
    try:
        tick_data = pd.read_csv(file_path)
        print(f"Loaded data for {year}")
        print(tick_data)
        
        # Convert 'DateTime' column to datetime format
        tick_data['DateTime'] = pd.to_datetime(tick_data['DateTime'], format="%Y%m%d %H:%M:%S.%f", errors='coerce')

        # Set DATETIME as the index
        tick_data.set_index('DateTime', inplace=True)
        
        # Apply filter for 2024 data to only include records up to 2024-07-31
        if year == '2024-09':
            tick_data = tick_data[tick_data.index <= '2024-07-31']
        
        # Convert it to OHLC Data
        hourly_ohlc = tick_data['Bid'].resample('h').ohlc()
        print(hourly_ohlc)

        # Drop rows where any of the 'close' columns is null values (market close)
        # before drop null
        print("\nMissing Values Before Drop Null:", hourly_ohlc.isnull().sum())
        # Drop rows where any of the 'close' columns have null values
        hourly_ohlc = hourly_ohlc.dropna(subset=['close'])
        # after drop null
        print("\nMissing Values After Drop Null:", hourly_ohlc.isnull().sum())

        # Add 'day' column
        hourly_ohlc = hourly_ohlc.assign(day=hourly_ohlc.index.dayofweek)

        # Append the data to the list
        all_data.append(hourly_ohlc)
    
    except FileNotFoundError:
        print(f"File for {year} not found.")

combined_data = pd.concat(all_data, ignore_index=True)

# --------------------------------------------------------------

def add_technical_indicators(df):
    # Calculate technical indicators
    # 1. Relative Strength Index (RSI)
    df['RSI'] = ta.rsi(df['close'], length=14)

    # 2. Moving Average Convergence Divergence (MACD)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    macd.columns = ['MACD', 'MACD_Histogram', 'MACD_Signal']
    df = df.join(macd)

    # 3. Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    df = df.join(bbands)

    # 4. Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df = df.join(stoch)

    # 5. Average Directional Index (ADX)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']

    # 6. Commodity Channel Index (CCI)
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)

    # 7. Parabolic Stop and Reverse (Parabolic SAR)
    df['PSAR'] = ta.psar(df['high'], df['low'], df['close'], af=0.02, max_af=0.2)['PSARl_0.02_0.2']

    # 8. Simple Moving Average (SMA)
    df['SMA'] = ta.sma(df['close'], length=50)

    # 9. Fibonacci Retracement Levels
    # Calculate the high and low over a specified period
    period = 20  # You can adjust this period as needed
    df['High_Max'] = df['high'].rolling(window=period).max()
    df['Low_Min'] = df['low'].rolling(window=period).min()
    # Calculate Fibonacci levels
    df['Fib_23.6'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.236
    df['Fib_38.2'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.382
    df['Fib_50.0'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.500
    df['Fib_61.8'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 0.618
    df['Fib_100.0'] = df['High_Max'] - (df['High_Max'] - df['Low_Min']) * 1.000
    # Drop intermediate columns
    df.drop(columns=['High_Max', 'Low_Min'], inplace=True)
    
    # 10. ATR (Average True Range)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Display the DataFrame with the new indicators
    print(df.tail())

    return df

combined_data = add_technical_indicators(combined_data)

# Save the combined data to a CSV file
combined_data.to_csv(os.path.join(f'{dataset}2013-202407.csv'), index=False)

# --------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

ohlc_df = pd.read_csv(f"{dataset}2013-202407.csv", low_memory=False)
ohlc_df['DateTime'] = pd.to_datetime(ohlc_df['DateTime'])

# Rename columns to match tgym requirements
ohlc_df = ohlc_df.rename(columns={
    "DateTime": "time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "day": "weekday"
})

# Add required columns: 'asset' and 'day' (weekday)
ohlc_df["symbol"] = "XAUUSD"  # Single asset in this case

# Add time-based feature
ohlc_df["hour"] = ohlc_df["time"].dt.hour
ohlc_df["hour_sin"] = np.sin(2 * np.pi * ohlc_df["hour"] / 24)  # Sine encoding
ohlc_df["hour_cos"] = np.cos(2 * np.pi * ohlc_df["hour"] / 24)  # Cosine encoding

# Select numeric columns (exclude 'time' and 'symbol')
numeric_cols = ohlc_df.select_dtypes(include=['float64', 'int64']).columns

# Forward fill all numeric columns
ohlc_df[numeric_cols] = ohlc_df[numeric_cols].fillna(method='ffill')

# Drop rows where indicators are NaN
ohlc_df = ohlc_df.dropna()

# Ensure no NaNs remain
assert not ohlc_df.isnull().any().any(), "NaNs still present!"

# Create new columns for normalize
cols = ['Open', 'High', 'Low', 'Close']
for col in cols:
    ohlc_df[f'{col}_norm'] = ohlc_df[col]

# Normalization
# List of columns to normalize
columns_to_normalize = [
    'Open_norm', 'High_norm', 'Low_norm', 'Close_norm',
    'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',  # Momentum indicators
    'STOCHk_14_3_3', 'STOCHd_14_3_3',  # Stochastic oscillators
    'ADX', 'CCI', 'SMA',  # Trend indicators
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',  # Bollinger Bands
    'Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_100.0',  # Fibonacci levels
    'ATR',  # Measure Market Volatility
]
# Rolling Standardization (window size = 100, adjust as needed)
window_size = 500
for col in columns_to_normalize:
    rolling_mean = ohlc_df[col].rolling(window=window_size, min_periods=1).mean()
    rolling_std = ohlc_df[col].rolling(window=window_size, min_periods=1).std()
    ohlc_df.loc[:, col] = (ohlc_df[col] - rolling_mean) / rolling_std

# Fill any NaNs (e.g., due to division by zero in early rows with zero std)
ohlc_df.fillna(0, inplace=True)

ohlc_df.to_csv(f"{dataset}xauusd_2013_202407_1h.csv", index=False)
print("Data saved to xauusd_2013_202407_1h.csv")



# ------------------------------------------------------------
# Combine Economic Calendar + OHLC Data Preprocessing
# ------------------------------------------------------------

event_agg = pd.read_pickle(f"{dataset}calendar_preprocessed.pkl")
event_agg['time'] = pd.to_datetime(event_agg['time'])
print(event_agg.head())

# Load OHLC data
ohlc_df = pd.read_csv(f"{dataset}xauusd_2013_202407_1h.csv", low_memory=False)
ohlc_df['time'] = pd.to_datetime(ohlc_df['time'])
print(ohlc_df.head())

df = pd.merge(
    ohlc_df,
    event_agg,
    on='time',
    how='left'
).fillna({'events': '[]'})  # Empty list for intervals with no events

# Save to pickle
df.to_pickle(f"{dataset}combined_dataset.pkl")



# --------------------------------------------
# Split Data for Training and Testing
# Train 70%, Validation 15%, Testing 15%
# --------------------------------------------

# Load combine data
df = pd.read_pickle(f"{dataset}combined_dataset.pkl")
df['time'] = pd.to_datetime(df['time'])
print(df.head())

print(df.isna().sum())

# Split the data chronologically
total_rows = len(df)
train_end = int(total_rows * 0.7)    # 70% for training
val_end = int(total_rows * 0.85)      # 15% for validation (70% + 15% = 85%)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Verify splits
print(f"Total rows: {total_rows}")
print(f"Training set: {len(train_df)} rows ({train_df['time'].min()} to {train_df['time'].max()})")
print(f"Validation set: {len(val_df)} rows ({val_df['time'].min()} to {val_df['time'].max()})")
print(f"Test set: {len(test_df)} rows ({test_df['time'].min()} to {test_df['time'].max()})")

# Save the split datasets
train_df.to_csv(f"{dataset}xauusd_train.csv", index=False)
val_df.to_csv(f"{dataset}xauusd_val.csv", index=False)
test_df.to_csv(f"{dataset}xauusd_test.csv", index=False)
print("Datasets saved: xauusd_train.csv, xauusd_val.csv, xauusd_test.csv")
