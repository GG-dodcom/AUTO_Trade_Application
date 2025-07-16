
import datetime
import ast
import torch.nn.functional as F
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import csv

from util.read_config import EnvConfig

class RealTimeTgym(gym.Env):
    metadata = {"render.modes": ["graph", "human", "file", "none"]}

    def __init__(
        self,
        df,
        event_map,
        currency_map,
        window_size=500,
        env_config_file="./neo_finrl/env_fx_trading/config/gdbusd-test-1.json",
    ):
        super().__init__()
        self.cf = EnvConfig(env_config_file)
        self.observation_list = self.cf.env_parameters("observation_list")  # Contains Open_norm, High_norm, etc.
        self.original_ohlc_cols = ["Open", "High", "Low", "Close"]  # Define original columns explicitly
        self.window_size = window_size

        # Economic data mappings
        self.event_map = event_map
        self.currency_map = currency_map
        self.max_events = 8

        # Initialize with 500-bar window
        if len(df) < window_size:
            raise ValueError(f"Initial DataFrame must have at least {window_size} rows, got {len(df)}")
        self.df = df.sort_values(self.cf.env_parameters("time_col")).tail(window_size).copy()
        if 'events' not in self.df.columns:
            raise ValueError("DataFrame must contain an 'events' column")

        def parse_events(x):
            if isinstance(x, str):
                try:
                    parsed = ast.literal_eval(x)
                    return parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    raise ValueError(f"Failed to parse events string: {x}")
            return x if isinstance(x, list) else []

        self.df['events'] = self.df['events'].apply(parse_events)

        if not isinstance(self.df['events'].iloc[0], list):
            raise ValueError("'events' must be a list")
        if self.df['events'].iloc[0] and not isinstance(self.df['events'].iloc[0][0], dict):
            raise ValueError("Elements in 'events' must be dictionaries")

        # Check for NaNs in input DataFrame
        if self.df.isna().any().any():
            raise ValueError(f"NaN found in input DataFrame: {self.df[self.df.isna().any(axis=1)]}")

        self.balance_initial = self.cf.env_parameters("balance")
        self.over_night_cash_penalty = self.cf.env_parameters("over_night_cash_penalty")
        self.asset_col = self.cf.env_parameters("asset_col")
        self.time_col = self.cf.env_parameters("time_col")
        self.random_start = self.cf.env_parameters("random_start")
        log_file_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_filename = (
            self.cf.env_parameters("log_filename")
            + log_file_datetime
            + ".csv"
        )
        self.analyze_transaction_history_log_filename = ("transaction_history_log" 
            + log_file_datetime
            + ".csv")

        self.df["_time"] = self.df[self.time_col]
        self.df["_day"] = self.df["weekday"]
        self.assets = self.df[self.asset_col].unique()
        self.dt_datetime = self.df[self.time_col].sort_values().unique()
        self.df = self.df.set_index(self.time_col)
        self.visualization = False

        # Reset values
        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.current_step = 0
        self.episode = 0  # Start from 0, increment on episode end
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True

        # Cache data for the initial window
        self._update_caches()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=3, shape=(len(self.assets),), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "ohlc_data": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.assets) * len(self.observation_list),), dtype=np.float32),
            "event_ids": spaces.Box(low=0, high=len(self.event_map)-1, shape=(self.max_events,), dtype=np.int32),
            "currency_ids": spaces.Box(low=0, high=len(self.currency_map)-1, shape=(self.max_events,), dtype=np.int32),
            "economic_numeric": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_events * 6,), dtype=np.float32),
            "portfolio_data": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 2 * len(self.assets),), dtype=np.float32),
            "weekday": spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            "hour_features": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # hour_sin, hour_cos
        })

        print(f"Initialized RealTimeTgym:\n"
              f"observation_list:{self.observation_list}\n"
              f"Assets: {self.assets}\n"
              f"Time range: {min(self.dt_datetime)} -> {max(self.dt_datetime)}\n"
              f"events: {len(self.event_map)}\n"
              f"currencies: {len(self.currency_map)}")

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_data(self, new_row):
        """Append new real-time data at the end and shift the window."""
        new_row = new_row.copy()
        if not isinstance(new_row, pd.DataFrame):
            new_row = pd.DataFrame([new_row]).copy()
        required_cols = self.observation_list + ['weekday', 'hour_sin', 'hour_cos', 'events', self.time_col, self.asset_col]
        missing_cols = [col for col in required_cols if col not in new_row.columns]
        if missing_cols:
            raise ValueError(f"New row missing columns: {missing_cols}")
                
        # Parse 'events' to ensure it's a list
        def parse_events(x):
            if isinstance(x, str):
                try:
                    parsed = ast.literal_eval(x)
                    return parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    raise ValueError(f"Failed to parse events string: {x}")
            return x if isinstance(x, list) else []
        
        new_row['events'] = new_row['events'].apply(parse_events)
        new_row["_time"] = new_row[self.time_col]
        new_row["_day"] = new_row["weekday"]
        new_row = new_row.set_index(self.time_col)
        
        # Drop duplicates based on index and asset_col before concatenation
        self.df = pd.concat([self.df, new_row]).sort_index()
        self.df = self.df.drop_duplicates(subset=[self.asset_col], keep='last').tail(self.window_size)
        self.dt_datetime = self.df.index.sort_values().unique()
        self._update_caches()

    def _update_caches(self):
        """Update cached data after new row is added."""
        self.cached_ohlc_data = [self.get_observation_vector(_dt) for _dt in self.dt_datetime]
        self.cached_original_ohlc = [self.get_original_ohlc_vector(_dt) for _dt in self.dt_datetime]
        self.cached_economic_data = [self.get_economic_vector(_dt) for _dt in self.dt_datetime]
        self.cached_time_features = self.df[["_day", "hour_sin", "hour_cos"]].values.tolist()
        self.cached_time_serial = self.df[["_time", "_day"]].sort_values("_time").drop_duplicates().values.tolist()
        print(f"Cache updated: len(dt_datetime)={len(self.dt_datetime)}, len(cached_original_ohlc)={len(self.cached_original_ohlc)}")

    def _take_action(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max)
        # _actions = np.floor(actions).astype(int)
        # _profit_takens = np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            raise ValueError(f"NaN/Inf in actions: {actions}")
        
        _action = 2
        _profit_taken = 0
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []

        # need use multiply assets
        for i, action in enumerate(actions):  # Actions are now floats between 0 and 3
            self._o = self.get_observation(self.current_step, i, "Open")
            self._h = self.get_observation(self.current_step, i, "High")
            self._l = self.get_observation(self.current_step, i, "Low")
            self._c = self.get_observation(self.current_step, i, "Close")
            self._t = self.get_observation(self.current_step, i, "_time")
            self._day = self.get_observation(self.current_step, i, "_day")
            
            # Extract integer action type and fractional part
            _action = math.floor(action)  # 0=Buy, 1=Sell, 2=Nothing
            rewards[i] = self._calculate_reward(i, done, _action)  # Pass action for exploration reward
     
            # print(f"Asset {self.assets[i]}: Action={action}, Action Float={_action}, Reward={rewards[i]}, Holding={self.current_holding[i]}")

            if self.cf.symbol(self.assets[i], "limit_order"):
                self._limit_order_process(i, _action, done)

            if (
                _action in (0, 1) 
                and not done 
                and self.current_holding[i] < self.cf.symbol(self.assets[i], "max_current_holding")):
                # Dynamically calculate PT using action fraction
                _profit_taken = math.ceil(
                    (action - _action) * self.cf.symbol(self.assets[i], "profit_taken_max")
                ) + self.cf.symbol(self.assets[i], "stop_loss_max")

                
                self.ticket_id += 1
                if self.cf.symbol(self.assets[i], "limit_order"):
                    # Limit order logic
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._l if _action == 0 else self._h,
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": -1,
                        "CloseStep": -1,
                    }
                    self.transaction_limit_order.append(transaction)
                else:
                    # Market order logic
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._c,
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }                    
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_live.append(transaction)
        
        return sum(rewards)

    def _calculate_reward(self, i, done, action):
        _total_reward = 0
        _max_draw_down = 0

        for tr in self.transaction_live[:]:  # Copy to avoid modification issues
            if tr["Symbol"] == self.assets[i]:
                _point = self.cf.symbol(self.assets[i], "point")
                # cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.symbol(self.assets[i], "over_night_penalty")
                
                if tr["Type"] == 0:  # Buy
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                    if done:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                        self.current_holding[i] -= 1  # Fix: Decrement here
                    elif self._l <= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._h >= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int((self._l - tr["ActionPrice"]) * _point)
                        _max_draw_down += self.current_draw_downs[i]
                        if self.current_draw_downs[i] < 0 and tr["MaxDD"] > self.current_draw_downs[i]:
                            tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == 1:  # Sell
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                    if done:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                        self.current_holding[i] -= 1  # Fix: Decrement here
                    elif self._h >= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._l <= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]
                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward

    def _limit_order_process(self, i, _action, done):
        for tr in self.transaction_limit_order[:]:
            if tr["Symbol"] == self.assets[i]:
                if tr["Type"] != _action or done:
                    self.transaction_limit_order.remove(tr)
                    tr["Status"] = 3
                    tr["CloseStep"] = self.current_step
                    self.transaction_history.append(tr)
                elif (tr["ActionPrice"] >= self._l and _action == 0) or (
                    tr["ActionPrice"] <= self._h and _action == 1):
                    tr["ActionStep"] = self.current_step
                    self.current_holding[i] += 1
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    if np.isnan(self.balance) or np.isinf(self.balance):
                        raise ValueError(f"NaN/Inf in balance after limit order for {self.assets[i]}: {self.balance}")
                    self.transaction_limit_order.remove(tr)
                    self.transaction_live.append(tr)
                    self.tranaction_open_this_step.append(tr)
                elif (tr["LimitStep"] + self.cf.symbol(self.assets[i], "limit_order_expiration")
                      > self.current_step):
                    tr["CloseStep"] = self.current_step
                    tr["Status"] = 4
                    self.transaction_limit_order.remove(tr)
                    self.transaction_history.append(tr)

    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)  # Realized profit/loss
        tr["Status"] = status  # 1=SL/PT, 2=Forced close, 3=Canceled limit, 4=Expired limit
        tr["CloseTime"] = self._t
        tr["CloseStep"] = self.current_step
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

        # Debug
        # print(f"Transaction {tr['Ticket']} Closed: Profit/Loss={tr['Reward']}, New Balance={self.balance}")

    def step(self, actions):
        old_balance = self.balance
        if not self.cached_ohlc_data or len(self.cached_ohlc_data) != len(self.dt_datetime):
            raise ValueError("Cache not updated correctly")

        # Define termination and truncation conditions
        terminated = self.balance <= 0  # Episode ends due to bankruptcy (terminal state)
        truncated = False  # No truncation in real-time
        done = terminated or truncated  # Combine into a single 'done' flag for VecEnv

        # For rendering or episode tracking, you might still check if either condition is true
        if done:
            self.done_information += f"Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n"
            self.visualization = True
            self.episode += 1  # Increment episode counter

        # Calculate base trading reward
        base_reward = self._take_action(actions, done)

        # Calculate unrealized profit from open positions
        unrealized_profit = 0
        atr_scaling = 0  # For market condition scaling
        for i, asset in enumerate(self.assets):
            atr = self.get_observation(self.current_step, i, "ATR_norm")
            atr_scaling += atr
            for tr in self.transaction_live:
                if tr["Symbol"] == asset:
                    if tr["Type"] == 0:  # Buy
                        unrealized = (self._c - tr["ActionPrice"]) * self.cf.symbol(asset, "point")
                    else:  # Sell
                        unrealized = (tr["ActionPrice"] - self._c) * self.cf.symbol(asset, "point")
                    unrealized_profit += unrealized

        atr_scaling = atr_scaling / len(self.assets) if atr_scaling > 0 else 1  # Avoid division by 0

        # Sustained reward: only applies to unrealized/realized profits, scaled by ATR
        sustained_reward = (unrealized_profit + base_reward) * 0.05 / atr_scaling if self.transaction_live else 0

        if not self.transaction_live and all(math.floor(a) == 2 for a in actions):
            sustained_reward -= 100  # penalty to encourage exploration

        total_reward = base_reward + sustained_reward
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty

        if self.balance != 0:
            self.max_draw_down_pct = abs(sum(self.max_draw_downs) / self.balance * 100)

        obs = {
            "ohlc_data": np.array(self.cached_ohlc_data[self.current_step], dtype=np.float32),
            "event_ids": self.cached_economic_data[self.current_step]["event_ids"],
            "currency_ids": self.cached_economic_data[self.current_step]["currency_ids"],
            "economic_numeric": self.cached_economic_data[self.current_step]["numeric"],
            "portfolio_data": np.array(
                [self.balance, self.total_equity, self.max_draw_down_pct] + self.current_holding + self.current_draw_downs,
                dtype=np.float32
            ),
            "weekday": np.array([self.cached_time_features[self.current_step][0]], dtype=np.int32),
            "hour_features": np.array(self.cached_time_features[self.current_step][1:], dtype=np.float32)
        }
        print(f"Step {self.current_step}: Action={actions}, Base Reward={base_reward}, Sustained Reward={sustained_reward}, Total={total_reward}, "
              f"Old Balance={old_balance}, Balance={self.balance}, Holding={self.current_holding}, Drawdowns={self.current_draw_downs}")
        
        # Info dictionary remains unchanged
        info = {"Close": self.tranaction_close_this_step}

        return obs, total_reward, done, info

    def get_observation(self, _step, _iter=0, col=None):
        if col is None:
            return self.cached_ohlc_data[_step]
        if col == "_day":
            return self.cached_time_serial[_step][1]
        elif col == "_time":
            return self.cached_time_serial[_step][0]
        elif col in self.original_ohlc_cols:
            # Fetch from original OHLC cache
            col_pos = self.original_ohlc_cols.index(col)
            return self.cached_original_ohlc[_step][_iter * len(self.original_ohlc_cols) + col_pos]
        else:
            col_pos = self.observation_list.index(col)
            return self.cached_ohlc_data[_step][_iter * len(self.observation_list) + col_pos]

    def get_original_ohlc_vector(self, _dt):
        """Fetch original OHLC data for all assets at a given timestamp."""
        v = []
        for a in self.assets:
            subset = self.df.query(f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            if subset.empty:
                raise ValueError(f"No data for asset {a} at {_dt}")
            v += subset.loc[_dt, self.original_ohlc_cols].tolist()
        assert len(v) == len(self.assets) * len(self.original_ohlc_cols)
        return v
    
    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list if cols is None else cols
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            if subset.empty:
                raise ValueError(f"No data for asset {a} at {_dt}")
            # Take the last row if duplicates exist
            v += subset.loc[_dt, cols].tolist()
        assert len(v) == len(self.assets) * len(cols)
        return v

    def get_economic_vector(self, _dt):
        subset = self.df.loc[_dt]
        events = subset['events'] if isinstance(subset, pd.Series) else subset['events'].iloc[0]
        event_ids = [self.event_map[e['event']] for e in events[:self.max_events]] + [0] * (self.max_events - len(events))
        currency_ids = [self.currency_map.get(e['currency'], 0) for e in events[:self.max_events]] + [0] * (self.max_events - len(events))
        numeric_fields = ['actual_norm', 'forecast_norm', 'previous_norm', 'surprise_norm', 'event_freq', 'impact_code']
        # numeric = [e[field] for e in events[:self.max_events] for field in numeric_fields] + [0] * (self.max_events * 6 - len(events) * 6)
        numeric = []
        for e in events[:self.max_events]:
            for field in numeric_fields:
                val = e.get(field, 0)
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    raise ValueError(f"NaN/Inf in economic data field {field} for event at {_dt}: {e}")
                numeric.append(float(val))
        numeric += [0] * (self.max_events * 6 - len(numeric))
        
        return {
            "event_ids": np.array(event_ids, dtype=np.int32),
            "currency_ids": np.array(currency_ids, dtype=np.int32),
            "numeric": np.array(numeric, dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility
        if seed is not None:
            self._seed(seed)
        self.current_step = 0
        obs = {
            "ohlc_data": np.array(self.cached_ohlc_data[self.current_step], dtype=np.float32),
            "event_ids": self.cached_economic_data[self.current_step]["event_ids"],
            "currency_ids": self.cached_economic_data[self.current_step]["currency_ids"],
            "economic_numeric": self.cached_economic_data[self.current_step]["numeric"],
            "portfolio_data": np.array(
                [self.balance, self.total_equity, self.max_draw_down_pct] + self.current_holding + self.current_draw_downs,
                dtype=np.float32
            ),
            "weekday": np.array([self.cached_time_features[self.current_step][0]], dtype=np.int32),
            "hour_features": np.array(self.cached_time_features[self.current_step][1:], dtype=np.float32)
        }

        info = {}
        return obs, info

    def get_window_obs(self):
        """Get observations for the full window for warm-up."""
        return [{
            "ohlc_data": np.array(self.cached_ohlc_data[i], dtype=np.float32),
            "event_ids": self.cached_economic_data[i]["event_ids"],
            "currency_ids": self.cached_economic_data[i]["currency_ids"],
            "economic_numeric": self.cached_economic_data[i]["numeric"],
            "portfolio_data": np.array(
                [self.balance, self.total_equity, self.max_draw_down_pct] + self.current_holding + self.current_draw_downs,
                dtype=np.float32
            ),
            "weekday": np.array([self.cached_time_features[i][0]], dtype=np.int32),
            "hour_features": np.array(self.cached_time_features[i][1:], dtype=np.float32)
        } for i in range(len(self.dt_datetime))]

