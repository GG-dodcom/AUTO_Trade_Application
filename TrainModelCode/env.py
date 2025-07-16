import datetime
import math
import random
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import csv

from util.log_render import render_to_file
from util.plot_chart import TradingChart
from util.read_config import EnvConfig
# from meta.env_fx_trading.util.log_render import render_to_file
# from meta.env_fx_trading.util.plot_chart import TradingChart
# from meta.env_fx_trading.util.read_config import EnvConfig

class tgym(gym.Env):
    """forex/future/option trading gym environment
    1. Three action space (0 Buy, 1 Sell, 2 Nothing)
    2. Multiple trading pairs (EURUSD, GBPUSD...) under same time frame
    3. Timeframe from 1 min to daily as long as use candlestick bar (Open, High, Low, Close)
    4. Use StopLose, ProfitTaken to realize rewards. each pair can configure it own SL and PT in configure file
    5. Configure over night cash penalty and each pair's transaction fee and overnight position holding penalty
    6. Split dataset into daily, weekly or monthly..., with fixed time steps, at end of len(df). The business
        logic will force to Close all positions at last Close price (game over).
    7. Must have df column name: [(time_col),(asset_col), Open,Close,High,Low,day] (case sensitive)
    8. Addition indicators can add during the data process. 78 available TA indicator from Finta
    9. Customized observation list handled in json config file.
    10. ProfitTaken = fraction_action * max_profit_taken + SL.
    11. SL is pre-fixed
    12. Limit order can be configure, if limit_order == True, the action will preset buy or sell at Low or High of the bar,
        with a limit_order_expiration (n bars). It will be triggered if the price go cross. otherwise, it will be drop off
    13. render mode:
        human -- display each steps realized reward on console
        file -- create a transaction log
        graph -- create transaction in graph (under development)
    14.
    15. Reward, we want to incentivize profit that is sustained over long periods of time.
        At each step, we will set the reward to the account balance multiplied by
        some fraction of the number of time steps so far.The purpose of this is to delay
        rewarding the agent too fast in the early stages and allow it to explore
        sufficiently before optimizing a single strategy too deeply.
        It will also reward agents that maintain a higher balance for longer,
        rather than those who rapidly gain money using unsustainable strategies.
    16. Observation_space contains all of the input variables we want our agent
        to consider before making, or not making a trade. We want our agent to “see”
        the forex data points (Open price, High, Low, Close, time serial, TA) in the game window,
        as well a couple other data points like its account balance, current positions,
        and current profit.The intuition here is that for each time step, we want our agent
        to consider the price action leading up to the current price, as well as their
        own portfolio’s status in order to make an informed decision for the next action.
    17. reward is forex trading unit Point, it can be configure for each trading pair
    18. To make the unrealized profit reward reflect market conditions, we’ll compute ATR for each asset and use it to scale the reward dynamically.
    """

    metadata = {"render.modes": ["graph", "human", "file", "none"]}

    def __init__(
        self,
        df,
        event_map,
        currency_map,
        env_config_file="./neo_finrl/env_fx_trading/config/gdbusd-test-1.json",
    ):
        assert df.ndim == 2, "DataFrame must be 2-dimensional"  
        super(tgym, self).__init__()
        self.cf = EnvConfig(env_config_file)
        self.observation_list = self.cf.env_parameters("observation_list")  # Contains Open_norm, High_norm, etc.
        self.original_ohlc_cols = ["Open", "High", "Low", "Close"]  # Define original columns explicitly

        # Economic data mappings
        self.event_map = event_map
        self.currency_map = currency_map
        self.max_events = 8

        self.df = df.copy()
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

        # Cache normalized OHLC data for model
        self.cached_ohlc_data = [self.get_observation_vector(_dt) for _dt in self.dt_datetime]
        # Cache original OHLC data for environment calculations
        self.cached_original_ohlc = [self.get_original_ohlc_vector(_dt) for _dt in self.dt_datetime]
        self.cached_economic_data = [self.get_economic_vector(_dt) for _dt in self.dt_datetime]
        self.cached_time_features = self.df[["_day", "hour_sin", "hour_cos"]].values.tolist()
        self.cached_time_serial = (
            self.df[["_time", "_day"]].sort_values("_time").drop_duplicates().values.tolist()
        )

        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=3, shape=(len(self.assets),), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "ohlc_data": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.assets) * len(self.observation_list),), dtype=np.float32), # 24
            "event_ids": spaces.Box(low=0, high=len(self.event_map)-1, shape=(self.max_events,), dtype=np.int32),
            "currency_ids": spaces.Box(low=0, high=len(self.currency_map)-1, shape=(self.max_events,), dtype=np.int32),
            "economic_numeric": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_events * 6,), dtype=np.float32),
            "portfolio_data": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 2 * len(self.assets),), dtype=np.float32),
            "weekday": spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            "hour_features": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # hour_sin, hour_cos
        })

        print(
            f"initial done:\n"
            f"observation_list:{self.observation_list}\n"
            f"assets:{self.assets}\n"
            f"time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}\n"
            f"events: {len(self.event_map)}, currencies: {len(self.currency_map)}"
        )
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max )
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
            for val, name in zip([self._o, self._h, self._l, self._c], ["Open", "High", "Low", "Close"]):
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f"NaN/Inf in {name} at step {self.current_step}, asset {self.assets[i]}: {val}")

            # Extract integer action type and fractional part
            _action = math.floor(action)  # 0=Buy, 1=Sell, 2=Nothing
            rewards[i] = self._calculate_reward(i, done, _action)  # Pass action for exploration reward
            if np.isnan(rewards[i]) or np.isinf(rewards[i]):
                raise ValueError(f"NaN/Inf in reward for asset {self.assets[i]}: {rewards[i]}")
            
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

                if np.isnan(_profit_taken) or np.isinf(_profit_taken):
                    raise ValueError(f"NaN/Inf in _profit_taken for asset {self.assets[i]}: {action}, {_action}, {_profit_taken}")
                
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
                    # Debug
                    # print(f"New Limit Order - Asset: {self.assets[i]}, Type: {'Buy' if _action == 0 else 'Sell'}, "
                    #     f"Take Profit: {_profit_taken}, Stop Loss: {self.cf.symbol(self.assets[i], 'stop_loss_max')}")
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
                    # Debug
                    # print(f"New Market Order - Asset: {self.assets[i]}, Type: {'Buy' if _action == 0 else 'Sell'}, "
                    #     f"Take Profit: {_profit_taken}, Stop Loss: {self.cf.symbol(self.assets[i], 'stop_loss_max')}")
                    
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_live.append(transaction)
            
        # Debug
        # print(f"Take Action: Live Transactions: {[tr['Type'] for tr in self.transaction_live]}")

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
                    if np.isnan(tr["Reward"]) or np.isinf(tr["Reward"]):
                        raise ValueError(f"NaN/Inf in reward after overnight penalty for {tr['Symbol']}: {tr['Reward']}")

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
                if np.isnan(_total_reward) or np.isinf(_total_reward):
                    raise ValueError(f"NaN/Inf in _total_reward for {self.assets[i]}: {_total_reward}")
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
        if np.isnan(_p) or np.isinf(_p):
            raise ValueError(f"NaN/Inf in profit/loss for transaction {tr['Ticket']}: {_p}")
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)  # Realized profit/loss
        tr["Status"] = status  # 1=SL/PT, 2=Forced close, 3=Canceled limit, 4=Expired limit
        tr["CloseTime"] = self._t
        tr["CloseStep"] = self.current_step
        self.balance += int(tr["Reward"])
        
        self.total_equity -= int(abs(tr["Reward"]))
        if np.isnan(self.balance) or np.isinf(self.balance):
            raise ValueError(f"NaN/Inf in balance after managing transaction {tr['Ticket']}: {self.balance}")
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

        # Debug
        # print(f"Transaction {tr['Ticket']} Closed: Profit/Loss={tr['Reward']}, New Balance={self.balance}")


    def analyze_transaction_history(self):
        if not self.transaction_history:
            metrics = {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "sharpe_ratio": 0.0, "total_profit": 0.0}
        else:
            trades = len(self.transaction_history)
            rewards = [tr["Reward"] for tr in self.transaction_history]
            wins = sum(1 for r in rewards if r > 0)
            losses = sum(1 for r in rewards if r < 0)
            gross_profit = sum(r for r in rewards if r > 0)
            gross_loss = abs(sum(r for r in rewards if r < 0))
            
            win_rate = wins / trades if trades > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            
            # Sharpe Ratio (simplified, assumes risk-free rate = 0)
            returns = np.array(rewards, dtype=np.float32)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            total_profit = sum(rewards)
            metrics = {
                "trades": trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "total_profit": total_profit
            }
        
        # Prepare metrics with timestamp and episode
        metrics["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["episode"] = self.episode
        
        # Check if file exists and is empty to write header
        import os
        file_exists = os.path.exists(self.analyze_transaction_history_log_filename)
        file_empty = file_exists and os.stat(self.analyze_transaction_history_log_filename).st_size == 0
        
        # Append to log file with header if it's new or empty
        with open(self.analyze_transaction_history_log_filename, 'a', newline='') as f:
            fieldnames = ["timestamp", "episode", "trades", "win_rate", "profit_factor", "sharpe_ratio", "total_profit"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only if file doesn't exist or is empty
            if not file_exists or file_empty:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        return metrics
    
    def _validate_obs(self, obs, step):
        # print(f"Step {step} Observation check:")
        for key, value in obs.items():
            # print(f"{key}: {value.shape}")
            if np.any(np.isnan(value)):
                raise ValueError(f"NaN detected in observation[{key}] at step {step}: {value}")
            if np.any(np.isinf(value)):
                raise ValueError(f"Inf detected in observation[{key}] at step {step}: {value}")

    def step(self, actions):
        old_balance = self.balance
        self.current_step += 1

        # Define termination and truncation conditions
        terminated = self.balance <= 0  # Episode ends due to bankruptcy (terminal state)
        truncated = self.current_step == len(self.dt_datetime) - 1  # Episode ends due to max steps (time limit)
        done = terminated or truncated  # Combine into a single 'done' flag for VecEnv

        # For rendering or episode tracking, you might still check if either condition is true
        if done:
            self.done_information += f"Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n"
            self.visualization = True
            self.episode += 1  # Increment episode counter

        # Calculate base trading reward
        base_reward = self._take_action(actions, done)
        if np.isnan(base_reward) or np.isinf(base_reward):
            raise ValueError(f"NaN/Inf in base_reward at step {self.current_step}: {base_reward}")

        # Calculate unrealized profit from open positions
        unrealized_profit = 0
        atr_scaling = 0  # For market condition scaling
        for i, asset in enumerate(self.assets):
            atr = self.get_observation(self.current_step, i, "ATR")
            if np.isnan(atr) or np.isinf(atr):
                raise ValueError(f"NaN/Inf in ATR for {asset} at step {self.current_step}: {atr}")
            atr_scaling += atr
            for tr in self.transaction_live:
                if tr["Symbol"] == asset:
                    if tr["Type"] == 0:  # Buy
                        unrealized = (self._c - tr["ActionPrice"]) * self.cf.symbol(asset, "point")
                    else:  # Sell
                        unrealized = (tr["ActionPrice"] - self._c) * self.cf.symbol(asset, "point")
                    if np.isnan(unrealized) or np.isinf(unrealized):
                        raise ValueError(f"NaN/Inf in unrealized profit for {asset}, ticket {tr['Ticket']}: {unrealized}")
                    unrealized_profit += unrealized

        atr_scaling = atr_scaling / len(self.assets) if atr_scaling > 0 else 1  # Avoid division by 0
        if np.isnan(atr_scaling) or np.isinf(atr_scaling):
            raise ValueError(f"NaN/Inf in atr_scaling at step {self.current_step}: {atr_scaling}")


        # Sustained reward: only applies to unrealized/realized profits, scaled by ATR
        # adjust 0.01 to 0.05
        sustained_reward = (unrealized_profit + base_reward) * 0.05 / atr_scaling if self.transaction_live else 0
        if np.isnan(sustained_reward) or np.isinf(sustained_reward):
            raise ValueError(f"NaN/Inf in sustained_reward at step {self.current_step}: {sustained_reward}")

        if not self.transaction_live and all(math.floor(a) == 2 for a in actions):
            sustained_reward -= 100  # penalty to encourage exploration

        total_reward = base_reward + sustained_reward
        if np.isnan(total_reward) or np.isinf(total_reward):
            raise ValueError(f"NaN/Inf in total_reward at step {self.current_step}: {total_reward}")

        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
            if np.isnan(self.balance) or np.isinf(self.balance):
                raise ValueError(f"NaN/Inf in balance after overnight penalty at step {self.current_step}: {self.balance}")

        if self.balance != 0:
            self.max_draw_down_pct = abs(sum(self.max_draw_downs) / self.balance * 100)
            if np.isnan(self.max_draw_down_pct) or np.isinf(self.max_draw_down_pct):
                raise ValueError(f"NaN/Inf in max_draw_down_pct at step {self.current_step}: {self.max_draw_down_pct}")

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
        self._validate_obs(obs, self.current_step)  # Add validation here
        print(f"Step {self.current_step}: Base Reward={base_reward}, Sustained Reward={sustained_reward}, Total={total_reward}, Balance={self.balance}, Holding={self.current_holding}, Drawdowns={self.current_draw_downs}")
        
        # Info dictionary remains unchanged
        info = {"Close": self.tranaction_close_this_step}

        # print(f"Step {self.current_step}: Old Balance={old_balance}, Base Reward={base_reward}, "
        #     f"Sustained Reward={sustained_reward}, Total={total_reward}, New Balance={self.balance}")
        return obs, total_reward, terminated, truncated, info

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
            # Fetch from normalized observation cache
            try:
                col_pos = self.observation_list.index(col)
            except ValueError:
                raise ValueError(f"Column '{col}' not found in observation_list")
            return self.cached_ohlc_data[_step][_iter * len(self.observation_list) + col_pos]

    def get_original_ohlc_vector(self, _dt):
        """Fetch original OHLC data for all assets at a given timestamp."""
        v = []
        for a in self.assets:
            subset = self.df.query(f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            assert not subset.empty, f"No data for asset {a} at {_dt}"
            v += subset.loc[_dt, self.original_ohlc_cols].tolist()
        assert len(v) == len(self.assets) * len(self.original_ohlc_cols)
        return v

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list if cols is None else cols
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            assert not subset.empty
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

        if self.random_start:
            self.current_step = random.choice(range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

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
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        self.visualization = False

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

    def render(self, mode="human", title=None, **kwargs):
        if mode in ("human", "file"):
            printout = mode == "human"
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information,
            }
            render_to_file(**pm)
            if self.log_header:
                self.log_header = False
        elif mode == "graph" and self.visualization:
            print("plotting...")
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
