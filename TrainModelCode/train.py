import yfinance as yf
import pandas as pd
import os
import quantstats as qs
import pandas_ta as ta
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import sys

# Step 4: Configure and Run the Environment
import optuna
import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import datetime


from env import tgym

DATA_FOL = "./data/dataset"

# Load datasets
train_df = pd.read_csv(f"{DATA_FOL}xauusd_train.csv")
val_df = pd.read_csv(f"{DATA_FOL}xauusd_val.csv")
test_df = pd.read_csv(f"{DATA_FOL}xauusd_test.csv")
train_df["time"] = pd.to_datetime(train_df["time"])
val_df["time"] = pd.to_datetime(val_df["time"])
test_df["time"] = pd.to_datetime(test_df["time"])

# Load event & currency data mappings
event_map = None
currency_map = None
try:
    with open("event_map.pkl", "rb") as f:
        event_map = pickle.load(f)
        print("Event Map: ", event_map)
    with open("currency_map.pkl", "rb") as f:
        currency_map = pickle.load(f)
        print("Currency Map: ", currency_map)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    raise ValueError(f"Failed to load event_map or currency_map: {e}")
if event_map is None or currency_map is None:
    raise ValueError("Event or currency map is None")

train_env = tgym(df=train_df, event_map=event_map, currency_map=currency_map, env_config_file="xauusd_config_train.json")
train_env_vec = DummyVecEnv([lambda: train_env])
val_env = tgym(df=val_df, event_map=event_map, currency_map=currency_map, env_config_file="xauusd_config_train.json")
val_env_vec = DummyVecEnv([lambda: val_env])
test_env = tgym(df=test_df, event_map=event_map, currency_map=currency_map, env_config_file="xauusd_config_train.json")
test_env_vec = DummyVecEnv([lambda: test_env])


# Custom Logger class
class Logger:
    def __init__(self, filename="log.txt", log_dir="data/log", echo_to_console=True):
        self.echo_to_console = echo_to_console
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = os.path.join(self.log_dir, filename)
        self.original_stdout = sys.stdout
        with open(self.filename, 'a') as f:
            f.write(f"Log started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

    def write(self, message):
        with open(self.filename, 'a') as f:
            f.write(message)
        if self.echo_to_console:
            self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.original_stdout

# Evaluation function
def evaluate(model, env_vec, n_episodes=10, return_mean_reward=False, deterministic=False, quantstats=False):
    # print(f"env_vec.num_envs: {env_vec.num_envs}")
    # print(f"env_vec.action_space: {env_vec.action_space}")

    total_rewards = []
    total_profits = []  # Track actual trading profit
    metrics = []
    balances, timestamps, steps_taken, trades = [], [], [], []

    for episode in range(n_episodes):
        obs = env_vec.reset()
        done = np.array([False] * env_vec.num_envs)
        episode_rewards = np.zeros(env_vec.num_envs)
        episode_profit = 0  # Sum of realized profits/losses
        step_count = 0
        max_steps = len(env_vec.envs[0].dt_datetime)
        if (quantstats):
            balances = [env_vec.envs[0].balance]  # Initial balance
            timestamps = [env_vec.envs[0].dt_datetime[env_vec.envs[0].current_step]]
            steps_taken = [env_vec.envs[0].current_step]
            trades = []

        while not np.all(done) and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            # print(f"Action shape: {action.shape}, Action: {action}")

            obs, rewards, done, info = env_vec.step(action)
            # print(f"obs: {obs}")
            episode_rewards += rewards

            step_count += 1

            if quantstats:
                current_step = env_vec.envs[0].current_step
                timestamp = env_vec.envs[0].dt_datetime[current_step]
                timestamps.append(timestamp)
                steps_taken.append(current_step)
                balances.append(env_vec.envs[0].balance)
                if info[0]["Close"]:
                    trades.extend(info[0]["Close"])
            for tr in info[0]["Close"]:
                episode_profit += tr["Reward"]

            print(f"Episode {episode}, Step {step_count}: Action={action}, Reward={rewards[0]:.2f}")

        total_rewards.extend(episode_rewards)
        total_profits.append(episode_profit)
        metrics.append(env_vec.envs[0].analyze_transaction_history())  # Logs to file here

        if quantstats:
            # Create performance DataFrame with actual timestamps
            perf_df = pd.DataFrame({"time": timestamps, "balance": balances, "step": steps_taken})

            # Convert 'time' to datetime explicitly
            perf_df['time'] = pd.to_datetime(perf_df['time'])

            # Handle duplicates: Print all duplicate rows and remove the last occurrence
            if perf_df["time"].duplicated().any():
                # Find all rows involved in duplicates (keep=False marks all occurrences)
                duplicate_mask = perf_df["time"].duplicated(keep=False)
                duplicate_rows = perf_df[duplicate_mask]
                print(f"Duplicate timestamps found in Episode {episode}:")
                print(duplicate_rows.to_string(index=False))

                # Remove the last occurrence of each duplicate, keeping the first
                perf_df = perf_df[~perf_df["time"].duplicated(keep='first')]

            # Set 'time' as the index, ensuring it’s a DatetimeIndex
            perf_df.set_index("time", inplace=True)

            # Verify the index is a DatetimeIndex
            if not isinstance(perf_df.index, pd.DatetimeIndex):
                raise ValueError("The index of perf_df is not a DatetimeIndex after conversion.")

            # Calculate returns
            returns = perf_df["balance"].pct_change().fillna(0)

            # Trade-level analysis
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["Reward"])
            profits = trades_df["Reward"] if not trades_df.empty else pd.Series()
            win_rate = len(profits[profits > 0]) / len(profits) if len(profits) > 0 else 0
            profit_factor = (profits[profits > 0].sum() / abs(profits[profits < 0].sum())
                            if profits[profits < 0].sum() != 0 else float('inf'))

            # QuantStats report
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_dir = "data/log"
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"xauusd_test_{time_str}.html"
            qs.reports.html(
                returns,
                output=os.path.join(output_dir, file_name),
                title=f"XAUUSD Test Performance {time_str}"
            )

            # Download report in Colab
            try:
                from google.colab import files
                files.download(f"{output_dir}/{file_name}")
            except ImportError:
                print("Not running in Colab, report saved locally.")

            print(f"QuantStats Report Saved: {output_dir}/{file_name}")
            print(f"Sharpe Ratio: {qs.stats.sharpe(returns):.2f}")
            print(f"Max Drawdown: {qs.stats.max_drawdown(returns):.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Profit Factor: {profit_factor:.2f}")

    mean_reward = np.mean(total_rewards)
    mean_profit = np.mean(total_profits)

    avg_metrics = {
        k: np.mean([m[k] for m in metrics])
        for k in ["trades", "win_rate", "profit_factor", "sharpe_ratio", "total_profit"]
    }

    print(f"Deterministic={deterministic}, Mean Reward: {mean_reward:.2f}, Mean Profit: {mean_profit:.2f}")
    print(f"Average Metrics: {avg_metrics}")

    return mean_reward if return_mean_reward else mean_profit  # Return profit instead of reward for optimization

# Callback for logging training rewards
class RewardLogger(BaseCallback):
    """
    Steady Improvemen
    1. Monitor Training Rewards:
      Use Stable Baselines3’s verbose=1 or 2 to log rewards during model.learn(). Look for:
        - Increasing Mean Episode Reward: Rewards should trend upward over timesteps.
        - Reduced Variance: Reward variability should decrease as the policy stabilizes.
    2. Visualize: Plot rewards over time using a library like Matplotlib to confirm a steady upward trend.

    Good Generalization
    1. Validation Performance:
      Evaluate on your val_env_vec periodically (e.g., in the callback or after training). Compare:
        - Training vs. Validation Profit: If training profit increases but validation profit plateaus or drops, the model is overfitting.
        - Metrics: Use win_rate, profit_factor, and sharpe_ratio from evaluate to assess trading quality.
    2. Test Set Evaluation: After training, use test_env_vec with quantstats=True to check final generalization. Look for:
      - Positive mean_profit.
      - Reasonable sharpe_ratio (>1 is good), low max_drawdown.
    3. Signs of Success:
      - Steady reward increase during training.
      - Validation profit tracks training profit without significant divergence.
      - Test profit aligns with validation profit, indicating the model generalizes to unseen data.
    """
    def __init__(self, log_freq=1000, plot_dir="plots"):
        super().__init__()
        self.log_freq = log_freq
        self.rewards = []
        self.steps = []
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def _on_step(self):
        self.rewards.append(self.locals["rewards"][0])
        self.steps.append(self.n_calls)
        if self.n_calls % self.log_freq == 0:
            mean_reward = np.mean(self.rewards[-self.log_freq:])
            print(f"Step {self.n_calls}: Mean Reward = {mean_reward:.2f}")
        return True

    def _on_training_end(self):
        # Plot rewards at the end of training
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.rewards, label="Training Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Reward Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.plot_dir}/training_rewards_trial_{self.locals.get('trial_number', 'final')}.png")
        plt.close()

# Custom callback to report intermediate rewards to Optuna
class TrialEvalCallback(BaseCallback):
    def __init__(self, train_env_vec, val_env_vec, trial, eval_freq=5000, verbose=2, plot_dir="plots"):
        super().__init__(verbose)
        self.train_env_vec = train_env_vec
        self.val_env_vec = val_env_vec
        self.trial = trial
        self.eval_freq = eval_freq
        self.step = 0
        self.train_profits = []
        self.val_profits = []
        self.eval_steps = []
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        self.step += 1
        if self.step % self.eval_freq == 0:
            # Evaluate on training env
            train_profit = evaluate(model=self.model, env_vec=self.train_env_vec, n_episodes=1, deterministic=True, quantstats=False)
            # Evaluate on validation env
            val_profit = evaluate(model=self.model, env_vec=self.val_env_vec, n_episodes=1, deterministic=True, quantstats=False)

            self.train_profits.append(train_profit)
            self.val_profits.append(val_profit)
            self.eval_steps.append(self.step)

            print(f"Step {self.step}: Training Profit = {train_profit:.2f}, Validation Profit = {val_profit:.2f}")

            # Report validation profit to Optuna for pruning
            self.trial.report(val_profit, self.step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True

    def _on_training_end(self):
        # Plot training vs validation profits
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_steps, self.train_profits, label="Training Profit")
        plt.plot(self.eval_steps, self.val_profits, label="Validation Profit")
        plt.xlabel("Timesteps")
        plt.ylabel("Profit")
        plt.title("Training vs Validation Profit Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.plot_dir}/profit_comparison_trial_{self.trial.number}.png")
        plt.close()

# Custom evaluation callback with early stopping
class EarlyStoppingEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=10000, patience=5, min_improvement=0.0, verbose=1, n_eval_episodes=5):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose,
            callback_on_new_best=None  # We’ll handle best model saving manually
        )
        self.patience = patience  # Number of evaluations with no improvement before stopping
        self.min_improvement = min_improvement  # Minimum improvement required to reset patience
        self.best_profit = float('-inf')  # Track best validation profit
        self.no_improvement_count = 0  # Count evaluations without improvement
        self.best_model_path = "best_model_temp"  # Temporary save path for best model

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate on validation set using your evaluate function
            mean_profit = evaluate(
                model=self.model,
                env_vec=self.eval_env,
                n_episodes=self.n_eval_episodes,
                deterministic=True,
                quantstats=False
            )
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Validation Profit = {mean_profit:.2f}, Best Profit = {self.best_profit:.2f}")

            # Check for improvement
            if mean_profit > self.best_profit + self.min_improvement:
                self.best_profit = mean_profit
                self.no_improvement_count = 0
                # Save the best model
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(f"New best model saved with profit {mean_profit:.2f} at step {self.n_calls}")
            else:
                self.no_improvement_count += 1
                if self.verbose > 0:
                    print(f"No improvement ({self.no_improvement_count}/{self.patience})")

            # Early stopping condition
            if self.no_improvement_count >= self.patience:
                print(f"Early stopping triggered after {self.no_improvement_count} evaluations without improvement.")
                return False  # Stop training

        return True  # Continue training

# policy_kwargs = {
#     "initial_balance": train_env.balance_initial,
#     "point_scale": train_env.cf.symbol(env.assets[0], "point"),
#     "embed_dim": 32,
#     "net_arch": dict(pi=[64, 64], vf=[64, 64]),
#     "impact_field_idx": 5,  # e.g., impact_code is the 6th field (0-based index)
#     "usd_currency_id": 4   # e.g., USD is ID 4 in currency_map
# }

# Objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 64, 8192, step=64)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-3, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.05, 0.5)
    n_epochs = trial.suggest_int("n_epochs", 5, 100, step=5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  # Added weight decay

    # Ensure batch_size divides n_steps * env_vec.num_envs evenly (assuming num_envs=1 here)
    if (n_steps * train_env_vec.num_envs) % batch_size != 0:
        # Adjust batch_size to the nearest valid value
        valid_batch_sizes = [b for b in range(32, 513, 32) if (n_steps * train_env_vec.num_envs) % b == 0]
        if valid_batch_sizes:
            batch_size = min(valid_batch_sizes, key=lambda x: abs(x - batch_size))
        else:
            # If no valid batch size, skip this trial
            raise optuna.TrialPruned()

    # Train PPO model on training set
    model = PPO(
        policy=CustomMultiInputPolicy,
        env=train_env_vec,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        max_grad_norm=max_grad_norm,
        ent_coef=ent_coef,  # Entropy coefficient for exploration
        verbose=2,
        # 0: No output during training,
        # 1: Prints basic training progress,
        # 2: More detailed output (Additional details like optimization steps, loss values (e.g., policy loss, value loss), and learning rate updates.)
        device="cpu",
    )
    model.optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Combine callbacks
    reward_logger = RewardLogger(log_freq=1000)
    eval_callback = TrialEvalCallback(train_env_vec, val_env_vec, trial, eval_freq=5000)
    callback = [reward_logger, eval_callback]

    model.learn(total_timesteps=100000, callback=callback)

    # Final evaluation
    mean_profit = evaluate(
        model=model,
        env_vec=val_env_vec,
        n_episodes=10,  # Number of evaluation episodes
        deterministic=True,  # Use deterministic actions for evaluation
        quantstats=False,  # Disable QuantStats for faster evaluation
    )

    print(f"Best Validation Average Profit: {mean_profit:.2f}")

    # Return the metric to maximize (mean profit)
    return mean_profit  # Maximize profit

# Main execution with logging
if __name__ == "__main__":
    # Initialize logger
    logger = Logger(filename=f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", echo_to_console=True)
    logger.start()

    try:
        # Specify the SQLite database file
        db_path = 'optuna_study.db'

        # Run optimization
        study = optuna.create_study(
            study_name='OHLC_EconomicCalender_ppo_study_1',
            storage=f'sqlite:///{db_path}',
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,  # Number of trials before pruning starts
                n_warmup_steps=2000,  # Steps before pruning evaluation begins
                interval_steps=1000,  # Pruning check interval
            ),
            load_if_exists=True,
        )
        n_trials = 50  # Number of trials to run (adjust as needed)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)  # 2+ epochs, reasonable for trials

        # Print the best parameters and result
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (Mean Profit): {trial.value:.2f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train final model with best parameters
        best_params = trial.params
        final_model = PPO(
            policy=CustomMultiInputPolicy,
            env=train_env_vec,
            policy_kwargs=policy_kwargs,
            verbose=2,
            n_steps=best_params["n_steps"],
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            n_epochs=best_params["n_epochs"],
            gamma=best_params["gamma"],
            gae_lambda=best_params["gae_lambda"],
            clip_range=best_params["clip_range"],
            max_grad_norm=best_params["max_grad_norm"],
            ent_coef=best_params["ent_coef"],
            device="cpu",
        )
        final_model.optimizer = torch.optim.Adam(final_model.policy.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
        final_model.learn(total_timesteps=500000)  # 10 epochs for thorough training

        # Train final model with reward logging
        reward_logger = RewardLogger(log_freq=1000)
        early_stopping_callback = EarlyStoppingEvalCallback(
            eval_env=val_env_vec,
            eval_freq=10000,  # Evaluate every 10,000 steps
            patience=5,       # Stop after 5 evaluations with no improvement
            min_improvement=0.0,  # Require any improvement (adjust as needed)
            verbose=1,
            n_eval_episodes=5  # Number of episodes for validation evaluation
        )
        callbacks = [reward_logger, early_stopping_callback]

        # Train with early stopping
        print("Training final model with early stopping...")
        final_model.learn(total_timesteps=500000, callback=callbacks)

        # Load the best model if early stopping occurred
        # Without loading best_model_temp.zip, ppo_xauusd_optimized_trial.zip
        # might not be the best model. If training continues past the peak
        # validation profit and performance degrades (overfitting), the final
        # saved model could be suboptimal.
        if os.path.exists(f"{early_stopping_callback.best_model_path}.zip"):
            final_model = PPO.load(early_stopping_callback.best_model_path, env=train_env_vec, device="cpu")
            print(f"Loaded best model from {early_stopping_callback.best_model_path}")
        else:
            print("No best model saved; using final trained model.")
            # Final evaluation on validation set
            mean_profit = evaluate(
                model=final_model,
                env_vec=val_env_vec,
                n_episodes=20,  # Number of evaluation episodes
                deterministic=True,  # Use deterministic actions for evaluation
                quantstats=False,  # Disable QuantStats for faster evaluation
            )
            print(f"Best Validation Average Profit: {mean_profit:.2f}")

        # Save the final model
        final_model.save(f"ppo_xauusd_optimized")

        final_model = PPO.load("ppo_xauusd_optimized.zip", env=train_env_vec, device="cpu")

        # Evaluate on test data with QuantStats
        print("\nEvaluating Final Model on Test Data:")
        test_avg_profit = evaluate(
            final_model, test_env_vec, n_episodes=1, deterministic=True, quantstats=True)
        print(f"Test Average Profit: {test_avg_profit:.2f}")

        # Clean up
        train_env_vec.close()
        val_env_vec.close()
        test_env_vec.close()

    finally:
        # Stop logging
        logger.stop()