import torch
import torch.nn as nn
import math
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, initial_balance=100000.0, point_scale=1000.0, embed_dim=32,
                 impact_field_idx=None, usd_currency_id=None):
        n_assets = (observation_space.spaces["portfolio_data"].shape[0] - 3) // 2
        ohlc_dim = observation_space.spaces["ohlc_data"].shape[0]  # 24
        max_events = observation_space.spaces["event_ids"].shape[0]  # 8
        economic_numeric_dim = observation_space.spaces["economic_numeric"].shape[0]  # 48
        portfolio_dim = observation_space.spaces["portfolio_data"].shape[0]  # 5
        hour_dim = observation_space.spaces["hour_features"].shape[0]  # 2

        # Increase embedding size for richer representations
        self.embed_dim = embed_dim  # Adjustable; larger for more capacity 32, 64, 128 could improve expressiveness
        features_dim = ohlc_dim + embed_dim * 2 + economic_numeric_dim + portfolio_dim + hour_dim  # embed_dim=32, 24 + 32*2 + 48 + 5 + 2 = 143

        self.device = torch.device("cpu")
        print(f"CustomFeaturesExtractor: n_assets={n_assets}, embed_dim={embed_dim}, features_dim={features_dim}, device={self.device}")

        super().__init__(observation_space, features_dim=features_dim)

        # Add gradient clipping parameter
        self.max_grad_norm = 0.1  # Moderate clipping (from Code 1, adjustable)

        # Embeddings for event IDs and currency IDs with smaller initialization
        self.event_embedding = nn.Embedding(num_embeddings=129, embedding_dim=embed_dim).to(self.device)
        self.currency_embedding = nn.Embedding(num_embeddings=6, embedding_dim=embed_dim).to(self.device)
        self.weekday_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embed_dim).to(self.device)

        nn.init.xavier_uniform_(self.event_embedding.weight, gain=0.01)
        nn.init.xavier_uniform_(self.currency_embedding.weight, gain=0.01)
        nn.init.xavier_uniform_(self.weekday_embedding.weight, gain=0.01)

        # Attention with sparsity-aware enhancements
        self.input_norm = nn.LayerNorm(embed_dim, eps=1e-5).to(self.device)  # Normalize before attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, dropout=0.1, batch_first=False  # Higher dropout for sparsity
        ).to(self.device)
        nn.init.xavier_uniform_(self.attention.in_proj_weight, gain=1.0)
        nn.init.constant_(self.attention.in_proj_bias, 0)
        self.attn_norm = nn.LayerNorm(embed_dim, eps=1e-5).to(self.device)  # stricter eps

        # Learned default embedding for no-event cases
        self.no_event_embedding = nn.Parameter(torch.zeros(embed_dim, device=self.device))
        nn.init.xavier_uniform_(self.no_event_embedding.unsqueeze(0), gain=0.01)  # Initialize as a small vector

        # Linear layer to combine OHLC and portfolio data if needed
        self.fc = nn.Linear(ohlc_dim + portfolio_dim, ohlc_dim + portfolio_dim).to(self.device)
        
        # Initialize weights to prevent exploding gradients
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)  # Smaller gain for stability
        nn.init.constant_(self.fc.bias, 0)
        self.fc_norm = nn.LayerNorm(ohlc_dim + portfolio_dim, eps=1e-5).to(self.device)

        # Running statistics for portfolio_data (shape: [5])
        self.portfolio_mean = torch.zeros(portfolio_dim, device=self.device)
        self.portfolio_var = torch.ones(portfolio_dim, device=self.device)  # Variance, not std
        self.portfolio_count = 0
        self.momentum = 0.1  # Exponential moving average factor

        # Use passed initial_balance and point_scale
        self.initial_balance = initial_balance  # Received from policy_kwargs
        self.point_scale = point_scale  # Received from policy_kwargs
        print(f"CustomFeaturesExtractor: initial_balance={self.initial_balance}, point_scale={self.point_scale}")

        # Configurable parameters
        self.impact_field_idx = impact_field_idx  # Index of impact_code in economic_numeric (0-based)
        self.usd_currency_id = usd_currency_id    # Currency ID for USD in currency_ids
        self.fields_per_event = economic_numeric_dim // max_events  # 6

        # Validation
        if self.impact_field_idx is not None and self.impact_field_idx >= self.fields_per_event:
            raise ValueError(f"impact_field_idx={self.impact_field_idx} exceeds fields per event ({self.fields_per_event})")
        if self.usd_currency_id is not None and self.usd_currency_id >= self.currency_embedding.num_embeddings:
            raise ValueError(f"usd_currency_id={self.usd_currency_id} exceeds currency embedding size ({self.currency_embedding.num_embeddings})")

    def _check_nan(self, tensor, name, step=None):
        if torch.any(torch.isnan(tensor)):
            msg = f"NaN detected in {name}"
            if step:
                msg += f" at step {step}"
            raise ValueError(f"{msg}: {tensor}")
        if torch.any(torch.isinf(tensor)):
            msg = f"Inf detected in {name}"
            if step:
                msg += f" at step {step}"
            raise ValueError(f"{msg}: {tensor}")

    def _safe_embedding(self, embedding_layer, indices, name):
        emb = embedding_layer(indices)
        emb = torch.clamp(emb, min=-5, max=5)  # immediate clamp
        if torch.any(torch.isnan(emb)):
            print(f"Warning: NaN in {name}, reinitializing")
            nn.init.xavier_uniform_(embedding_layer.weight, gain=0.01)
            emb = embedding_layer(indices)
            emb = torch.clamp(emb, min=-5, max=5)
        self._check_nan(emb, name)
        return emb

    def update_running_stats(self, portfolio_data):
        """Update running mean and variance for portfolio_data using Welford’s method."""
        self.portfolio_count += 1
        batch_mean = portfolio_data.mean(dim=0)  # Mean across batch (if batch > 1)
        batch_var = portfolio_data.var(dim=0, unbiased=False)  # Variance across batch

        # Exponential moving average
        if self.portfolio_count == 1:
            self.portfolio_mean = batch_mean
            self.portfolio_var = batch_var
        else:
            delta = batch_mean - self.portfolio_mean
            self.portfolio_mean += self.momentum * delta
            delta2 = batch_mean - self.portfolio_mean
            self.portfolio_var = (1 - self.momentum) * (self.portfolio_var + self.momentum * delta * delta2)

    def normalize_portfolio(self, portfolio_data):
        """Normalize portfolio_data using running statistics."""
        # Initial static scaling
        portfolio_data_scaled = portfolio_data.clone()
        if portfolio_data_scaled.dim() == 1:
            portfolio_data_scaled = portfolio_data_scaled.unsqueeze(0)  # Add batch dimension
        portfolio_data_scaled[:, 0] /= self.initial_balance  # balance
        portfolio_data_scaled[:, 1] /= self.initial_balance  # total_equity
        portfolio_data_scaled[:, 2] /= 100.0  # max_draw_down_pct
        # current_holding (index 3) remains unscaled (0 or 1)
        portfolio_data_scaled[:, 4] /= self.point_scale  # current_draw_downs

        # Update running stats
        self.update_running_stats(portfolio_data_scaled)

        # Normalize with running mean and std
        std = torch.sqrt(self.portfolio_var.clamp(min=1e-6))  # Avoid division by zero
        normalized = (portfolio_data_scaled - self.portfolio_mean) / std
        return torch.clamp(normalized, min=-5, max=5)  # Prevent extreme values

    def forward(self, obs):
        # Scale inputs dynamically （z-score)
        ohlc_data = obs["ohlc_data"].to(self.device)  # (batch, 24)
        event_ids = obs["event_ids"].to(self.device, dtype=torch.long)  # (batch, 8)
        currency_ids = obs["currency_ids"].to(self.device, dtype=torch.long)  # (batch, 8)
        economic_numeric = obs["economic_numeric"].to(self.device)  # (batch, 48)
        portfolio_data = obs["portfolio_data"].to(self.device)  # (batch, 5)
        weekday = obs["weekday"].to(self.device, dtype=torch.long)  # (batch, 1)
        hour_features = obs["hour_features"].to(self.device)  # (batch, 2)

        # Dynamically normalize portfolio_data
        portfolio_data_scaled = self.normalize_portfolio(portfolio_data)
        
        # Debugging input shapes
        # print(f"ohlc_data shape: {ohlc_data.shape}")
        # print(f"event_ids shape: {event_ids.shape}")
        # print(f"currency_ids shape: {currency_ids.shape}")
        # print(f"portfolio_data_scaled: mean={portfolio_data_scaled.mean()}, std={portfolio_data_scaled.std()}")

        self._check_nan(ohlc_data, "ohlc_data")
        self._check_nan(event_ids, "event_ids")
        self._check_nan(currency_ids, "currency_ids")
        self._check_nan(economic_numeric, "economic_numeric")
        self._check_nan(portfolio_data_scaled, "portfolio_data_scaled")
        self._check_nan(weekday, "weekday")
        self._check_nan(hour_features, "hour_features")

        if event_ids.max() >= self.event_embedding.num_embeddings:
            raise ValueError(f"event_ids out of bounds: max={event_ids.max()}, num_embeddings={self.event_embedding.num_embeddings}")
        if currency_ids.max() >= self.currency_embedding.num_embeddings:
            raise ValueError(f"currency_ids out of bounds: max={currency_ids.max()}, num_embeddings={self.currency_embedding.num_embeddings}")

        # Embeddings with NaN protection (keep as (batch, seq_len, embed_dim))
        event_emb = self._safe_embedding(self.event_embedding, event_ids, "event_emb")  # (batch, 8, embed_dim)
        currency_emb = self._safe_embedding(self.currency_embedding, currency_ids, "currency_emb")  # (batch, 8, embed_dim)
        weekday_emb = self._safe_embedding(self.weekday_embedding, weekday, "weekday_emb")  # (batch, 1, embed_dim)

        # Combine event and currency embeddings (e.g., element-wise addition)
        combined_emb = event_emb + currency_emb  # (batch, 8, embed_dim)
        combined_emb = torch.tanh(combined_emb) * 2.0  # Stable scaling
        self._check_nan(combined_emb, "combined_emb")

        # Identify valid events
        valid_mask = (event_ids != 0)  # (batch, 8)
        batch_size = event_ids.shape[0]

        # Count valid events per batch item
        num_valid_events = valid_mask.sum(dim=1)  # (batch,)
        has_events = num_valid_events > 0  # (batch,)

        # Initialize output tensor
        event_summary = torch.zeros(batch_size, self.embed_dim, device=self.device)

        if has_events.any():
            # Select batch items with valid events
            batch_indices = torch.where(has_events)[0]  # Indices of batches with events
            valid_combined_emb_subset = [combined_emb[b][valid_mask[b]] for b in batch_indices]
            valid_indices_subset = [valid_mask[b].nonzero(as_tuple=False).squeeze(-1) for b in batch_indices]
            
            # Extract valid economic numeric subsets
            valid_economic_numeric_subset = []
            for b, valid_indices in zip(batch_indices, valid_indices_subset):
                # Get indices of valid events
                slices = []
                for idx in valid_indices:
                    # Compute scalar start and end indices
                    start = idx * self.fields_per_event
                    end = (idx + 1) * self.fields_per_event
                    # Extract the slice for this event
                    slice_tensor = economic_numeric[b][start:end]
                    slices.append(slice_tensor)
                if slices:
                    # Stack slices into a tensor of shape [num_valid_events, fields_per_event]
                    valid_economic_numeric_subset.append(torch.stack(slices, dim=0))
                else:
                    # Handle batches with no valid events
                    valid_economic_numeric_subset.append(torch.empty(0, self.fields_per_event, device=self.device))

            # Pad the subset
            valid_combined_emb_padded = torch.nn.utils.rnn.pad_sequence(
                valid_combined_emb_subset, batch_first=False, padding_value=0.0
            )  # (max_valid_events_subset, batch_subset, embed_dim)
            valid_economic_numeric_padded = torch.nn.utils.rnn.pad_sequence(
                valid_economic_numeric_subset, batch_first=True, padding_value=0.0
            )  # (batch_subset, max_valid_events_subset, fields_per_event)

            # # Debugging shapes
            # for i, tensor in enumerate(valid_economic_numeric_subset):
            #     print(f"Batch {batch_indices[i]}: valid_economic_numeric_subset shape={tensor.shape}")

            # Create key padding mask for subset
            num_valid_events_subset = num_valid_events[has_events]
            max_valid_events_subset = num_valid_events_subset.max()
            valid_key_padding_mask = torch.zeros(
                len(batch_indices), max_valid_events_subset, dtype=torch.bool, device=self.device
            )
            for i, n in enumerate(num_valid_events_subset):
                valid_key_padding_mask[i, n:] = True

            # Normalize inputs (no custom scaling)
            valid_combined_emb_scaled = self.input_norm(valid_combined_emb_padded)
            valid_combined_emb_scaled = torch.clamp(valid_combined_emb_scaled, min=-0.05, max=0.05)
            self._check_nan(valid_combined_emb_scaled, "valid_combined_emb_scaled")

            # Apply attention
            attn_output, attn_weights = self.attention(
                query=valid_combined_emb_scaled,
                key=valid_combined_emb_scaled,
                value=valid_combined_emb_padded,
                key_padding_mask=valid_key_padding_mask,
                need_weights=True
            )
            self._check_nan(attn_output, "attn_output")
            self._check_nan(attn_weights, "attn_weights")

            # Post-process attention output
            attn_output = attn_output.transpose(0, 1)  # (batch_subset, max_valid_events_subset, embed_dim)
            attn_output = self.attn_norm(attn_output)
            attn_output = torch.clamp(attn_output, min=-5, max=5)

            # Aggregate based on configuration
            if self.impact_field_idx is not None and self.usd_currency_id is not None:
                # impact_weights = impact_codes + 1 (shifts 0,1,2 to 1,2,3).
                impact_codes = valid_economic_numeric_padded[:, :, self.impact_field_idx]
                impact_weights = (impact_codes + 1).float() # (batch, 8), shift to 1,2,3
                # currency_weights = 1.5 for USD (4), 1.0 otherwise.
                currency_weights = torch.ones_like(impact_weights, device=self.device)
                for i, b in enumerate(batch_indices):
                    valid_idx = valid_indices_subset[i]
                    valid_currency_ids = currency_ids[b, valid_idx]
                    currency_weights[i, :len(valid_idx)][valid_currency_ids == self.usd_currency_id] = 1.5
                weights = (~valid_key_padding_mask).float() * impact_weights * currency_weights
                weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                weights = weights / weights_sum
                event_summary_subset = torch.sum(attn_output * weights.unsqueeze(-1), dim=1)
            else:
                valid_events = (~valid_key_padding_mask).float().unsqueeze(-1)
                event_sum = (attn_output * valid_events).sum(dim=1)
                event_count = valid_events.sum(dim=1).clamp(min=1e-6)
                event_summary_subset = event_sum / event_count

            # Assign results to corresponding batch items
            event_summary[has_events] = event_summary_subset

        # Assign no-event embedding to batches without events
        event_summary[~has_events] = self.no_event_embedding
        self._check_nan(event_summary, "event_summary")

        # Combine OHLC and portfolio
        ohlc_portfolio = torch.cat([ohlc_data, portfolio_data_scaled], dim=1)
        ohlc_portfolio = self.fc(ohlc_portfolio)
        ohlc_portfolio = self.fc_norm(ohlc_portfolio)
        ohlc_portfolio = torch.clamp(ohlc_portfolio, min=-1000, max=1000)
        self._check_nan(ohlc_portfolio, "ohlc_portfolio")

        # Final feature concatenation
        weekday_emb = weekday_emb.squeeze(1)
        features = torch.cat([ohlc_portfolio, event_summary, weekday_emb, economic_numeric, hour_features], dim=1)
        self._check_nan(features, "features")

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        return features
    
class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        device = torch.device("cpu")
        # print(f"CustomMultiInputPolicy using device: {device}")

        # Extract action space bounds and move them to the selected device
        self.action_space_low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self.action_space_high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        action_dim = action_space.shape[0]  # Number of assets

        # Extract features_extractor_kwargs from kwargs, using observation_space from self
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {
            "initial_balance": kwargs.pop("initial_balance", 100000.0),
            "point_scale": kwargs.pop("point_scale", 1000.0),
            "embed_dim": kwargs.pop("embed_dim", 32),
            "impact_field_idx": kwargs.pop("impact_field_idx", None),
            "usd_currency_id": kwargs.pop("usd_currency_id", None),
        })

        # Call parent class with filtered kwargs
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=kwargs.pop("net_arch", dict(pi=[64, 64], vf=[64, 64])),  # Use from kwargs or default
            use_sde=kwargs.pop("use_sde", False),  # Ensure use_sde is passed if present
            *args,
            **kwargs
        )

        self.mlp_extractor = MlpExtractor(
            self.features_extractor.features_dim, net_arch=self.net_arch,
            activation_fn=nn.ReLU, device=device
        ).to(device)

        self.action_net = nn.Linear(64, action_dim).to(device)
        self.log_std = nn.Parameter(torch.ones(action_dim, device=device) * 0.5)  # Std ≈ 1.65 (less aggressive)
        self.value_net = nn.Linear(64, 1).to(device)

        # Initialize weights to prevent exploding gradients
        for layer in [self.action_net, self.value_net]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(layer.bias, 0)

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.num_timesteps = 0
        self.max_grad_norm = 0.1  # Fix #5: Tighter clipping

    def _check_nan(self, tensor, name, step=None):
        if torch.any(torch.isnan(tensor)):
            msg = f"NaN detected in {name}"
            if step:
                msg += f" at step {step}"
            raise ValueError(f"{msg}: {tensor}")
        if torch.any(torch.isinf(tensor)):
            msg = f"Inf detected in {name}"
            if step:
                msg += f" at step {step}"
            raise ValueError(f"{msg}: {tensor}")

    def forward(self, obs, deterministic=False):
        # Increment timestep on each forward pass
        self.num_timesteps += 1

        # Extract features 
        features = self.extract_features(obs)
        self._check_nan(features, "features", self.num_timesteps)

        latent_pi, latent_vf = self.mlp_extractor(features)
        self._check_nan(latent_pi, "latent_pi", self.num_timesteps)
        self._check_nan(latent_vf, "latent_vf", self.num_timesteps)

        # Get mean and log_std from action_net
        mean_actions = self.action_net(latent_pi)  # (batch_size, action_dim)
        self._check_nan(mean_actions, "mean_actions_before_clamp", self.num_timesteps)
        mean_actions = torch.clamp(mean_actions, -5.0, 5.0)  # Limit mean to [-10, 10]
        self._check_nan(mean_actions, "mean_actions_after_clamp", self.num_timesteps)
        
        log_std = self.log_std.expand_as(mean_actions)  # Match batch size
        log_std = torch.clamp(log_std, min=-2, max=2)  # Stabilize log_std
        self._check_nan(log_std, "log_std", self.num_timesteps)

        # Create a fresh distribution instance with current parameters
        distribution = SquashedDiagGaussianDistribution(self.action_dist.action_dim)

        # Sample actions or get deterministic actions
        actions = distribution.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        self._check_nan(actions, "actions", self.num_timesteps)

        # Map from [-1, 1] to [0, 3]
        squashed_actions = self._squash_to_range(actions, self.action_space_low, self.action_space_high)
        self._check_nan(squashed_actions, "squashed_actions", self.num_timesteps)

        # Compute log probabilities of the unsquashed actions
        log_prob = distribution.log_prob(actions)  # Use log_prob on the unsquashed actions
        self._check_nan(log_prob, "log_prob", self.num_timesteps)

        # Value prediction
        values = self.value_net(latent_vf)
        self._check_nan(values, "values", self.num_timesteps)

        # print(f"Step {self.num_timesteps}: Mean={mean_actions.mean().item()}, Log Std={log_std.mean().item()}, Unsquashed={actions.tolist()}, Squashed={squashed_actions.tolist()}, Deterministic={deterministic}")        
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm) 
        # if self.num_timesteps % 1000 == 0:
        #     print(f"Step {self.num_timesteps}, Actions: {squashed_actions}, Mean: {mean_actions.mean()}, Log Std: {log_std.mean()}")
        #     # if log_std.mean too low (e.g., near -20), increase exploration by adjusting PPO’s clip_range or policy entropy regularization (e.g., ent_coef).

        return squashed_actions, values, log_prob

    def _squash_to_range(self, actions, low, high):
        """Scale squashed actions from [-1, 1] to [low, high]."""
        result = (actions + 1) * (high - low) / 2 + low
        self._check_nan(result, "squashed_range_output")
        return result

    def _predict(self, observation, deterministic=False):
        # Convert observation to tensor if needed
        obs_tensor = self._process_observation(observation)
        with torch.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        # Convert back to numpy and reshape
        return actions

    def _process_observation(self, obs):
        # Handle dictionary observation from env
        if isinstance(obs, dict):
            # print({k: v.shape for k, v in obs.items()})
            # Remove extra dimensions from DummyVecEnv and ensure (batch, feature)
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in obs.items()}
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def extract_features(self, obs):
        return self.features_extractor(obs)

    def predict_values(self, obs):
        features = self.extract_features(obs)
        self._check_nan(features, "features_in_predict_values")
        _, latent_vf = self.mlp_extractor(features)
        self._check_nan(latent_vf, "latent_vf_in_predict_values")
        values = self.value_net(latent_vf)
        self._check_nan(values, "values_in_predict_values")
        return values

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for training (used by PPO)."""
        features = self.extract_features(obs)
        self._check_nan(features, "features_in_evaluate", self.num_timesteps)
        latent_pi, latent_vf = self.mlp_extractor(features)
        self._check_nan(latent_pi, "latent_pi_in_evaluate", self.num_timesteps)
        self._check_nan(latent_vf, "latent_vf_in_evaluate", self.num_timesteps)

        mean_actions = self.action_net(latent_pi)
        self._check_nan(mean_actions, "mean_actions_in_evaluate", self.num_timesteps)

        mean_actions = torch.clamp(mean_actions, -5.0, 5.0)
        log_std = self.log_std.expand_as(mean_actions)
        log_std = torch.clamp(log_std, min=-2, max=2)
        self._check_nan(log_std, "log_std_in_evaluate", self.num_timesteps)

        # Ensure actions are float and match expected shape
        actions = actions.to(self.device, dtype=torch.float32)
        # print(f"evaluate_actions: actions shape={actions.shape}, dtype={actions.dtype}, values={actions[:5]}")

        # Create a fresh distribution instance
        distribution = SquashedDiagGaussianDistribution(self.action_dist.action_dim)
        distribution.proba_distribution(mean_actions, log_std)  # Set mean and log_std

        # Unsquash the actions back to [-1, 1] for log_prob calculation
        unsquashed_actions = 2 * (actions - self.action_space_low) / (self.action_space_high - self.action_space_low) - 1
        self._check_nan(unsquashed_actions, "unsquashed_actions_in_evaluate", self.num_timesteps)

        # Compute log_prob and entropy
        log_prob = distribution.log_prob(unsquashed_actions)  # Use log_prob on unsquashed actions
        self._check_nan(log_prob, "log_prob_in_evaluate", self.num_timesteps)

        entropy = distribution.entropy()  # Entropy doesn’t need actions
        if entropy is None:
          # Fallback: Compute entropy manually if None
          std = torch.exp(log_std)
          entropy = 0.5 * torch.sum(torch.log(2 * np.pi * std**2) + 1, dim=-1)
          # print(f"Entropy was None, computed manually: {entropy.shape}")
        self._check_nan(entropy, "entropy_in_evaluate", self.num_timesteps)

        values = self.value_net(latent_vf)
        self._check_nan(values, "values_in_evaluate", self.num_timesteps)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        return values, log_prob, entropy