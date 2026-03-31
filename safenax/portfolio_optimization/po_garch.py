from typing import Dict, Tuple
from enum import Enum
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments import spaces
from flax import struct
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from argparse import ArgumentParser


class BinanceFeeTier(Enum):
    OFF = 0.0
    REGULAR = 0.001
    VIP_1 = 0.001
    VIP_2 = 0.001
    VIP_3 = 0.0006
    VIP_4 = 0.00052
    VIP_5 = 0.00031
    VIP_6 = 0.00029
    VIP_7 = 0.00028
    VIP_8 = 0.00025
    VIP_9 = 0.00023


@struct.dataclass
class GARCHParams:
    omega: float  # Constant term in variance equation
    alpha: jax.Array  # ARCH coefficients (length q)
    beta: jax.Array  # GARCH coefficients (length p)
    mu: float  # Mean return
    initial_price: float  # Starting price for the asset


@struct.dataclass
class EnvState:
    step: int
    time: int
    path_id: int  # Which path this environment is using
    prices: jax.Array  # Current prices for all assets
    log_returns: jax.Array  # Current returns for all assets
    volatilities: jax.Array  # Current volatilities for all assets
    holdings: jax.Array  # Current holdings
    values: jax.Array  # Current values
    total_value: float


@struct.dataclass
class EnvParams:
    max_steps: int
    initial_cash: float
    taker_fee: float
    gas_fee: float
    trade_threshold: float
    garch_params: Dict[str, GARCHParams]  # GARCH params for each asset
    all_prices: jax.Array
    all_vols: jax.Array
    all_log_returns: jax.Array


class ObsType(Enum):
    EASY = "easy"
    MARKET = "market"
    FULL = "full"


class CostType(Enum):
    NEGATIVE_RETURN = "negative_return"
    VARIANCE = "variance"


@jax.jit
def _sample_garch(carry, x):
    """
    JIT-compiled step function for jax.lax.scan.

    This function is vectorized to process all assets in parallel for a single time step.

    Args:
        carry: A tuple containing (params, garch_state)
            - params: A GARCHParams pytree where each field is a (num_assets, ...) array.
            - garch_state: A tuple (last_vols, last_returns, last_price)
                - last_vols: (num_assets, p) array of past volatilities
                - last_returns: (num_assets, q) array of past returns
                - last_price: (num_assets,) array of prices from t-1
        x: (num_assets,) array of standard normal random shocks for time t.

    Returns:
        new_carry: The updated carry for time t+1
        y_t: A tuple (new_return, new_vol, new_price) of outputs to stack
            - new_return: (num_assets,) array of returns for time t
            - new_vol: (num_assets,) array of volatilities for time t
            - new_price: (num_assets,) array of prices for time t
    """
    # Unpack carry
    params: GARCHParams = carry[0]
    garch_state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] = carry[1]
    last_vols, last_returns, last_price = garch_state
    noise = x  # (num_assets,)

    # GARCH(p,q) equations (all vectorized)
    # sigma_t^2 = omega + sum(alpha * residuals^2) + sum(beta * vols^2)
    residuals = last_returns - params.mu[:, None]  # (num_assets, q)
    arch_term = (params.alpha * residuals**2).sum(axis=-1)
    garch_term = (params.beta * last_vols**2).sum(axis=-1)
    variance = params.omega + arch_term + garch_term
    new_vol = jnp.sqrt(jnp.maximum(variance, 1e-8))  # Ensure positive variance

    # r_t = mu + sigma_t * epsilon_t
    new_log_return = params.mu + new_vol * noise
    new_price = last_price * jnp.exp(new_log_return)

    # Update GARCH state
    y = (new_log_return, new_vol, new_price)
    new_last_vols = jnp.roll(last_vols, shift=1, axis=-1)
    new_last_vols = new_last_vols.at[:, 0].set(new_vol)
    new_last_returns = jnp.roll(last_returns, shift=1, axis=-1)
    new_last_returns = new_last_returns.at[:, 0].set(new_log_return)
    new_carry = (params, (new_last_vols, new_last_returns, new_price))
    return new_carry, y


class PortfolioOptimizationGARCH(Environment):
    """Portfolio optimization environment with GARCH-simulated asset returns."""

    def __init__(
        self,
        step_size: int = 1,
        num_samples: int = 1_000_000,
        num_paths: int = 1,
        obs_type: ObsType = ObsType.MARKET,
        cost_type: CostType = CostType.NEGATIVE_RETURN,
    ):
        """
        Initialize GARCH portfolio environment.

        Args:
            rng: Random key for generating GARCH paths
            garch_params: Dict mapping asset names to GARCHParams
            step_size: Step size for sampling (if subsampling the data)
            num_samples: Total number of time steps to generate
            num_paths: Number of parallel paths to simulate
            obs_type: Type of observation to return (easy, market, full)
            cost_type: Type of cost to compute (negative return or variance)
        """
        super().__init__()
        self.step_size = step_size
        self.num_samples = num_samples
        self.num_paths = num_paths
        self.obs_type = obs_type
        self.cost_type = cost_type

    def sample_paths(self, rng: jax.Array, params: EnvParams) -> EnvParams:
        """Sample GARCH paths for all assets and store in params."""
        garch_params = params.garch_params
        asset_names = list(garch_params.keys())
        num_assets = len(asset_names)
        # Stack GARCHParams into vectorized arrays for parallel processing
        # Each field becomes (num_assets, ...) shaped
        max_p = max(len(gp.beta) for gp in garch_params.values())
        max_q = max(len(gp.alpha) for gp in garch_params.values())

        # Pad alpha and beta to same length for vectorization
        omega_vec = jnp.array(
            [garch_params[name].omega for name in asset_names], dtype=jnp.float32
        )
        mu_vec = jnp.array(
            [garch_params[name].mu for name in asset_names], dtype=jnp.float32
        )
        initial_price_vec = jnp.array(
            [garch_params[name].initial_price for name in asset_names],
            dtype=jnp.float32,
        )

        alpha_list = []
        beta_list = []
        for name in asset_names:
            gp = garch_params[name]
            # Pad with zeros to max length
            alpha_padded = jnp.pad(gp.alpha, (0, max_q - len(gp.alpha)))
            beta_padded = jnp.pad(gp.beta, (0, max_p - len(gp.beta)))
            alpha_list.append(alpha_padded)
            beta_list.append(beta_padded)

        alpha_vec = jnp.stack(alpha_list, axis=0)  # (num_assets, max_q)
        beta_vec = jnp.stack(beta_list, axis=0)  # (num_assets, max_p)

        self.vec_params = GARCHParams(
            omega=omega_vec,
            alpha=alpha_vec,
            beta=beta_vec,
            mu=mu_vec,
            initial_price=initial_price_vec,
        )

        # Initialize GARCH state with unconditional variance
        alpha_sum = alpha_vec.sum(axis=-1)  # (num_assets,)
        beta_sum = beta_vec.sum(axis=-1)  # (num_assets,)
        denominator = 1 - alpha_sum - beta_sum
        uncond_var = jnp.where(denominator < 1e-6, omega_vec, omega_vec / denominator)
        uncond_vol = jnp.sqrt(jnp.maximum(uncond_var, 1e-8))

        # Initialize state with burn-in using unconditional moments
        initial_vols = jnp.tile(uncond_vol[:, None], (1, max_p))  # (num_assets, p)
        initial_returns = jnp.tile(mu_vec[:, None], (1, max_q))  # (num_assets, q)
        initial_prices = initial_price_vec  # (num_assets,)

        initial_state = (initial_vols, initial_returns, initial_prices)

        # Generate multiple independent paths using vmap
        # Generate all noise: (num_paths, num_samples, num_assets)
        rng_keys = jax.random.split(rng, self.num_paths)
        all_noise = jax.vmap(
            lambda key: jax.random.normal(key, (self.num_samples, num_assets))
        )(rng_keys)

        # Vectorize GARCH simulation over paths
        def simulate_one_path(noise):
            """Run GARCH simulation for one path."""
            _, outputs = jax.lax.scan(
                _sample_garch, (self.vec_params, initial_state), noise
            )
            return outputs  # (returns, volatilities, prices)

        # Run all paths in parallel: (num_paths, num_samples, num_assets)
        log_returns, volatilities, prices = jax.vmap(simulate_one_path)(all_noise)

        return params.replace(
            all_log_returns=log_returns, all_vols=volatilities, all_prices=prices
        )

    @property
    def name(self) -> str:
        return "PO-GARCH"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps=self.num_samples // self.step_size,
            initial_cash=1000.0,
            taker_fee=BinanceFeeTier.OFF.value,
            gas_fee=0.0,
            trade_threshold=0.0,
            garch_params={},
            all_prices=jnp.array([]),
            all_vols=jnp.array([]),
            all_log_returns=jnp.array([]),
        )

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space: portfolio weights (including cash)."""
        num_assets = len(params.garch_params)
        return spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(num_assets + 1,),  # +1 for cash
            dtype=jnp.float32,
        )

    def _obs_space_market(self, params: EnvParams) -> spaces.Box:
        num_assets = len(params.garch_params)
        obs_shape = (num_assets,)
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=obs_shape, dtype=jnp.float32
        )

    def _obs_space_easy(self, params: EnvParams) -> spaces.Box:
        num_assets = len(params.garch_params)
        obs_shape = (num_assets * 2,)
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=obs_shape, dtype=jnp.float32
        )

    def _obs_space_full(self, params: EnvParams) -> spaces.Box:
        num_assets = len(params.garch_params)
        obs_dim = (
            1  # log(total_value)
            + (num_assets + 1)  # normalized holdings (weights)
            + 3  # fees: taker_fee, gas_fee, trade_threshold
            + self.step_size * num_assets * 2  # returns + volatilities
            + num_assets * 4  # GARCH params: mu, omega, alpha, beta
        )
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return observation space based on configured observation type."""
        if self.obs_type == ObsType.EASY:
            return self._obs_space_easy(params)
        elif self.obs_type == ObsType.MARKET:
            return self._obs_space_market(params)
        elif self.obs_type == ObsType.FULL:
            return self._obs_space_full(params)
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")

    def _get_obs_easy(self, state: EnvState, params: EnvParams) -> jax.Array:
        num_assets = len(params.garch_params)
        next_time = state.time + self.step_size
        # Index into the correct path: (num_paths, num_samples, num_assets)
        # Use dynamic_slice for JIT compatibility
        path_vols = jax.lax.dynamic_slice(
            params.all_vols,
            (state.path_id, next_time, 0),
            (1, 1, num_assets),
        )
        next_vol = path_vols.squeeze()  # (num_assets,)
        mu = self.vec_params.mu  # (num_assets,)

        obs = jnp.concatenate([next_vol.flatten(), mu.flatten()])
        return obs

    def _get_obs_market(self, state: EnvState, params: EnvParams) -> jax.Array:
        num_assets = len(params.garch_params)
        # Extract recent returns and volatilities from pre-generated path
        start_time_idx = jnp.maximum(0, state.time - self.step_size + 1)

        # Index into correct path using dynamic_slice
        log_returns_window = jax.lax.dynamic_slice(
            params.all_log_returns,
            (state.path_id, start_time_idx, 0),
            (1, self.step_size, num_assets),
        ).squeeze(0)  # Remove path dimension

        step_log_return = log_returns_window.sum(axis=0)

        return step_log_return

    def _get_obs_full(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation from current state."""
        num_assets = len(params.garch_params)
        # Extract recent returns and volatilities from pre-generated path
        start_time_idx = jnp.maximum(0, state.time - self.step_size + 1)

        # Index into correct path using dynamic_slice
        log_returns_window = jax.lax.dynamic_slice(
            params.all_log_returns,
            (state.path_id, start_time_idx, 0),
            (1, self.step_size, num_assets),
        ).squeeze(0)  # (step_size, num_assets)

        vols_window = jax.lax.dynamic_slice(
            params.all_vols,
            (state.path_id, start_time_idx, 0),
            (1, self.step_size, num_assets),
        ).squeeze(0)  # (step_size, num_assets)

        mu = self.vec_params.mu
        omega = self.vec_params.omega
        alpha = self.vec_params.alpha
        beta = self.vec_params.beta

        fees = jnp.array([params.taker_fee, params.gas_fee, params.trade_threshold])

        obs = jnp.concatenate(
            [
                jnp.array([jnp.log(state.total_value)]),
                state.values / state.total_value,  # Normalize holdings by total value
                fees,
                log_returns_window.flatten(),
                vols_window.flatten(),
                mu.flatten(),
                omega.flatten(),
                alpha.flatten(),
                beta.flatten(),
            ]
        )
        return obs

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation based on configured observation type."""
        if self.obs_type == ObsType.EASY:
            return self._get_obs_easy(state, params)
        elif self.obs_type == ObsType.MARKET:
            return self._get_obs_market(state, params)
        elif self.obs_type == ObsType.FULL:
            return self._get_obs_full(state, params)
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check if episode is done."""
        max_steps_reached = state.step >= params.max_steps
        portfolio_bankrupt = state.total_value <= 0
        return jnp.logical_or(max_steps_reached, portfolio_bankrupt)

    def step_env(
        self, key: jax.Array, state: EnvState, action: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Execute one environment step with pre-generated GARCH prices."""
        num_assets = len(params.garch_params)
        time = state.time + self.step_size

        # Index into correct path using dynamic_slice
        path_prices = jax.lax.dynamic_slice(
            params.all_prices, (state.path_id, time, 0), (1, 1, num_assets)
        ).squeeze()  # (num_assets,)

        path_log_returns = jax.lax.dynamic_slice(
            params.all_log_returns, (state.path_id, time, 0), (1, 1, num_assets)
        ).squeeze()

        path_volatilities = jax.lax.dynamic_slice(
            params.all_vols, (state.path_id, time, 0), (1, 1, num_assets)
        ).squeeze()

        prices = jnp.concatenate([jnp.array([1.0]), path_prices])
        log_returns = jnp.concatenate([jnp.array([0.0]), path_log_returns])
        volatilities = jnp.concatenate([jnp.array([0.0]), path_volatilities])

        # Normalize action to portfolio weights
        weights = jax.nn.softmax(action)

        ############### UPDATE PORTFOLIO WITH FEES ###############
        values = state.holdings * prices
        total_value = jnp.sum(values)
        asset_values = values[1:]
        asset_weights = weights[1:]
        new_asset_values_no_fee = total_value * asset_weights
        deltas_no_fee = new_asset_values_no_fee - asset_values
        num_trades = jnp.sum(jnp.abs(deltas_no_fee) > params.trade_threshold)
        gas_cost = params.gas_fee * num_trades

        # Split buy and sell orders
        buy_indices = deltas_no_fee > params.trade_threshold
        sell_indices = deltas_no_fee < -params.trade_threshold
        no_trade_indices = jnp.concatenate(
            [jnp.array([False]), jnp.abs(deltas_no_fee) <= params.trade_threshold]
        )
        buy_weights = jnp.where(buy_indices, asset_weights, 0.0)
        sell_weights = jnp.where(sell_indices, asset_weights, 0.0)
        current_buy_values = jnp.where(buy_indices, asset_values, 0.0)
        current_sell_values = jnp.where(sell_indices, asset_values, 0.0)

        # Calculate new portfolio value after fees
        fee_param = params.taker_fee / (1 - params.taker_fee)
        numerator = (
            total_value
            - gas_cost
            + fee_param * (jnp.sum(current_buy_values) - jnp.sum(current_sell_values))
        )
        denominator = 1 + fee_param * (jnp.sum(buy_weights) - jnp.sum(sell_weights))
        new_total_value = numerator / denominator
        new_values = new_total_value * weights
        adj_new_values = jnp.where(no_trade_indices, values, new_values)
        delta_values = new_values - adj_new_values
        delta_cash = jnp.sum(delta_values)
        adj_new_values = adj_new_values.at[0].add(delta_cash)
        new_holdings = adj_new_values / prices

        reward = jnp.log(new_total_value) - jnp.log(state.total_value)

        if self.cost_type == CostType.VARIANCE:
            cost = jnp.sum((asset_weights**2) * (path_volatilities**2))
        elif self.cost_type == CostType.NEGATIVE_RETURN:
            cost = -reward
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")

        next_state = EnvState(
            step=state.step + 1,
            time=time,
            path_id=state.path_id,
            prices=prices,
            log_returns=log_returns,
            volatilities=volatilities,
            holdings=new_holdings,
            values=adj_new_values,
            total_value=new_total_value,
        )

        obs = self.get_obs(next_state, params)
        done = self.is_terminal(next_state, params)
        info = {"cost": cost}
        return obs, next_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """
        Reset environment and sample from pre-generated GARCH paths.

        Args:
            key: Random key for sampling path and start time
            params: Environment parameters
        """
        num_assets = len(params.garch_params)
        # Sample path randomly
        key, path_key, time_key = jax.random.split(key, 3)
        path_id = jax.random.randint(path_key, (), 0, self.num_paths)

        episode_length = params.max_steps * self.step_size
        max_start = (
            params.all_prices.shape[1] - episode_length
        )  # shape[1] is num_samples
        min_start = self.step_size
        time = jax.random.randint(time_key, (), min_start, max_start)

        # Index into correct path using dynamic_slice
        path_prices = jax.lax.dynamic_slice(
            params.all_prices, (path_id, time, 0), (1, 1, num_assets)
        ).squeeze()

        path_log_returns = jax.lax.dynamic_slice(
            params.all_log_returns, (path_id, time, 0), (1, 1, num_assets)
        ).squeeze()

        path_volatilities = jax.lax.dynamic_slice(
            params.all_vols, (path_id, time, 0), (1, 1, num_assets)
        ).squeeze()

        prices = jnp.concatenate([jnp.array([1.0]), path_prices])
        log_returns = jnp.concatenate([jnp.array([0.0]), path_log_returns])
        volatilities = jnp.concatenate([jnp.array([0.0]), path_volatilities])

        holdings = jnp.zeros(num_assets + 1)
        holdings = holdings.at[0].set(params.initial_cash)
        values = holdings * prices
        state = EnvState(
            step=0,
            time=time,
            path_id=path_id,
            prices=prices,
            log_returns=log_returns,
            volatilities=volatilities,
            holdings=holdings,
            values=values,
            total_value=jnp.sum(values),
        )
        obs = self.get_obs(state, params)
        return obs, state

    def plot_garch(self, params: EnvParams, path_id: int = 0):
        """
        Plot the generated GARCH price paths, returns, and volatilities for all assets.

        Args:
            params: Environment parameters
            path_id: Which path to plot (0 to num_paths-1)
        """
        asset_names = list(params.garch_params.keys())
        path_id = path_id % self.num_paths
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot prices
        for i, name in enumerate(asset_names):
            axes[0].plot(params.all_prices[path_id, :, i], label=f"{name}")
        axes[0].set_title(f"GARCH-Simulated Asset Prices (Path {path_id})")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Price")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Plot returns
        for i, name in enumerate(asset_names):
            axes[1].plot(
                params.all_log_returns[path_id, :, i], label=f"{name}", alpha=0.7
            )
        axes[1].set_title("GARCH Log Returns")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Log Return")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)

        # Plot volatilities
        for i, name in enumerate(asset_names):
            axes[2].plot(params.all_vols[path_id, :, i], label=f"{name}")
        axes[2].set_title("GARCH Conditional Volatilities")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Volatility")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def calibrate_var_params(
        self,
        params: EnvParams,
        safe_asset: str,
        risky_asset: str,
        epsilon: float = 0.05,
    ):
        """
        Estimate VaR threshold (rho) using ALL pre-generated GARCH paths.

        Computes the distribution of cumulative returns over episode length (max_steps)
        and estimates the epsilon-percentile (VaR) for both assets. The recommended rho
        is set between the safe and risky asset VaR values.

        Args:
            safe_asset: Name of the safe asset (e.g., "APPL")
            risky_asset: Name of the risky asset (e.g., "BTC")
            epsilon: Probability level for VaR (e.g., 0.05 for 5% VaR)
            params: Environment parameters (uses default_params if None)

        Returns:
            Tuple of (rho, var_safe, var_risky) - all as floats
        """
        asset_names = list(params.garch_params.keys())
        # Episode length based on environment configuration
        horizon = params.max_steps * self.step_size

        # Get asset indices
        safe_idx = asset_names.index(safe_asset)
        risky_idx = asset_names.index(risky_asset)

        # Get returns for ALL paths
        # Shape: (num_paths, num_samples, num_assets)
        log_returns_safe_all = params.all_log_returns[
            :, :, safe_idx
        ]  # (num_paths, num_samples)
        log_returns_risky_all = params.all_log_returns[
            :, :, risky_idx
        ]  # (num_paths, num_samples)

        # Calculate cumulative returns over episode-length windows for all paths
        def compute_cumulative_returns(log_returns_path):
            """Compute sliding window cumulative returns for one path."""
            num_windows = log_returns_path.shape[0] - horizon + 1

            # Create sliding windows
            cumulative_returns = jnp.array(
                [log_returns_path[i : i + horizon].sum() for i in range(num_windows)]
            )
            return cumulative_returns

        # Vectorize over all paths
        cumulative_returns_safe = jax.vmap(compute_cumulative_returns)(
            log_returns_safe_all
        )
        cumulative_returns_risky = jax.vmap(compute_cumulative_returns)(
            log_returns_risky_all
        )

        # Flatten across all paths to get the full distribution
        # Shape: (num_paths * num_windows,)
        log_returns_safe = cumulative_returns_safe.flatten()
        log_returns_risky = cumulative_returns_risky.flatten()

        # --- Calculate Empirical VaR (epsilon-percentile) ---
        # We want P(return <= rho) <= epsilon
        # So we look for the value at the epsilon-th percentile
        var_safe = jnp.percentile(log_returns_safe, epsilon * 100)
        var_risky = jnp.percentile(log_returns_risky, epsilon * 100)

        print(
            f"\n--- Empirical VaR Results (Episode Length: {params.max_steps} steps, All {self.num_paths} Paths) ---"
        )
        print(
            f"Total samples analyzed: {len(log_returns_safe):,} ({self.num_paths} paths × {len(log_returns_safe) // self.num_paths} windows)"
        )
        print(f"{safe_asset} {epsilon * 100}% VaR (log return): {float(var_safe):.6f}")
        print(
            f"{safe_asset} {epsilon * 100}% VaR (% return): {(jnp.exp(var_safe) - 1) * 100:.4f}%"
        )
        print(
            f"{risky_asset} {epsilon * 100}% VaR (log return): {float(var_risky):.6f}"
        )
        print(
            f"{risky_asset} {epsilon * 100}% VaR (% return): {(jnp.exp(var_risky) - 1) * 100:.4f}%"
        )
        print(f"\nSample statistics (log returns):")
        print(
            f"  {safe_asset} - mean: {float(log_returns_safe.mean()):.6f}, std: {float(log_returns_safe.std()):.6f}"
        )
        print(
            f"  {risky_asset} - mean: {float(log_returns_risky.mean()):.6f}, std: {float(log_returns_risky.std()):.6f}"
        )

        # --- Determine Rho ---
        if var_safe <= var_risky:
            print(
                f"\n⚠ WARNING: {risky_asset} has better (higher) VaR than {safe_asset}."
            )
            print(
                f"  This might indicate that {safe_asset} is actually riskier at the {epsilon * 100}% tail level."
            )

        # Set rho in the middle (conservative choice)
        rho = (var_safe + var_risky) / 2

        print(f"\n{'=' * 60}")
        print(f">>> RECOMMENDED SETTINGS <<<")
        print(f"{'=' * 60}")
        print(f"Episode Length (max_steps)  : {params.max_steps}")
        print(f"Epsilon (Probability Level) : {epsilon}")
        print(f"Rho (VaR Threshold)         : {float(rho):.6f} (log return)")
        print(
            f"Rho (VaR Threshold)         : {(jnp.exp(rho) - 1) * 100:.4f}% (percentage return)"
        )
        print(f"{'=' * 60}\n")

        # Determine plot bounds (use 1st and 99th percentile for better visualization)
        all_returns = jnp.concatenate([log_returns_safe, log_returns_risky])
        lower_bound = float(jnp.percentile(all_returns, 1))
        upper_bound = float(jnp.percentile(all_returns, 99))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Histograms with VaR lines
        axes[0].hist(
            log_returns_safe,
            range=(lower_bound, upper_bound),
            bins=100,
            alpha=0.5,
            label=f"{safe_asset} (Safe)",
            density=True,
            color="green",
        )
        axes[0].hist(
            log_returns_risky,
            range=(lower_bound, upper_bound),
            bins=100,
            alpha=0.5,
            label=f"{risky_asset} (Risky)",
            density=True,
            color="red",
        )
        axes[0].axvline(
            float(rho),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Rho (Threshold): {float(rho):.4f}",
        )
        axes[0].axvline(
            float(var_safe),
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"{safe_asset} VaR: {float(var_safe):.4f}",
        )
        axes[0].axvline(
            float(var_risky),
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"{risky_asset} VaR: {float(var_risky):.4f}",
        )
        axes[0].set_title(
            f"Episode Cumulative Returns ({params.max_steps} steps)\n({self.num_paths:,} Paths, {epsilon * 100}% VaR)"
        )
        axes[0].set_xlabel("Cumulative Return")
        axes[0].set_ylabel("Density")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # Right plot: Left tail zoom (for VaR visualization)
        tail_upper = float(jnp.percentile(all_returns, 10))  # Focus on left tail
        axes[1].hist(
            log_returns_safe,
            range=(lower_bound, tail_upper),
            bins=50,
            alpha=0.5,
            label=f"{safe_asset}",
            density=True,
            color="green",
        )
        axes[1].hist(
            log_returns_risky,
            range=(lower_bound, tail_upper),
            bins=50,
            alpha=0.5,
            label=f"{risky_asset}",
            density=True,
            color="red",
        )
        axes[1].axvline(
            float(rho),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Rho: {float(rho):.4f}",
        )
        axes[1].axvline(
            float(var_safe),
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"{safe_asset} VaR",
        )
        axes[1].axvline(
            float(var_risky),
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"{risky_asset} VaR",
        )
        axes[1].set_title(f"Left Tail (VaR Region)")
        axes[1].set_xlabel("Cumulative Return")
        axes[1].set_ylabel("Density")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return float(rho), float(var_safe), float(var_risky)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Test portfolio optimization environment with different policies"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "hold_btc", "hold_appl", "hold_cash"],
        help="Policy to use: random, hold_btc, hold_appl, or hold_cash",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    garch_params = {
        "BTC": GARCHParams(
            mu=5e-3,
            omega=1e-4,
            alpha=jnp.array([0.165]),
            beta=jnp.array([0.8]),
            initial_price=1.0,
        ),
        "APPL": GARCHParams(
            mu=3e-3,
            omega=1e-5,
            alpha=jnp.array([0.15]),
            beta=jnp.array([0.5]),
            initial_price=1.0,
        ),
    }

    env = PortfolioOptimizationGARCH(
        num_paths=10_000,
        num_samples=1000,
    )
    params = env.default_params.replace(garch_params=garch_params)
    params = env.sample_paths(rng, params)
    asset_names = list(garch_params.keys())
    num_assets = len(asset_names)

    # Plot first path
    env.plot_garch(params, path_id=1)

    # Run a full episode with random actions using JAX scan (fast!)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, params)

    max_steps = env.default_params.max_steps
    appl_idx = asset_names.index("APPL") + 1
    btc_idx = asset_names.index("BTC") + 1

    # Generate actions based on policy
    rng, action_rng = jax.random.split(rng)

    if args.policy == "random":
        # Random actions (unnormalized)
        action_keys = jax.random.split(action_rng, max_steps)
        all_actions = jax.random.normal(action_keys[0], (max_steps, num_assets + 1))
    elif args.policy == "hold_cash":
        # Hold 100% cash: [large, small, small]
        all_actions = jnp.tile(jnp.array([10.0, -10.0, -10.0]), (max_steps, 1))
    elif args.policy == "hold_btc":
        # Hold 100% BTC: [small, large, small]
        all_actions = jnp.tile(jnp.array([-10.0, -10.0, -10.0]), (max_steps, 1))
        all_actions = all_actions.at[:, btc_idx].set(10.0)
    elif args.policy == "hold_appl":
        # Hold 100% APPL: [small, small, large]
        all_actions = jnp.tile(jnp.array([-10.0, -10.0, -10.0]), (max_steps, 1))
        all_actions = all_actions.at[:, appl_idx].set(10.0)

    print(f"\nRunning {args.policy} policy for {max_steps} steps...")

    # Define step function for scan
    def scan_step(carry, action):
        state, rng = carry
        rng, step_rng = jax.random.split(rng)
        obs, next_state, reward, done, info = env.step_env(
            step_rng, state, action, params
        )
        weights = jax.nn.softmax(action)

        # Return updated carry and outputs to collect
        outputs = {
            "reward": reward,
            "portfolio_value": next_state.total_value,
            "weights": weights,
            "done": done,
        }
        return (next_state, rng), outputs

    # Run entire episode with scan (JIT-compiled, runs on GPU/TPU)
    (final_state, _), path = jax.lax.scan(scan_step, (state, rng), all_actions)

    # Extract results
    rewards = path["reward"]
    portfolio_values = jnp.concatenate(
        [jnp.array([state.total_value]), path["portfolio_value"]]
    )
    cash_weights = path["weights"][:, 0]
    btc_weights = path["weights"][:, btc_idx]
    appl_weights = path["weights"][:, appl_idx]

    # Calculate cumulative return
    cumulative_rewards = jnp.cumsum(rewards)
    total_log_return = float(cumulative_rewards[-1])
    final_value = float(portfolio_values[-1])
    initial_value = float(portfolio_values[0])

    print(f"\n{'=' * 60}")
    print(f"Episode Summary ({args.policy} policy):")
    print(f"{'=' * 60}")
    print(f"Steps completed: {max_steps}")
    print(f"Initial portfolio value: ${initial_value:.2f}")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return (log): {total_log_return:.6f}")
    print(f"Total return (percentage): {(final_value / initial_value - 1) * 100:.2f}%")
    print(f"Average reward per step: {float(jnp.mean(rewards)):.6f}")
    print(f"Reward std dev: {float(jnp.std(rewards)):.6f}")
    print(
        f"Sharpe ratio (approx): {float(jnp.mean(rewards) / (jnp.std(rewards) + 1e-8)):.4f}"
    )
    print(f"{'=' * 60}\n")

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Rewards over time
    axes[0].plot(rewards, alpha=0.7, linewidth=1)
    axes[0].set_title(f"Rewards per Step - {args.policy} policy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative returns
    axes[1].plot(cumulative_rewards, color="green", linewidth=2)
    axes[1].set_title("Cumulative Log Returns")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative Log Return")
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Portfolio value over time
    axes[2].plot(portfolio_values, color="blue", linewidth=2)
    axes[2].set_title("Portfolio Value Over Time")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Portfolio Value ($)")
    axes[2].axhline(
        y=initial_value, color="k", linestyle="--", linewidth=0.5, label="Initial Value"
    )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Portfolio allocation over time
    axes[3].plot(cash_weights, label="Cash", alpha=0.7)
    axes[3].plot(btc_weights, label="BTC", alpha=0.7)
    axes[3].plot(appl_weights, label="APPL", alpha=0.7)
    axes[3].set_title("Portfolio Weights Over Time")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Weight")
    axes[3].set_ylim([0, 1])
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
