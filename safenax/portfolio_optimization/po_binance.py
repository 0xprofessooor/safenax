from typing import Dict, Tuple
from enum import Enum
import os
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments import spaces
from flax import struct
import polars as pl
import jax
from jax import numpy as jnp
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative


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


class KLineFeatures(Enum):
    CLOSE = 0
    OPEN = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    TAKER_BUY_VOLUME = 5
    NUM_TRADES = 6


@struct.dataclass
class EnvState:
    step: int
    time: int
    prices: jax.Array
    holdings: jax.Array
    values: jax.Array
    total_value: float


@struct.dataclass
class EnvParams:
    max_steps: int
    initial_cash: float
    taker_fee: float
    gas_fee: float
    trade_threshold: float


class PortfolioOptimizationBinance(Environment):
    def __init__(
        self, data_paths: Dict[str, str], step_size: int = 1, window_size: int = 50
    ):
        super().__init__()
        data_dict = {key: load_binance_klines(path) for key, path in data_paths.items()}
        self.assets = sorted(data_dict.keys())
        self.data = jnp.stack(
            [data_dict[asset] for asset in self.assets], axis=1
        )  # shape (num_rows, num_assets, num_features)
        self.step_size = step_size
        self.window_size = window_size

        print(f"Loaded data for assets: {self.assets}")
        print(f"Data shape: {self.data.shape} (time, assets, features)")

    @property
    def name(self) -> str:
        return "PO-Binance"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps=1000,
            initial_cash=1000.0,
            taker_fee=BinanceFeeTier.OFF.value,
            gas_fee=0.0,
            trade_threshold=0.0,
        )

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.data.shape[1] + 1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_shape = (self.window_size * self.data.shape[1] * self.data.shape[2],)
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=obs_shape, dtype=jnp.float32
        )

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        start_time_idx = jnp.maximum(0, state.time - self.window_size + 1)
        start_indices = (start_time_idx, 0, 0)
        slice_sizes = (self.window_size, self.data.shape[1], self.data.shape[2])
        step_data = jax.lax.dynamic_slice(self.data, start_indices, slice_sizes)
        return step_data.flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        max_steps_reached = state.step >= params.max_steps
        portfolio_bankrupt = state.total_value <= 0
        return jnp.logical_or(max_steps_reached, portfolio_bankrupt)

    def step_env(
        self, key: jax.Array, state: EnvState, action: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        time = state.time + self.step_size
        prices = jnp.concatenate(
            [jnp.array([1.0]), self.data[time, :, KLineFeatures.CLOSE.value]]
        )

        # normalize action
        weights = action

        ############### UPDATE PORTFOLIO WITH FEES ###############
        values = state.holdings * prices
        total_value = jnp.sum(values)
        asset_values = values[1:]
        asset_weights = weights[1:]
        new_asset_values_no_fee = total_value * asset_weights
        deltas_no_fee = new_asset_values_no_fee - asset_values
        num_trades = jnp.sum(jnp.abs(deltas_no_fee) > params.trade_threshold)
        gas_cost = params.gas_fee * num_trades

        # split buy and sell orders
        buy_indices = deltas_no_fee > params.trade_threshold
        sell_indices = deltas_no_fee < -params.trade_threshold
        no_trade_indices = jnp.concatenate(
            [jnp.array([False]), jnp.abs(deltas_no_fee) <= params.trade_threshold]
        )
        buy_weights = jnp.where(buy_indices, asset_weights, 0.0)
        sell_weights = jnp.where(sell_indices, asset_weights, 0.0)
        current_buy_values = jnp.where(buy_indices, asset_values, 0.0)
        current_sell_values = jnp.where(sell_indices, asset_values, 0.0)

        # calculate new portfolio value after fees
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

        next_state = EnvState(
            step=state.step + 1,
            time=time,
            prices=prices,
            holdings=new_holdings,
            values=adj_new_values,
            total_value=new_total_value,
        )
        obs = self.get_obs(next_state, params)
        done = self.is_terminal(next_state, params)
        info = {"cost": -reward}
        return obs, next_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        episode_length = params.max_steps * self.step_size
        max_start = self.data.shape[0] - episode_length
        min_start = self.window_size
        time = jax.random.randint(key, (), min_start, max_start)
        prices = jnp.concatenate(
            [jnp.array([1.0]), self.data[time, :, KLineFeatures.CLOSE.value]]
        )
        holdings = jnp.zeros(len(self.assets) + 1)
        holdings = holdings.at[0].set(params.initial_cash)
        values = holdings * prices
        state = EnvState(
            step=0,
            time=time,
            prices=prices,
            holdings=holdings,
            values=values,
            total_value=jnp.sum(values),
        )
        obs = self.get_obs(state, params)
        return obs, state


def load_binance_klines(filepath: str) -> jax.Array:
    if os.path.isdir(filepath):
        files = sorted(
            os.path.join(filepath, f)
            for f in os.listdir(filepath)
            if f.endswith(".parquet")
        )
        df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
    else:
        df = pl.read_parquet(filepath)
    data = df.select(
        (pl.col("close")),
        (pl.col("open")),
        (pl.col("high")),
        (pl.col("low")),
        (pl.col("quote_asset_volume")),
        (pl.col("taker_buy_quote_volume")),
        (pl.col("number_of_trades")),
    )
    data = data.to_jax()
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    data_paths = {
        "BTC": "data/binance/BTCUSDT/klines/",
        "ETH": "data/binance/ETHUSDT/klines/",
        "SOL": "data/binance/SOLUSDT/klines/",
        "BNB": "data/binance/BNBUSDT/klines/",
        "DOGE": "data/binance/DOGEUSDT/klines/",
        "ADA": "data/binance/ADAUSDT/klines/",
        "LINK": "data/binance/LINKUSDT/klines/",
        "XRP": "data/binance/XRPUSDT/klines/",
        "XTZ": "data/binance/XTZUSDT/klines/",
    }

    window_size = 1440
    env = PortfolioOptimizationBinance(
        data_paths=data_paths, step_size=1, window_size=window_size
    )
    params = env.default_params.replace(
        max_steps=1000000, taker_fee=BinanceFeeTier.REGULAR.value, trade_threshold=10.0
    )
    num_assets = len(env.assets)
    num_features = env.data.shape[2]

    @jax.jit
    def max_sharpe_action(obs: jax.Array) -> jax.Array:
        """Compute max-Sharpe weights using Projected Gradient Descent for strict long-only bounds."""
        # 1. Extract and calculate Mu and Sigma
        close = obs.reshape(window_size, num_assets, num_features)[
            :, :, KLineFeatures.CLOSE.value
        ]
        log_returns = jnp.diff(jnp.log(close), axis=0)
        mu_crypto = jnp.mean(log_returns, axis=0)
        centered = log_returns - mu_crypto

        # Calculate Covariance (Sigma)
        sigma_crypto = (centered.T @ centered) / (log_returns.shape[0] - 1)

        # 2. Setup Cash + Crypto arrays
        n = num_assets + 1
        mu = jnp.concatenate([jnp.array([0.0]), mu_crypto])

        # Add a tiny bit of regularization to the diagonal to keep the matrix stable
        sigma = jnp.zeros((n, n)).at[0, 0].set(1e-8)
        sigma = sigma.at[1:, 1:].set(sigma_crypto + jnp.eye(num_assets) * 1e-8)

        # 3. Define the Quadratic Programming objective function
        def objective(w, sigma, mu):
            # f(w) = 0.5 * w^T * Sigma * w - mu^T * w
            variance_penalty = 0.5 * jnp.dot(w, jnp.dot(sigma, w))
            return variance_penalty - jnp.dot(mu, w)

        # 4. Initialize the JAXopt Projected Gradient solver
        # It strictly enforces the projection_non_negative constraint (w >= 0) at every step
        pg = ProjectedGradient(
            fun=objective, projection=projection_non_negative, maxiter=100, tol=1e-4
        )

        # 5. Run the optimizer from an equal-weight starting guess
        w_init = jnp.ones(n) / n
        state = pg.run(w_init, sigma=sigma, mu=mu)

        # Extract the optimal weights
        w_optimal = state.params

        # 6. Normalize so the weights sum to exactly 1.0 (100% of portfolio)
        w_final = w_optimal / (jnp.sum(w_optimal) + 1e-10)

        return w_final

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)

    portfolio_values = [float(state.total_value)]
    done = False
    start = time.perf_counter()
    while not done:
        action = max_sharpe_action(obs)
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = env.step(step_key, state, action, params)
        if not done:
            portfolio_values.append(float(state.total_value))
    end = time.perf_counter()
    print(f"{params.max_steps / (end - start):.2f} steps/s")

    print(f"Final portfolio value: {portfolio_values[-1]:.2f}")
    print(
        f"Total return: {(portfolio_values[-1] / portfolio_values[0] - 1) * 100:.2f}%"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(portfolio_values)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value (USDT)")
    plt.title(f"Markowitz Max-Sharpe Portfolio ({window_size}-step lookback)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("markowitz_portfolio.png")
    plt.show()
