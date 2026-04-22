from typing import Dict, Tuple
from enum import Enum
import os
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments import spaces
from flax import struct
import polars as pl
import jax
from jax import numpy as jnp


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
            low=-jnp.inf,
            high=jnp.inf,
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

        # Get the first close price in the window for each asset
        initial_close = step_data[0:1, :, KLineFeatures.CLOSE.value] + 1e-8

        # Normalize Open, High, Low, and Close by the initial close price
        for feature in [
            KLineFeatures.OPEN,
            KLineFeatures.HIGH,
            KLineFeatures.LOW,
            KLineFeatures.CLOSE,
        ]:
            norm_feature = step_data[:, :, feature.value] / initial_close
            step_data = step_data.at[:, :, feature.value].set(norm_feature)

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
        info = {"cost": jnp.maximum(0.0, -reward)}
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
