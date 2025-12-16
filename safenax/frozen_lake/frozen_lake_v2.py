"""JAX-compatible FrozenLakeV2 environment following the gymnax interface."""

from typing import Optional, Tuple, Union, List

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

# --- Constants ---
# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Tile Types (using ASCII values for readability during debugging)
TILE_START = ord("S")
TILE_FROZEN = ord("F")
TILE_THIN = ord("T")
TILE_GOAL = ord("G")


@struct.dataclass
class EnvState:
    """Environment state for FrozenLake."""

    pos: int
    time: int


@struct.dataclass
class EnvParams:
    """Environment parameters for FrozenLake."""

    desc: jax.Array  # Map description as integer array (nrow, ncol)
    nrow: int
    ncol: int
    is_slippery: bool
    success_rate: float
    reward_schedule: jax.Array  # (goal_reward, thin_reward, frozen_reward)
    max_steps_in_episode: int
    safe_cost_mean: float
    safe_cost_std: float
    thin_cost_base: float
    thin_shock_prob: float
    thin_shock_val: float


# --- Helper Functions ---


def string_map_to_array(map_desc: List[str]) -> jax.Array:
    """Convert a list of strings into a JAX-friendly integer array."""
    return jnp.array([[ord(c) for c in row] for row in map_desc], dtype=jnp.int32)


# Predefined maps
MAPS = {
    "4x4": string_map_to_array(["SFFF", "FTFT", "FFFT", "TFFG"]),
    "8x8": string_map_to_array(
        [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFTFFFF",
            "FFFFFTFF",
            "FFFTFFFF",
            "FTTFFFTF",
            "FTFFTFTF",
            "FFFTFFFG",
        ]
    ),
}


class FrozenLakeV2(environment.Environment):
    """
    JAX-compatible FrozenLake environment with "Thin Ice" cost augmentation.

    The agent controls the movement of a character in a grid world to reach a Goal (G).
    The map contains two types of traversable terrain that introduce a Mean-Variance trade-off:

    1. Safe Ice ('F'):
       - Represents a stable but arduous path (deep snow is hard to traverse).
       - Cost: High Mean (e.g., 2.0), Small Variance (e.g., std=0.1).
       - Risk: Deterministically expensive, but safe.

    2. Thin Ice ('T', formerly Holes):
       - Represents a dangerous shortcut, prone to cracking or falling but can glide over with little effort.
       - Cost: Low Mean (e.g., 1.5), High Variance Shocks (e.g. std=4.35) (ice cracks).
       - Risk: Usually cheap to traverse, but prone to random, catastrophic cost spikes (cracking ice).

    The agent can also slip perpendicular to its intended direction based on the `is_slippery` and `success_rate` parameters.
    """

    def __init__(
        self,
        map_name: str = "4x4",
        desc: Optional[jax.Array] = None,
        is_slippery: bool = True,
        success_rate: float = 1.0 / 3.0,
        reward_schedule: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        safe_cost_mean: float = 2.0,
        safe_cost_std: float = 0.1,
        thin_cost_base: float = 0.5,
        thin_shock_prob: float = 0.05,
        thin_shock_val: float = 20.0,
    ):
        super().__init__()

        if desc is None:
            desc = MAPS[map_name]

        self.desc = desc
        self.nrow, self.ncol = desc.shape
        self.is_slippery = is_slippery
        self.success_rate = success_rate
        self.reward_schedule = jnp.array(reward_schedule)
        self.safe_cost_mean = safe_cost_mean
        self.safe_cost_std = safe_cost_std
        self.thin_cost_base = thin_cost_base
        self.thin_shock_prob = thin_shock_prob
        self.thin_shock_val = thin_shock_val

        # Determine max steps based on map size convention
        if map_name == "4x4" or (desc.shape[0] == 4 and desc.shape[1] == 4):
            self.max_steps = 100
        else:
            self.max_steps = 200

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            desc=self.desc,
            nrow=self.nrow,
            ncol=self.ncol,
            is_slippery=self.is_slippery,
            success_rate=self.success_rate,
            reward_schedule=self.reward_schedule,
            max_steps_in_episode=self.max_steps,
            safe_cost_mean=self.safe_cost_mean,
            safe_cost_std=self.safe_cost_std,
            thin_cost_base=self.thin_cost_base,
            thin_shock_prob=self.thin_shock_prob,
            thin_shock_val=self.thin_shock_val,
        )

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform a single environment step with JIT compatibility."""
        action = jnp.int32(action)

        # 1. Determine the actual direction (handling slippery logic)
        rng_slip, rng_cost, key = jax.random.split(key, 3)

        def get_slippery_action(k):
            # 0: intended, 1: perpendicular left, 2: perpendicular right
            # Probabilities: [success, (1-success)/2, (1-success)/2]
            fail_prob = (1.0 - params.success_rate) / 2.0
            probs = jnp.array([params.success_rate, fail_prob, fail_prob])

            # Map samples (0, 1, 2) to actions (action, action-1, action+1)
            delta_idx = jax.random.choice(k, jnp.arange(3), p=probs)

            # (action - 1) % 4  <-- Perpendicular Left
            # (action)          <-- Intended
            # (action + 1) % 4  <-- Perpendicular Right

            # Using a lookup array for cleaner mapping
            candidates = jnp.array(
                [
                    action,  # Index 0: Success
                    (action - 1) % 4,  # Index 1: Fail Left
                    (action + 1) % 4,  # Index 2: Fail Right
                ]
            )
            return candidates[delta_idx]

        actual_action = jax.lax.cond(
            params.is_slippery,
            get_slippery_action,
            lambda k: action,  # Deterministic branch
            rng_slip,
        )

        # 2. Calculate Movement (Optimized Vectorized Approach)
        row = state.pos // params.ncol
        col = state.pos % params.ncol

        next_row, next_col = self._apply_action(row, col, actual_action, params)
        next_pos = next_row * params.ncol + next_col

        # 3. Check Rewards and Termination
        tile_type = params.desc[next_row, next_col]

        is_goal = tile_type == TILE_GOAL
        is_thin = tile_type == TILE_THIN

        # Reward Schedule: [Goal, Thin, Frozen/Start]
        # We select index 0, 1, or 2 based on tile type
        tile_idx = jnp.where(is_goal, 0, jnp.where(is_thin, 1, 2))
        reward = params.reward_schedule[tile_idx]

        safe_noise = jax.random.normal(rng_cost) * params.safe_cost_std
        cost_safe = params.safe_cost_mean + safe_noise

        is_shock = jax.random.bernoulli(rng_cost, p=params.thin_shock_prob)
        cost_thin = params.thin_cost_base + (is_shock * params.thin_shock_val)

        cost = jnp.select([is_goal, is_thin], [0.0, cost_thin], default=cost_safe)

        # 4. Update State
        # Time limit truncation is handled in Gymnax wrappers usually,
        # but we track it here for the 'done' flag consistency.
        new_time = state.time + 1
        truncated = new_time >= params.max_steps_in_episode
        done = is_goal | truncated

        new_state = EnvState(pos=next_pos, time=new_time)

        return self.get_obs(new_state, params), new_state, reward, done, {"cost": cost}

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment to the start position."""
        # By definition, FrozenLake always starts at (0,0)
        # If dynamic start positions are needed, one would scan params.desc for 'S'
        state = EnvState(pos=0, time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return scalar observation (flat position index)."""
        return jnp.array(state.pos, dtype=jnp.int32)

    def _apply_action(
        self, row: int, col: int, action: int, params: EnvParams
    ) -> Tuple[int, int]:
        """Optimized movement logic using delta arrays and clipping."""
        # Gym Direction Mapping:
        # 0: Left  (0, -1)
        # 1: Down  (1,  0)
        # 2: Right (0,  1)
        # 3: Up    (-1, 0)

        dr = jnp.array([0, 1, 0, -1])
        dc = jnp.array([-1, 0, 1, 0])

        new_row = row + dr[action]
        new_col = col + dc[action]

        # Ensure we stay within the grid
        new_row = jnp.clip(new_row, 0, params.nrow - 1)
        new_col = jnp.clip(new_col, 0, params.ncol - 1)

        return new_row, new_col

    @property
    def name(self) -> str:
        return "FrozenLake-v2"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(params.nrow * params.ncol)


# --- Factory Helper ---


def make_frozen_lake(
    map_name: str = "4x4",
    is_slippery: bool = True,
    success_rate: float = 1.0 / 3.0,
    reward_schedule: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> FrozenLakeV2:
    """Factory function to easier initialization."""
    return FrozenLakeV2(
        map_name=map_name,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=reward_schedule,
    )
