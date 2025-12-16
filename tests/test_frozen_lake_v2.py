"""Unit tests for FrozenLakeV2 environment."""

import jax
import jax.numpy as jnp
import pytest
from safenax.frozen_lake.frozen_lake_v2 import (
    FrozenLakeV2,
    EnvState,
    TILE_FROZEN,
    TILE_THIN,
    TILE_GOAL,
)

# Fix random key for reproducibility
KEY = jax.random.PRNGKey(42)


@pytest.fixture
def env():
    """Fixture to create a standard 4x4 FrozenLakeV2 environment."""
    return FrozenLakeV2(map_name="4x4", is_slippery=True)


@pytest.fixture
def params(env):
    """Fixture for default environment parameters."""
    return env.default_params


def test_initialization(env, params):
    """Test that the environment initializes and resets correctly."""
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    # Check initial state
    assert state.pos == 0
    assert state.time == 0
    assert obs.shape == ()  # Scalar observation
    assert obs == 0


def test_step_mechanics(env, params):
    """Test basic stepping functionality."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    # Take an action (e.g., RIGHT = 2)
    action = 2
    obs, next_state, reward, done, info = env.step_env(key, state, action, params)

    # Basic shape/type checks
    assert isinstance(next_state, EnvState)
    assert next_state.time == 1
    assert "cost" in info
    assert info["cost"] >= 0


def test_slippery_dynamics(env, params):
    """
    Test that the agent actually slips when is_slippery=True.
    Starting at (1,1), taking action RIGHT (2) should stochastically result
    in moving RIGHT, UP, or DOWN.
    """
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 100)  # Run 100 steps

    # Force state to be at (1,1) [pos=5 for 4x4] to allow movement in all directions
    # Map:
    # S F F F (0, 1, 2, 3)
    # F T F T (4, 5, 6, 7)
    state = EnvState(pos=5, time=0)
    action = 2  # RIGHT

    # Vmap the step function to run in parallel
    def step_fn(k):
        return env.step_env(k, state, action, params)[1].pos

    next_positions = jax.vmap(step_fn)(keys)

    # From 5 (1,1):
    # Right -> 6 (1,2) [Intended]
    # Down  -> 9 (2,1) [Slip Right]
    # Up    -> 1 (0,1) [Slip Left]

    unique_positions = jnp.unique(next_positions)

    # Assert that we ended up in more than just the intended position
    assert len(unique_positions) > 1
    assert 6 in unique_positions  # Intended
    assert (9 in unique_positions) or (1 in unique_positions)  # Slips


def test_cost_mean_variance(env, params):
    """
    CRITICAL TEST: Verify the Mean-Variance trade-off for VaR-CPO.
    Safe Ice should have Mean~2.0, Std~0.1.
    Thin Ice should have Mean~1.5, Std~4.35.
    """
    N = 5000  # High sample count for statistical significance
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N)

    # We will force the map description to be entirely one type to isolate the cost function logic
    # regardless of where the agent moves.

    def get_batch_costs(tile_type):
        """Helper to get costs for N steps on a map filled with `tile_type`."""
        # Create a dummy map filled with the specific tile type
        custom_desc = jnp.full((4, 4), tile_type, dtype=jnp.int32)
        custom_params = params.replace(desc=custom_desc)

        def step_fn(k):
            # Start at 0, take action 0. Destination will be same tile type.
            _, _, _, _, info = env.step_env(k, EnvState(0, 0), 0, custom_params)
            return info["cost"]

        return jax.vmap(step_fn)(keys)

    # 1. Test Safe Ice ('F')
    safe_costs = get_batch_costs(TILE_FROZEN)

    # Check Safe Statistics: Mean ~ 2.0, Std ~ 0.1
    assert jnp.abs(jnp.mean(safe_costs) - 2.0) < 0.1, (
        f"Safe Mean {jnp.mean(safe_costs)} != 2.0"
    )
    assert jnp.std(safe_costs) < 0.2, f"Safe Std {jnp.std(safe_costs)} is too high"

    # 2. Test Thin Ice ('T')
    thin_costs = get_batch_costs(TILE_THIN)

    # Check Thin Statistics: Mean ~ 1.5, Std ~ 4.35
    assert jnp.abs(jnp.mean(thin_costs) - 1.5) < 0.1, (
        f"Thin Mean {jnp.mean(thin_costs)} != 1.5"
    )
    assert jnp.abs(jnp.std(thin_costs) - 4.35) < 0.1, (
        f"Thin Std {jnp.std(thin_costs)} is too low (expected ~4.35)"
    )

    print(
        f"\nStats Verification:\nSafe: Mean={jnp.mean(safe_costs):.2f}, Std={jnp.std(safe_costs):.2f}"
    )
    print(f"Thin: Mean={jnp.mean(thin_costs):.2f}, Std={jnp.std(thin_costs):.2f}")


def test_termination(env, params):
    """Test episode termination logic."""
    key = jax.random.PRNGKey(0)

    # 1. Test Goal Termination
    # Force map to be all GOAL
    goal_desc = jnp.full((4, 4), TILE_GOAL, dtype=jnp.int32)
    goal_params = params.replace(desc=goal_desc)

    # Agent takes a step on a Goal tile -> should be done
    _, _, _, done, _ = env.step_env(key, EnvState(0, 0), 0, goal_params)
    assert done == True

    # 2. Test Time Truncation
    # Set max_steps to 1
    short_params = params.replace(max_steps_in_episode=1)

    # Step 1: time becomes 1 -> truncated >= max_steps (1) -> done=True
    state = EnvState(0, 0)
    _, next_state, _, done, _ = env.step_env(key, state, 0, short_params)

    assert next_state.time == 1
    assert done == True
