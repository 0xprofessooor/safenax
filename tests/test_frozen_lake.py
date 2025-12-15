import pytest
import jax
import jax.numpy as jnp
from safenax.frozen_lake import FrozenLake, EnvParams
from safenax.frozen_lake import (
    TILE_START,
    TILE_FROZEN,
    TILE_HOLE,
    TILE_GOAL,
    string_map_to_array,
)

# --- Fixtures ---


@pytest.fixture
def env():
    """Initializes the environment."""
    return FrozenLake()


@pytest.fixture
def rng():
    """Base random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def params_deterministic():
    """
    Creates a simple 2x2 deterministic map for testing logic.
    Layout:
      S F  (0, 1)
      H G  (2, 3)
    """
    desc = jnp.array(
        [[TILE_START, TILE_FROZEN], [TILE_HOLE, TILE_GOAL]], dtype=jnp.int32
    )

    return EnvParams(
        desc=desc,
        nrow=2,
        ncol=2,
        is_slippery=False,  # Deterministic
        success_rate=1.0,
        reward_schedule=jnp.array([1.0, 0.0, 0.0]),  # Goal=1, Hole=0, Step=0
        max_steps_in_episode=10,
    )


@pytest.fixture
def params_slippery(params_deterministic: EnvParams) -> EnvParams:
    """Same 2x2 map but with slippery physics enabled."""
    return params_deterministic.replace(is_slippery=True, success_rate=0.5)


# --- Test Cases ---


def test_jit_compilation(
    env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams
):
    """Verifies that reset and step can be JIT compiled without errors."""
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    # Run once to trigger compilation
    obs, state = reset_jit(rng, params_deterministic)
    # Take a dummy step
    step_jit(rng, state, 0, params_deterministic)

    assert True, "JIT compilation failed if this line is not reached."


def test_map_parsing():
    """Verifies the string-to-array conversion works."""
    desc_str = ["SFFF", "HFFG"]
    desc_arr = string_map_to_array(desc_str)

    assert desc_arr.shape == (2, 4)
    # Check Goal at (1, 3)
    assert desc_arr[1, 3] == TILE_GOAL
    # Check Hole at (1, 0)
    assert desc_arr[1, 0] == TILE_HOLE


def test_deterministic_reach_goal(
    env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams
):
    """
    Path: Start(0) -> Right(1) -> Down(3/Goal).
    Verifies reward and termination.
    """
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    rng, key = jax.random.split(rng)
    _, state = reset_jit(key, params_deterministic)

    # 1. Move Right (Action 2) -> Pos 1 (Frozen)
    rng, key = jax.random.split(rng)
    _, state, reward, done, _ = step_jit(key, state, 2, params_deterministic)

    assert state.pos == 1
    assert reward == 0.0
    assert not done

    # 2. Move Down (Action 1) -> Pos 3 (Goal)
    # NOTE: Gymnax auto-resets on DONE. The returned 'state' will be the NEW start state (0).
    # We must check 'reward' and 'done' to verify success.
    rng, key = jax.random.split(rng)
    _, state, reward, done, _ = step_jit(key, state, 1, params_deterministic)

    assert done
    assert reward == 1.0
    # Verify auto-reset happened (back at start)
    assert state.pos == 0
    assert state.time == 0


def test_deterministic_fall_in_hole(
    env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams
):
    """
    Path: Start(0) -> Down(2/Hole).
    Verifies termination without reward.
    """
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    rng, key = jax.random.split(rng)
    _, state = reset_jit(key, params_deterministic)

    # Move Down (Action 1) -> Pos 2 (Hole)
    rng, key = jax.random.split(rng)
    _, state, reward, done, _ = step_jit(key, state, 1, params_deterministic)

    assert done
    assert reward == 0.0
    # Verify auto-reset happened
    assert state.pos == 0


def test_custom_reward_schedule(
    env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams
):
    """
    Verifies that changing reward_schedule affects the output.
    We set Hole Reward = -5.0.
    """
    # Create params with negative reward for holes
    custom_params = params_deterministic.replace(
        reward_schedule=jnp.array([10.0, -5.0, -0.1])  # [Goal, Hole, Step]
    )

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    rng, key = jax.random.split(rng)
    _, state = reset_jit(key, custom_params)

    # Move Down -> Hole
    rng, key = jax.random.split(rng)
    _, _, reward, done, _ = step_jit(key, state, 1, custom_params)

    assert done
    assert reward == -5.0


def test_slippery_dynamics(env: FrozenLake, rng: jax.Array, params_slippery: EnvParams):
    """
    Statistical test for slippery logic.
    We start at (0,0) and try to move Right.
    - Success (50%): Move Right (Pos 1)
    - Slip Up (25%): Hit Wall -> Stay (Pos 0)
    - Slip Down (25%): Hit Hole (Pos 2) -> Auto-reset to Start (Pos 0)

    If logic works, we should see a mix of Pos 1 and Pos 0.
    If it were deterministic, we would ONLY see Pos 1.
    """
    batch_size = 100

    # Batch Reset
    batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    keys = jax.random.split(rng, batch_size)
    _, states = batch_reset(keys, params_slippery)

    # Batch Step (Action 2 = Right)
    batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    step_keys = jax.random.split(rng, batch_size)
    actions = jnp.full((batch_size,), 2)

    _, next_states, _, _, _ = batch_step(step_keys, states, actions, params_slippery)

    unique_positions = jnp.unique(next_states.pos)

    # We expect to see 0 and 1.
    assert len(unique_positions) > 1, (
        f"Expected stochasticity, got only: {unique_positions}"
    )
    assert 1 in unique_positions


def test_truncation(env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams):
    """Test that episode ends when time limit is reached."""
    # Set max steps to 2
    short_params = params_deterministic.replace(max_steps_in_episode=2)

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    rng, key = jax.random.split(rng)
    _, state = reset_jit(key, short_params)

    # Step 1: Move Left (into wall, stay at 0)
    _, state, _, done, _ = step_jit(key, state, 0, short_params)
    assert not done
    assert state.time == 1

    # Step 2: Move Left (into wall, stay at 0) -> Time hits 2 -> Done
    _, state, _, done, _ = step_jit(key, state, 0, short_params)
    assert done
    # Note: Gymnax resets time to 0 on done
    assert state.time == 0


def test_jit_rollout_scan(env: FrozenLake, rng: jax.Array, params_slippery: EnvParams):
    """
    Verifies that the environment can be run in a fully compiled jax.lax.scan loop.
    This ensures no shape mismatches or control flow issues exist in the step logic.
    """
    num_steps = 100

    # 2. Define the scan step function
    def rollout_step(carry, _):
        key, state = carry
        key, step_key, action_key = jax.random.split(key, 3)

        # Select random action
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=4)

        # Step env
        obs, next_state, reward, done, info = env.step(
            step_key, state, action, params_slippery
        )

        # Stack outputs
        return (key, next_state), {
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
        }

    # 3. JIT Compile the entire loop
    @jax.jit
    def run_scan(rng):
        # Reset
        rng, reset_key = jax.random.split(rng)
        _, start_state = env.reset(reset_key, params_slippery)

        # Scan
        final_carry, data = jax.lax.scan(
            rollout_step, (rng, start_state), None, length=num_steps
        )
        return data

    # 4. Execute
    # If this crashes, the env is not fully JIT-compatible
    rollout_data = run_scan(rng)

    # 5. Basic Assertions
    # Check shapes to ensure we got data for every step
    assert rollout_data["obs"].shape == (num_steps,)
    assert rollout_data["reward"].shape == (num_steps,)
    assert rollout_data["done"].shape == (num_steps,)

    # Check data validity
    assert jnp.all(rollout_data["action"] >= 0)
    assert jnp.all(rollout_data["action"] < 4)
    # Rewards should be 0.0 or 1.0 (unless customized)
    assert jnp.all(jnp.isin(rollout_data["reward"], jnp.array([0.0, 1.0])))


def test_cost_signal(env: FrozenLake, rng: jax.Array, params_deterministic: EnvParams):
    """Ensure cost is 1.0 ONLY when falling into a hole."""
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    rng, key = jax.random.split(rng)
    _, state = reset_jit(key, params_deterministic)

    # 1. Safe Move (Right -> Frozen)
    rng, key = jax.random.split(rng)
    _, state, _, _, info = step_jit(key, state, 2, params_deterministic)

    assert info["cost"] == 0.0, "Cost should be 0 on safe tiles"

    # 2. Unsafe Move (Down -> Hole) (Reset to start for clarity)
    _, state = reset_jit(key, params_deterministic)
    rng, key = jax.random.split(rng)
    _, state, _, _, info = step_jit(key, state, 1, params_deterministic)

    assert info["cost"] == 1.0, "Cost should be 1 when entering a hole"
