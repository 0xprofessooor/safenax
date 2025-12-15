import pytest
import jax
import jax.numpy as jnp
from safenax import EcoAntV1
from safenax.wrappers import BraxToGymnaxWrapper
from brax.envs.ant import Ant


@pytest.fixture
def env():
    """Initializes the environment for testing."""
    return EcoAntV1(battery_limit=10.0, noise_scale=0.1)


@pytest.fixture
def key():
    """Provides a JAX PRNG key."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def wrapped_env(env):
    """Initializes the Wrapper with the EcoAnt instance."""
    # We set a short max_episode_length to ensure we don't hit time limits easily
    return BraxToGymnaxWrapper(env=env, episode_length=10)


def test_obs_size(env: EcoAntV1):
    """Verifies that observation size is correct (original + 1 for battery)."""
    original_obs_size = Ant().observation_size
    expected_size = original_obs_size + 1  # +1 for battery percentage
    assert env.observation_size == expected_size


def test_initialization(env: EcoAntV1, key: jax.Array):
    """Verifies that reset places the battery PERCENTAGE (1.0) in observation."""
    state = env.reset(key)

    # Check battery PCT is at the end of observation (Should be 1.0, i.e., 100%)
    initial_battery_pct = state.obs[-1]
    assert initial_battery_pct == 1.0

    # Check raw battery in info matches the limit
    assert state.info["battery"] == 10.0

    # Check cost is initialized to 0
    assert state.info["cost"] == 0.0


def test_battery_decay(env: EcoAntV1, key: jax.Array):
    """Verifies that taking a non-zero action reduces the battery percentage."""
    state = env.reset(key)

    # Create a generic action
    action = jnp.ones(env.action_size) * 0.5
    next_state = env.step(state, action)

    prev_battery_pct = state.obs[-1]
    new_battery_pct = next_state.obs[-1]

    # Battery percentage should decrease
    assert new_battery_pct < prev_battery_pct

    # Check that raw battery in info also decreased
    assert next_state.info["battery"] < state.info["battery"]

    # Check that raw battery hasn't gone down by more than 4.0
    assert state.info["battery"] - next_state.info["battery"] <= 4.0


def test_stochasticity_consumes_energy(env: EcoAntV1, key: jax.Array):
    """Verifies that noise causes energy use even when action is zero."""
    state = env.reset(key)

    # Action is zero
    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    # Due to noise_scale=0.1, actual action != 0, so energy must be used
    assert next_state.obs[-1] < 1.0


def test_termination_and_cost(env: EcoAntV1, key: jax.Array):
    """Verifies cost signal and termination when battery hits zero."""
    state = env.reset(key)

    # 1. Manually set battery to near-death in INFO (not Obs)
    # The environment logic reads current_battery = state.info["battery"]
    # So to test termination, we must hack the info dict.
    env.battery_limit = 0.1  # Temporarily set limit low for test
    new_info = state.info.copy()
    new_info["battery"] = jnp.array(0.1)

    # Ideally we update obs too for consistency, but physics logic relies on info
    near_death_state = state.replace(info=new_info)

    # 2. Apply a large action to ensure depletion
    # Energy used approx 4.0, which is > 0.1
    action = jnp.ones(env.action_size)
    next_state = env.step(near_death_state, action)

    # Check Battery Floor (Percentage should be 0.0)
    assert next_state.obs[-1] <= 0.0

    # Check Raw Battery Floor
    assert next_state.info["battery"] == 0.0

    # Check Cost Signal
    assert next_state.info["cost"] == 1.0

    # Check Termination
    assert next_state.done == 1.0


def test_wrapper_reset_api(wrapped_env: BraxToGymnaxWrapper, key: jax.Array):
    """Verifies wrapper reset returns (obs, state) tuple correctly."""
    obs, state = wrapped_env.reset(key, wrapped_env.default_params)

    # Check return types
    assert isinstance(obs, jax.Array)
    # Check battery is present in the unwrapped observation
    assert obs[-1] == 1.0
    # Check state is carried through
    assert state.info["battery"] == 10.0


def test_wrapper_step_api(wrapped_env: BraxToGymnaxWrapper, key: jax.Array):
    """Verifies wrapper step returns (obs, state, reward, done, info)."""
    obs, state = wrapped_env.reset(key)
    action = jnp.zeros(wrapped_env.action_size)

    next_obs, next_state, reward, done, info = wrapped_env.step(key, state, action)

    # 1. Check Structure
    assert next_obs.shape == obs.shape
    assert isinstance(reward, jax.Array)
    assert isinstance(done, jax.Array)

    # 2. Check Custom Info Passthrough
    # The wrapper should preserve the 'cost' and 'battery' keys
    assert "cost" in info
    assert "battery" in info
    # Should be slightly less than 10.0 due to noise
    assert info["battery"] < 10.0


def test_wrapper_autoreset_logic(key: jax.Array):
    """
    CRITICAL TEST: Verifies AutoResetWrapper behavior on battery death.

    Expected Behavior:
    1. Input State has battery ~0.1
    2. Action drains battery -> Done=True, Cost=1.0
    3. AutoReset triggers immediately:
       - Returns `done=True` (signaling the end of the dying episode)
       - Returns `obs` from the NEW reset state (Battery=1.0, not 0.0!)
    """
    brax_env = EcoAntV1(battery_limit=0.1, noise_scale=0.1)
    wrapped_env = BraxToGymnaxWrapper(env=brax_env, episode_length=10)
    obs, state = wrapped_env.reset(key)

    # 2. Kill the agent
    action = jnp.ones(wrapped_env.action_size)
    next_obs, next_state, reward, done, info = wrapped_env.step(key, state, action)

    # --- Assertions ---

    # A. The Episode ended
    assert done == 1.0

    # B. The 'info' contains the terminal cost of the DEAD agent
    # But includes the battery reset
    assert info["cost"] == 1.0
    assert info["battery"] == 0.0

    # C. The 'next_obs' is from the NEW alive agent (AutoReset happened)
    # The observation returned is for the *next* step.
    assert next_obs[-1] == 1.0  # Full battery!

    # D. The 'next_state' is the NEW alive state
    assert next_state.info["cost"] == 1.0
    assert next_state.info["battery"] == 0.0
