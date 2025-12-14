import pytest
import jax
import jax.numpy as jnp
from safenax import EcoAntV1


@pytest.fixture
def env():
    """Initializes the environment for testing."""
    return EcoAntV1(battery_limit=10.0, noise_scale=0.1)

@pytest.fixture
def key():
    """Provides a JAX PRNG key."""
    return jax.random.PRNGKey(0)

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
    new_info = state.info.copy()
    new_info["battery"] = jnp.array(0.1)
    
    # Ideally we update obs too for consistency, but physics logic relies on info
    near_death_state = state.replace(info=new_info)
    
    # 2. Apply a large action to ensure depletion
    # Energy used approx 4.0, which is > 0.1
    action = jnp.ones(env.action_size)
    next_state = env.step(near_death_state, action)
    
    # Check Battery Floor (Percentage should be 0.0)
    assert next_state.obs[-1] == 0.0
    
    # Check Raw Battery Floor
    assert next_state.info["battery"] == 0.0
    
    # Check Cost Signal
    assert next_state.info["cost"] == 1.0
    
    # Check Termination
    assert next_state.done == 1.0