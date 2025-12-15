import pytest
import jax
import jax.numpy as jnp
from safenax import EcoAntV2


@pytest.fixture
def env():
    """Initializes the V2 environment for testing."""
    return EcoAntV2(battery_limit=10.0, noise_scale=0.1)


@pytest.fixture
def key():
    """Provides a JAX PRNG key."""
    return jax.random.PRNGKey(0)


def test_observation_size_unchanged(env, key):
    """
    Verifies that V2 does NOT modify the observation size.
    It should match the standard Ant exactly.
    """
    state = env.reset(key)

    # Since we didn't override observation_size in V2, it should equal super's
    assert env.observation_size == super(EcoAntV2, env).observation_size

    # Verify the actual output shape matches
    assert state.obs.shape[-1] == env.observation_size


def test_initialization(env: EcoAntV2, key: jax.Array):
    """Verifies correct initialization of info dict."""
    state = env.reset(key)

    # Battery should start at limit in info
    assert state.info["battery"] == 10.0
    # Cost should start at 0
    assert state.info["cost"] == 0.0


def test_energy_cost_tracking(env: EcoAntV2, key: jax.Array):
    """
    Verifies that 'cost' in info reflects the actual energy used per step.
    """
    state = env.reset(key)

    # Create a generic action
    action = jnp.ones(env.action_size) * 0.5
    next_state = env.step(state, action)

    cost = next_state.info["cost"]

    # 1. Cost should be strictly positive (energy used)
    assert cost > 0.0

    # 2. Verify logic: cost should roughly equal 0.5 * sum(action^2)
    # Note: We must account for noise, so we check approximate bounds.
    # Base energy ~ 0.5 * sum(0.5^2) for 8 joints = 1.0
    assert jnp.abs(cost - 1.0) < 0.5  # Large tolerance for noise


def test_battery_depletion_in_info(env: EcoAntV2, key: jax.Array):
    """
    Verifies that battery decreases in info['battery'] exactly by the cost amount.
    """
    state = env.reset(key)
    start_battery = state.info["battery"]

    action = jnp.ones(env.action_size)
    next_state = env.step(state, action)

    energy_used = next_state.info["cost"]
    current_battery = next_state.info["battery"]

    # Verify math: New = Old - Cost
    assert jnp.allclose(start_battery - energy_used, current_battery, atol=1e-5)


def test_termination_logic(key: jax.Array):
    """
    Verifies that the episode terminates when info['battery'] hits zero.
    """
    env = EcoAntV2(battery_limit=0.1, noise_scale=0.1)
    near_death_state = env.reset(key)

    # 2. Take a large step to consume > 0.1 energy
    # Action of 1.0s usually consumes ~4.0 energy
    action = jnp.ones(env.action_size)
    next_state = env.step(near_death_state, action)

    # 3. Verify Termination
    assert next_state.done == 1.0

    # 4. Verify Battery Floor
    assert next_state.info["battery"] == 0.0


def test_stochasticity_impact(env: EcoAntV2, key: jax.Array):
    """
    Verifies that even a zero action has a non-zero cost due to noise.
    """
    state = env.reset(key)
    action = jnp.zeros(env.action_size)

    next_state = env.step(state, action)

    # Cost should be > 0 because noise makes the effective action non-zero
    assert next_state.info["cost"] > 0.0
    # Battery should have decreased
    assert next_state.info["battery"] < 10.0
