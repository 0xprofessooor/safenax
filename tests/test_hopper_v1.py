import jax
import jax.numpy as jnp
import pytest

from safenax import HopperV1
from safenax.wrappers import BraxToGymnaxWrapper


@pytest.fixture
def env():
    return HopperV1()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def wrapped_env(env):
    return BraxToGymnaxWrapper(env=env, episode_length=10)


def test_obs_size(env):
    """Hopper observation is 11-dim."""
    assert env.observation_size == 11
    assert env.action_size == 3


def test_reset(env, key):
    state = env.reset(key)
    assert state.obs.shape == (11,)


def test_transition_fn_output_shape(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.5
    next_obs = env.transition_fn(state.obs, action)
    assert next_obs.shape == state.obs.shape


def test_transition_fn_differentiable_action(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3

    grad_fn = jax.grad(lambda a: jnp.sum(env.transition_fn(state.obs, a)))
    grads = grad_fn(action)
    assert grads.shape == action.shape
    assert not jnp.any(jnp.isnan(grads))


def test_transition_fn_differentiable_obs(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3

    grad_fn = jax.grad(lambda o: jnp.sum(env.transition_fn(o, action)))
    grads = grad_fn(state.obs)
    assert grads.shape == state.obs.shape
    assert not jnp.any(jnp.isnan(grads))


def test_transition_fn_joint_velocities_advance(env, key):
    """Positive action with zero joint velocities should produce non-zero qd_next."""
    state = env.reset(key)
    action = jnp.ones(env.action_size)
    next_obs = env.transition_fn(state.obs, action)
    # Joint velocities at obs[8:11] should change with action
    assert jnp.any(next_obs[8:11] != state.obs[8:11])


def test_reward_fn_shape(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3
    next_obs = env.transition_fn(state.obs, action)
    reward = env.reward_fn(state.obs, action, next_obs)
    assert reward.shape == ()


def test_reward_fn_differentiable(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3
    next_obs = env.transition_fn(state.obs, action)

    grad_action = jax.grad(lambda a: jnp.sum(env.reward_fn(state.obs, a, next_obs)))(
        action
    )
    assert not jnp.any(jnp.isnan(grad_action))


def test_wrapper_exposes_methods(wrapped_env, key):
    """BraxToGymnaxWrapper.transition_fn / reward_fn should pass through."""
    obs, state = wrapped_env.reset(key)
    action = jnp.ones(wrapped_env.action_size) * 0.3
    next_obs = wrapped_env.transition_fn(obs, action)
    assert next_obs.shape == obs.shape
    reward = wrapped_env.reward_fn(obs, action, next_obs)
    assert reward.shape == ()


def test_jacobian_norm_finite(env, key):
    """Frobenius norm of the obs→next_obs Jacobian must be finite (sanity for gating)."""
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3

    J = jax.jacfwd(lambda o: env.transition_fn(o, action))(state.obs)
    norm = jnp.linalg.norm(J, "fro")
    assert jnp.isfinite(norm)
    # Should be non-trivial — there's actual dynamics
    assert norm > 0.0
