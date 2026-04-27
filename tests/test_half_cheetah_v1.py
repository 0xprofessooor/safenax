import jax
import jax.numpy as jnp
import pytest

from safenax import HalfCheetahV1
from safenax.wrappers import BraxToGymnaxWrapper


@pytest.fixture
def env():
    return HalfCheetahV1()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def wrapped_env(env):
    return BraxToGymnaxWrapper(env=env, episode_length=10)


def test_obs_size(env):
    assert env.observation_size == 17
    assert env.action_size == 6


def test_reset(env, key):
    state = env.reset(key)
    assert state.obs.shape == (17,)


def test_transition_fn_output_shape(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.5
    next_obs = env.transition_fn(state.obs, action)
    assert next_obs.shape == state.obs.shape


def test_transition_fn_differentiable_action(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3
    grads = jax.grad(lambda a: jnp.sum(env.transition_fn(state.obs, a)))(action)
    assert grads.shape == action.shape
    assert not jnp.any(jnp.isnan(grads))


def test_transition_fn_differentiable_obs(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3
    grads = jax.grad(lambda o: jnp.sum(env.transition_fn(o, action)))(state.obs)
    assert grads.shape == state.obs.shape
    assert not jnp.any(jnp.isnan(grads))


def test_joint_velocities_advance(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size)
    next_obs = env.transition_fn(state.obs, action)
    assert jnp.any(next_obs[11:17] != state.obs[11:17])


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
    grads = jax.grad(lambda a: jnp.sum(env.reward_fn(state.obs, a, next_obs)))(action)
    assert not jnp.any(jnp.isnan(grads))


def test_wrapper_exposes_methods(wrapped_env, key):
    obs, state = wrapped_env.reset(key)
    action = jnp.ones(wrapped_env.action_size) * 0.3
    next_obs = wrapped_env.transition_fn(obs, action)
    assert next_obs.shape == obs.shape
    reward = wrapped_env.reward_fn(obs, action, next_obs)
    assert reward.shape == ()


def test_jacobian_norm_finite(env, key):
    state = env.reset(key)
    action = jnp.ones(env.action_size) * 0.3
    J = jax.jacfwd(lambda o: env.transition_fn(o, action))(state.obs)
    norm = jnp.linalg.norm(J, "fro")
    assert jnp.isfinite(norm)
    assert norm > 0.0
