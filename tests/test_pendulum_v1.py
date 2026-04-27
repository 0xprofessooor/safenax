import jax
import jax.numpy as jnp
import pytest

from safenax import PendulumV1


@pytest.fixture
def env():
    return PendulumV1()


@pytest.fixture
def params(env):
    return env.default_params


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_obs_shape(env, params, key):
    obs, state = env.reset(key, params)
    assert obs.shape == (3,)
    assert jnp.isclose(obs[0] ** 2 + obs[1] ** 2, 1.0)


def test_transition_matches_step(env, params, key):
    """transition_fn must produce the same next obs as step_env (deterministic)."""
    obs, state = env.reset(key, params)
    action = jnp.array([0.5])

    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)

    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-5), (
        f"transition_fn={next_obs_pred} step={next_obs_step}"
    )


def test_transition_matches_step_zero_action(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([0.0])
    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)
    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-5)


def test_transition_matches_step_max_action(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([10.0])  # exceeds max_torque, will be clipped
    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)
    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-5)


def test_reward_matches_step(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([0.3])
    next_obs, _, reward_step, _, _ = env.step(key, state, action, params)
    reward_pred = env.reward_fn(obs, action, next_obs, params)
    assert jnp.allclose(reward_pred, reward_step, atol=1e-5), (
        f"reward_fn={reward_pred} step={reward_step}"
    )


def test_obs_norm_preserved(env, params, key):
    """After a step, cos²+sin² should still equal 1 (within tolerance)."""
    obs, state = env.reset(key, params)
    action = jnp.array([1.0])
    next_obs = env.transition_fn(obs, action, params)
    norm_sq = next_obs[0] ** 2 + next_obs[1] ** 2
    assert jnp.isclose(norm_sq, 1.0, atol=1e-5), f"cos²+sin²={norm_sq}"


def test_transition_differentiable(env, params, key):
    obs, _ = env.reset(key, params)
    action = jnp.array([0.5])

    grad_action = jax.grad(lambda a: jnp.sum(env.transition_fn(obs, a, params)))(action)
    assert grad_action.shape == action.shape
    assert not jnp.any(jnp.isnan(grad_action))

    grad_obs = jax.grad(lambda o: jnp.sum(env.transition_fn(o, action, params)))(obs)
    assert grad_obs.shape == obs.shape
    assert not jnp.any(jnp.isnan(grad_obs))


def test_reward_differentiable(env, params, key):
    obs, _ = env.reset(key, params)
    action = jnp.array([0.5])
    next_obs = env.transition_fn(obs, action, params)

    grad_action = jax.grad(lambda a: jnp.sum(env.reward_fn(obs, a, next_obs, params)))(
        action
    )
    assert not jnp.any(jnp.isnan(grad_action))

    grad_obs = jax.grad(lambda o: jnp.sum(env.reward_fn(o, action, next_obs, params)))(
        obs
    )
    assert not jnp.any(jnp.isnan(grad_obs))


def test_transition_fn_no_params(env, key):
    """sobolev/ppo.py calls transition_fn / reward_fn without params."""
    obs, _ = env.reset(key, env.default_params)
    action = jnp.array([0.5])
    next_obs = env.transition_fn(obs, action)
    assert next_obs.shape == (3,)

    reward = env.reward_fn(obs, action, next_obs)
    assert reward.shape == ()


def test_rollout_consistency(env, params, key):
    """K-step rollout via transition_fn must match K-step rollout via step_env."""
    obs, state = env.reset(key, params)

    K = 10
    rng = key
    actions = jax.random.uniform(rng, (K, 1), minval=-1.0, maxval=1.0)

    pred_obs = obs
    for k in range(K):
        pred_obs = env.transition_fn(pred_obs, actions[k], params)

    cur_state = state
    for k in range(K):
        _, cur_state, _, _, _ = env.step(key, cur_state, actions[k], params)

    real_obs = env.get_obs(cur_state)
    assert jnp.allclose(pred_obs, real_obs, atol=1e-4), (
        f"K-step pred={pred_obs} real={real_obs}"
    )
