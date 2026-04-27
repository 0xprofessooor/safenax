import jax
import jax.numpy as jnp
import pytest

from safenax import MountainCarContinuousV0


@pytest.fixture
def env():
    return MountainCarContinuousV0()


@pytest.fixture
def params(env):
    return env.default_params


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_obs_shape(env, params, key):
    obs, state = env.reset(key, params)
    assert obs.shape == (2,)
    # initial position in [-0.6, -0.4], velocity = 0
    assert -0.6 <= obs[0] <= -0.4
    assert obs[1] == 0.0


def test_transition_matches_step(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([0.5])

    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)

    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-6), (
        f"transition_fn={next_obs_pred} step={next_obs_step}"
    )


def test_transition_matches_step_clipped_action(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([5.0])  # exceeds max_action=1.0
    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)
    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-6)


def test_transition_matches_step_neg_action(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([-0.7])
    next_obs_step, _, _, _, _ = env.step(key, state, action, params)
    next_obs_pred = env.transition_fn(obs, action, params)
    assert jnp.allclose(next_obs_pred, next_obs_step, atol=1e-6)


def test_reward_matches_step_off_goal(env, params, key):
    obs, state = env.reset(key, params)
    action = jnp.array([0.3])
    next_obs, _, reward_step, _, _ = env.step(key, state, action, params)
    reward_pred = env.reward_fn(obs, action, next_obs, params)
    assert jnp.allclose(reward_pred, reward_step, atol=1e-6)


def test_reward_matches_step_at_goal(env, params, key):
    """Near-goal state: reward_fn (using transition_fn output, pre auto-reset) must match step reward."""
    from safenax.mountain_car.mountain_car_continuous_v0 import EnvState

    state = EnvState(position=jnp.array(0.44), velocity=jnp.array(0.06), time=0)
    obs = env.get_obs(state)
    action = jnp.array([1.0])
    # gymnax `step` auto-resets on terminal, so its returned next_obs is the
    # reset state. The Sobolev pipeline uses transition_fn directly, which
    # does NOT auto-reset — so we compare against that.
    next_obs_pred = env.transition_fn(obs, action, params)
    _, _, reward_step, _, _ = env.step(key, state, action, params)
    reward_pred = env.reward_fn(obs, action, next_obs_pred, params)
    assert jnp.allclose(reward_pred, reward_step, atol=1e-6)
    assert reward_pred > 50.0


def test_position_clipping(env, params):
    """Position must respect min/max bounds even with extreme actions."""
    from safenax.mountain_car.mountain_car_continuous_v0 import EnvState

    state = EnvState(
        position=jnp.array(params.min_position), velocity=jnp.array(0.0), time=0
    )
    obs = env.get_obs(state)
    action = jnp.array([-1.0])
    next_obs = env.transition_fn(obs, action, params)
    assert next_obs[0] >= params.min_position - 1e-6


def test_transition_differentiable(env, params, key):
    obs, _ = env.reset(key, params)
    action = jnp.array([0.5])

    grad_action = jax.grad(lambda a: jnp.sum(env.transition_fn(obs, a, params)))(action)
    assert not jnp.any(jnp.isnan(grad_action))

    grad_obs = jax.grad(lambda o: jnp.sum(env.transition_fn(o, action, params)))(obs)
    assert not jnp.any(jnp.isnan(grad_obs))


def test_reward_differentiable(env, params, key):
    obs, _ = env.reset(key, params)
    action = jnp.array([0.5])
    next_obs = env.transition_fn(obs, action, params)

    grad_action = jax.grad(lambda a: jnp.sum(env.reward_fn(obs, a, next_obs, params)))(
        action
    )
    assert not jnp.any(jnp.isnan(grad_action))


def test_no_params_call(env, key):
    obs, _ = env.reset(key, env.default_params)
    action = jnp.array([0.5])
    next_obs = env.transition_fn(obs, action)
    assert next_obs.shape == (2,)
    reward = env.reward_fn(obs, action, next_obs)
    assert reward.shape == ()


def test_rollout_consistency(env, params, key):
    obs, state = env.reset(key, params)
    K = 20
    actions = jax.random.uniform(key, (K, 1), minval=-1.0, maxval=1.0)

    pred_obs = obs
    for k in range(K):
        pred_obs = env.transition_fn(pred_obs, actions[k], params)

    cur_state = state
    for k in range(K):
        _, cur_state, _, _, _ = env.step(key, cur_state, actions[k], params)

    real_obs = env.get_obs(cur_state)
    assert jnp.allclose(pred_obs, real_obs, atol=1e-5)
