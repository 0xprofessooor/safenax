"""MountainCarContinuous-v0 mirrored from gymnax with differentiable transition_fn / reward_fn."""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    position: jax.Array
    velocity: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: float = -1.0
    max_action: float = 1.0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025
    max_steps_in_episode: int = 999


class MountainCarContinuousV0(environment.Environment[EnvState, EnvParams]):
    """MountainCarContinuous-v0 with differentiable transition_fn / reward_fn.

    Dynamics mirror gymnax exactly. The goal-bonus term in the reward is an
    indicator (discontinuous): its gradient is zero almost everywhere, so the
    Sobolev gradient signal will be uninformative w.r.t. the bonus, but the
    smooth -0.1·action² penalty contributes a usable gradient.
    """

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "MountainCarContinuous-v0"

    @property
    def num_actions(self) -> int:
        return 1

    def reward_fn(
        self,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        params: Optional[EnvParams] = None,
    ) -> jax.Array:
        """Reward computed from the next position/velocity (matches gymnax)."""
        if params is None:
            params = self.default_params
        next_position = next_obs[..., 0]
        next_velocity = next_obs[..., 1]
        reached_goal = (next_position >= params.goal_position) & (
            next_velocity >= params.goal_velocity
        )
        reward = -0.1 * jnp.sum(action**2, axis=-1) + 100.0 * reached_goal.astype(
            jnp.float32
        )
        return reward

    def transition_fn(
        self,
        obs: jax.Array,
        action: jax.Array,
        params: Optional[EnvParams] = None,
    ) -> jax.Array:
        """Differentiable f(obs, action) -> next_obs."""
        if params is None:
            params = self.default_params
        position = obs[..., 0]
        velocity = obs[..., 1]
        force = jnp.clip(action, params.min_action, params.max_action).squeeze()

        new_velocity = (
            velocity + force * params.power - jnp.cos(3.0 * position) * params.gravity
        )
        new_velocity = jnp.clip(new_velocity, -params.max_speed, params.max_speed)
        new_position = position + new_velocity
        new_position = jnp.clip(new_position, params.min_position, params.max_position)
        # Zero velocity if we hit goal moving in the wrong direction.
        new_velocity = new_velocity * (
            1.0
            - ((new_position >= params.goal_position) & (new_velocity < 0)).astype(
                new_velocity.dtype
            )
        )

        return jnp.stack([new_position, new_velocity], axis=-1)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        force = jnp.clip(action, params.min_action, params.max_action)
        velocity = (
            state.velocity
            + force * params.power
            - jnp.cos(3.0 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position >= params.goal_position) * (velocity < 0))

        reward = -0.1 * action**2 + 100 * (
            (position >= params.goal_position) * (velocity >= params.goal_velocity)
        )
        reward = reward.squeeze()

        state = EnvState(
            position=position.squeeze(),
            velocity=velocity.squeeze(),
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        init_position = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_position, velocity=jnp.array(0.0), time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return jnp.array([state.position, state.velocity]).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        reached_goal = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )
        out_of_time = state.time >= params.max_steps_in_episode
        return jnp.logical_or(reached_goal, out_of_time).squeeze()

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(low=params.min_action, high=params.max_action, shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array([params.min_position, -params.max_speed], dtype=jnp.float32)
        high = jnp.array([params.max_position, params.max_speed], dtype=jnp.float32)
        return spaces.Box(low, high, shape=(2,), dtype=jnp.float32)
