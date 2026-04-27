"""Pendulum-v1 mirrored from gymnax with differentiable transition_fn / reward_fn."""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    theta: jax.Array
    theta_dot: jax.Array
    last_u: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0  # noqa: E741
    max_steps_in_episode: int = 200


def angle_normalize(x: jax.Array) -> jax.Array:
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


class PendulumV1(environment.Environment[EnvState, EnvParams]):
    """Pendulum-v1 with differentiable transition_fn / reward_fn for Sobolev PPO.

    Dynamics mirror gymnax exactly so existing baselines transfer. The
    observation is [cos(theta), sin(theta), theta_dot] and the transition
    is written directly in obs-space using the trig addition identity so
    no atan2 round-trip is needed.
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (3,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "Pendulum-v1"

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
        """Differentiable reward computed from the current obs (matches gymnax).

        Gymnax's step_env evaluates reward at the *pre-integration* state, so we
        recover theta from obs = [cos(theta), sin(theta), theta_dot] via atan2.
        """
        if params is None:
            params = self.default_params
        u = jnp.clip(action, -params.max_torque, params.max_torque).squeeze()
        c = obs[..., 0]
        s = obs[..., 1]
        theta_dot = obs[..., 2]
        theta = jnp.arctan2(s, c)
        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * (u**2))
        return reward

    def transition_fn(
        self,
        obs: jax.Array,
        action: jax.Array,
        params: Optional[EnvParams] = None,
    ) -> jax.Array:
        """Differentiable f(obs, action) -> next_obs.

        Uses the trig addition identity to advance cos/sin without atan2.
        """
        if params is None:
            params = self.default_params
        u = jnp.clip(action, -params.max_torque, params.max_torque).squeeze()
        c = obs[..., 0]
        s = obs[..., 1]
        theta_dot = obs[..., 2]

        new_theta_dot = (
            theta_dot
            + (
                3.0 * params.g / (2.0 * params.l) * s
                + 3.0 / (params.m * params.l**2) * u
            )
            * params.dt
        )
        new_theta_dot = jnp.clip(new_theta_dot, -params.max_speed, params.max_speed)

        delta = new_theta_dot * params.dt
        cd = jnp.cos(delta)
        sd = jnp.sin(delta)
        new_c = c * cd - s * sd
        new_s = s * cd + c * sd

        return jnp.stack([new_c, new_s, new_theta_dot], axis=-1)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        u = jnp.clip(action, -params.max_torque, params.max_torque)
        reward = -(
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot**2
            + 0.001 * (u**2)
        )
        reward = reward.squeeze()

        new_theta_dot = (
            state.theta_dot
            + (
                3.0 * params.g / (2.0 * params.l) * jnp.sin(state.theta)
                + 3.0 / (params.m * params.l**2) * u
            )
            * params.dt
        )
        new_theta_dot = jnp.clip(new_theta_dot, -params.max_speed, params.max_speed)
        new_theta = state.theta + new_theta_dot * params.dt

        state = EnvState(
            theta=new_theta.squeeze(),
            theta_dot=new_theta_dot.squeeze(),
            last_u=u.reshape(),
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
        high = jnp.array([jnp.pi, 1.0])
        sampled = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(
            theta=sampled[0],
            theta_dot=sampled[1],
            last_u=jnp.array(0.0),
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return jnp.array(
            [jnp.cos(state.theta), jnp.sin(state.theta), state.theta_dot]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.array(state.time >= params.max_steps_in_episode)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)
