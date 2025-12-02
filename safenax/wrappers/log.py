from functools import partial
from typing import Any
import jax
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_cost_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_cost_returns: float
    returned_episode_lengths: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns, costs and lengths."""

    #   def __init__(self, env: environment.Environment):
    #     super().__init__(env)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, bool, dict[Any, Any]]:
        """Step the environment.


        Args:
          key: Pkey key.
          state: The current state of the environment.
          action: The action to take.
          params: The parameters of the environment.


        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_cost_return = state.episode_cost_returns + info.get("cost", 0.0)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_cost_returns=new_episode_cost_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_cost_returns=state.returned_episode_cost_returns
            * (1 - done)
            + new_episode_cost_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode_cost_returns"] = state.returned_episode_cost_returns
        info["returned_episode"] = done
        return obs, state, reward, done, info
