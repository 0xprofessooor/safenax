from typing import Optional
from brax import envs, State
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
import jax
from jax import numpy as jnp
from gymnax.environments import spaces, EnvParams
from chex import PRNGKey


class BraxToGymnaxWrapper:
    def __init__(
        self, env_name: str, episode_length: int = 1000, backend: str = "positional"
    ):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key: PRNGKey, params: Optional[EnvParams] = None):
        state = self._env.reset(key)
        return state.obs, state

    def step(
        self,
        key: PRNGKey,
        state: State,
        action: jax.Array,
        params: Optional[EnvParams] = None,
    ):
        next_state = self._env.step(state, action)
        # Return a copy of info dict to prevent downstream wrappers from mutating state.info
        info_copy = {**next_state.info}
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done > 0.5,
            info_copy,
        )

    def observation_space(self, params: Optional[EnvParams] = None):
        # Get actual observation spec from Brax if available
        obs_spec = getattr(self._env, "observation_spec", None)
        if obs_spec is not None:
            return spaces.Box(
                low=obs_spec.minimum,
                high=obs_spec.maximum,
                shape=(self._env.observation_size,),
            )
        # Fallback to unbounded
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params: Optional[EnvParams] = None):
        # Get actual action spec from Brax
        action_spec = getattr(self._env.sys, "actuator", None)
        if action_spec is not None and hasattr(action_spec, "ctrl_range"):
            ctrl_range = action_spec.ctrl_range
            return spaces.Box(
                low=ctrl_range[:, 0],
                high=ctrl_range[:, 1],
                shape=(self._env.action_size,),
            )
        # Fallback to [-1, 1]
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
