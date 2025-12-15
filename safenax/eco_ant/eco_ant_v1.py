import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import State
from brax.envs.ant import Ant


class EcoAntV1(Ant):
    """
    Ant with a 'low battery' constraint.

    Modifications:
    1. Stochasticity: Adds Gaussian noise to actions to simulate motor imperfection.
    2. Cost Signal: Returns a cost of 1.0 if the total energy used exceeds the 'battery_limit'.
    """

    def __init__(
        self, battery_limit: float = 500.0, noise_scale: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.battery_limit = battery_limit
        self.noise_scale = noise_scale

    @property
    def name(self) -> str:
        return "EcoAnt-v1"

    def step(self, state: State, action: jax.Array) -> State:
        # 1. RETRIEVE BATTERY FROM CURRENT OBSERVATION
        current_battery_pct = state.obs[-1]
        current_battery = current_battery_pct * self.battery_limit

        # 2. HANDLE STOCHASTICITY
        _, noise_key = jax.random.split(state.info["rng"])
        noise = jax.random.normal(noise_key, shape=action.shape) * self.noise_scale
        noisy_action = action + noise
        noisy_action = jnp.clip(noisy_action, -1.0, 1.0)

        # 3. CALCULATE ENERGY AND NEW BATTERY
        energy_used = jnp.sum(jnp.square(noisy_action)) * 0.5
        new_battery = current_battery - energy_used

        # Check constraints
        is_empty = new_battery <= 0.0
        new_battery = jnp.maximum(new_battery, 0.0)
        new_battery_pct = new_battery / self.battery_limit

        # 4. PHYSICS STEP
        next_state = super().step(state, noisy_action)

        # 5. Termination: OR with existing done condition
        new_done = jnp.max(jnp.array([next_state.done, is_empty]))

        # Cost Signal: 1.0 if battery died this step
        cost = jnp.where(is_empty, 1.0, 0.0)

        # Observation: Append the new battery level to the observation vector
        new_obs = jnp.concatenate([next_state.obs, jnp.array([new_battery_pct])])

        new_info = {
            **next_state.info,
            "rng": noise_key,
            "cost": cost,
            "battery": new_battery,
        }

        return next_state.replace(obs=new_obs, done=new_done, info=new_info)

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        # Append initial battery pct to observation
        new_obs = jnp.concatenate([state.obs, jnp.array([1.0])])

        # Initialize info
        new_info = {
            **state.info,
            "rng": rng,
            "cost": jnp.array(0.0),
            "battery": jnp.array(self.battery_limit),
        }

        return state.replace(obs=new_obs, info=new_info)


envs.register_environment(EcoAntV1.name, EcoAntV1)
