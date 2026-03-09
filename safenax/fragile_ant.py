import jax
import jax.numpy as jnp
from brax import envs, State
from brax.envs.ant import Ant


class FragileAnt(Ant):
    """
    Ant with a 'Fragile Gearbox' constraint.

    Modifications:
    1. Stochasticity: Adds Gaussian noise to actions to simulate motor imperfection.
    2. Cost Signal: Measures cumulative mechanical shock (squared angular acceleration).
    """

    def __init__(self, cost_budget: float, noise_scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale
        self.cost_budget = cost_budget

    @property
    def name(self) -> str:
        return "FragileAnt"

    def cost_fn(
        self, obs: jax.Array, action: jax.Array, next_obs: jax.Array
    ) -> jax.Array:
        """
        Measures the absolute change in angular velocity (acceleration/impacts).
        """
        # Indices 19 through 26 hold the angular velocities of the 8 hinges
        current_joint_vels = obs[..., 19:27]
        next_joint_vels = next_obs[..., 19:27]

        # Acceleration = delta Velocity
        joint_accelerations = next_joint_vels - current_joint_vels

        # The cost is the total square shock experienced by all joints this step
        shock_cost = jnp.sum(jnp.square(joint_accelerations), axis=-1)

        return shock_cost

    def reward_fn(
        self, obs: jax.Array, action: jax.Array, next_obs: jax.Array
    ) -> jax.Array:
        """
        Analytic, fully differentiable reward function for Model-Based RL.
        Calculates the exact reward using only the observable state space.
        """
        # 1. Forward Reward
        # The physics engine calculated velocity as (x1 - x0) / dt.
        # The observation builder placed this exact value at index 13.
        forward_reward = 0.5 * (obs[..., 13] + next_obs[..., 13])

        # 2. Healthy Reward
        # Torso Z-coordinate is at index 0.
        z_pos = next_obs[..., 0]
        min_z, max_z = self._healthy_z_range

        is_healthy = jnp.where(z_pos < min_z, 0.0, 1.0)
        is_healthy = jnp.where(z_pos > max_z, 0.0, is_healthy)

        if self._terminate_when_unhealthy:
            healthy_reward = jnp.full_like(forward_reward, self._healthy_reward)
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # 3. Control Cost
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)

        # Contact cost is 0.0 in your current implementation
        contact_cost = 0.0

        # Total Expected Reward
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost

        return reward

    def step(self, state: State, action: jax.Array) -> State:
        current_budget_pct = state.obs[-1]
        current_budget = current_budget_pct * self.cost_budget

        # 1. HANDLE STOCHASTICITY
        # Generate Gaussian noise and add to action
        # This prevents the agent from perfectly memorizing a trajectory
        rng, noise_key = jax.random.split(state.info["rng"])
        noise = jax.random.normal(noise_key, shape=action.shape) * self.noise_scale
        noisy_action = action + noise

        # Clip action to valid range (usually -1, 1) so physics doesn't explode
        noisy_action = jnp.clip(noisy_action, -1.0, 1.0)

        # 2. PHYSICS STEP
        # Pass the noisy action to the physics engine
        next_state = super().step(state, noisy_action)

        # 3. CALCULATE COST (The Fragile Gearbox)
        cost = self.cost_fn(state.obs, noisy_action, next_state.obs)
        new_budget = current_budget - cost
        is_broken = new_budget <= 0.0
        new_budget = jnp.maximum(new_budget, 0.0)
        new_budget_pct = new_budget / self.cost_budget

        # 4. UPDATE STATE INFO
        # We must update the rng in the state info for the next step
        new_obs = jnp.concatenate([next_state.obs, jnp.array([new_budget_pct])])
        new_done = jnp.max(jnp.array([next_state.done, is_broken]))
        new_info = {
            **next_state.info,
            "rng": rng,
            "cost": cost,
            "cost_budget": new_budget,
        }

        return next_state.replace(obs=new_obs, done=new_done, info=new_info)

    def reset(self, rng: jax.Array) -> State:
        # Standard reset, but initialize the RNG key in info
        state = super().reset(rng)
        new_obs = jnp.concatenate([state.obs, jnp.array([1.0])])
        new_info = {
            **state.info,
            "rng": rng,
            "cost": jnp.array(0.0),
            "cost_budget": self.cost_budget,
        }
        return state.replace(obs=new_obs, info=new_info)


envs.register_environment("fragile_ant", FragileAnt)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper

    print("Testing FragileAnt with random policy...")

    # Run N episodes in parallel
    n_episodes = 1000
    episode_length = 1000

    # Create environment with better limit
    base_env = FragileAnt(cost_budget=100_000.0, noise_scale=0.1)
    env = LogWrapper(BraxToGymnaxWrapper(base_env))

    def single_episode(rng):
        """Run a single episode with random actions."""
        reset_rng, rollout_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, None)  # Use Gymnax API

        def step_fn(carry, _):
            obs, state, rng = carry
            action_rng, rng = jax.random.split(rng)
            action = env.action_space().sample(key=action_rng)
            next_obs, next_state, reward, done, info = env.step(
                action_rng, state, action, None
            )
            cost = info.get("cost", 0.0)
            # Return observations in the scan output
            return (next_obs, next_state, rng), (cost, reward, obs)

        final_carry, (costs, rewards, all_obs) = jax.lax.scan(
            step_fn, (obs, state, rollout_rng), None, length=episode_length
        )
        return costs, rewards, jnp.sum(costs), all_obs

    # Vectorize over episodes and JIT compile
    batched_rollout = jax.jit(jax.vmap(single_episode))

    print(f"\nRunning {n_episodes} episodes (JIT compiling on first run)...")

    master_rng = jax.random.PRNGKey(42)
    episode_rngs = jax.random.split(master_rng, n_episodes)

    all_costs, all_rewards, total_costs, all_obs = batched_rollout(episode_rngs)

    # Print statistics
    print(f"\nResults:")
    print(
        f"  Observation shape: {all_obs.shape}"
    )  # (n_episodes, episode_length, obs_dim)
    print(f"  Observation range: [{jnp.min(all_obs):.2f}, {jnp.max(all_obs):.2f}]")
    print(f"  Mean observation: {jnp.mean(all_obs):.4f} ± {jnp.std(all_obs):.4f}")
    print(
        f"  Mean episode reward: {jnp.mean(jnp.sum(all_rewards, axis=1)):.2f} ± {jnp.std(jnp.sum(all_rewards, axis=1)):.2f}"
    )
    print(
        f"  Mean episode cost: {jnp.mean(total_costs):.2f} ± {jnp.std(total_costs):.2f}"
    )
    print(f"  Timestep violation rate: {jnp.mean(all_costs > 0) * 100:.2f}%")
    print(f"  Episode violation rate: {jnp.mean(total_costs > 0) * 100:.2f}%")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Cost per timestep distribution
    ax = axes[0, 0]
    flat_costs = all_costs.flatten()
    ax.hist(flat_costs, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Cost per timestep")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Cost Distribution (Violation rate: {jnp.mean(flat_costs > 0) * 100:.1f}%)"
    )
    ax.axvline(0.5, color="red", linestyle="--", label="Cost = 1.0")
    ax.legend()

    # 2. Total cost per episode
    ax = axes[0, 1]
    ax.hist(total_costs, bins=30, alpha=0.7, edgecolor="black", color="orange")
    ax.set_xlabel("Total cost per episode")
    ax.set_ylabel("Frequency")
    ax.set_title("Total Episode Cost Distribution")
    ax.axvline(
        jnp.mean(total_costs),
        color="red",
        linestyle="--",
        label=f"Mean: {jnp.mean(total_costs):.1f}",
    )
    ax.legend()

    # 3. Cost over time (first 10 episodes)
    ax = axes[1, 0]
    for i in range(min(10, n_episodes)):
        ax.plot(all_costs[i], alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cost")
    ax.set_title("Cost over Time (first 10 episodes)")
    ax.set_ylim([-0.1, 1.1])

    # 4. Cumulative cost over time (mean across episodes)
    ax = axes[1, 1]
    mean_cumulative_cost = jnp.mean(jnp.cumsum(all_costs, axis=1), axis=0)
    std_cumulative_cost = jnp.std(jnp.cumsum(all_costs, axis=1), axis=0)
    timesteps = jnp.arange(episode_length)
    ax.plot(timesteps, mean_cumulative_cost, linewidth=2, label="Mean")
    ax.fill_between(
        timesteps,
        mean_cumulative_cost - std_cumulative_cost,
        mean_cumulative_cost + std_cumulative_cost,
        alpha=0.3,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Cost")
    ax.set_title("Cumulative Cost over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
