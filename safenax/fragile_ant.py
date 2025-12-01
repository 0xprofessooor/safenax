import jax
import jax.numpy as jnp
import chex
from brax import envs, State
from brax.envs.ant import Ant
from safenax.wrappers.brax import BraxToGymnaxWrapper


class FragileAnt(Ant):
    """
    Ant with a 'Fragile Gearbox' constraint.

    Modifications:
    1. Stochasticity: Adds Gaussian noise to actions to simulate motor imperfection.
    2. Cost Signal: Returns a cost of 1.0 if any joint velocity exceeds the 'gearbox_limit'.
    """

    def __init__(self, gearbox_limit: float = 2.0, noise_scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gearbox_limit = gearbox_limit
        self.noise_scale = noise_scale

    def step(self, state: State, action: jax.Array) -> State:
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

        # 3. CALCULATE VaR COST (The Fragile Gearbox)
        # We want to ignore the Torso velocity (running speed) and only penalize the limbs.
        pipeline_state = next_state.pipeline_state
        limb_velocities = pipeline_state.xd.vel[1:]  # Skip torso, get limb velocities

        # Check if ANY limb velocity exceeds the limit
        max_vel = jnp.max(jnp.abs(limb_velocities))

        # Binary Cost: 1.0 if limit violated, 0.0 otherwise
        is_violation = max_vel > self.gearbox_limit
        cost = jnp.where(is_violation, 1.0, 0.0)

        # 4. UPDATE STATE INFO
        # We must update the rng in the state info for the next step
        new_info = {**next_state.info, "rng": rng, "cost": cost}

        return next_state.replace(info=new_info)

    def reset(self, rng: chex.PRNGKey) -> State:
        # Standard reset, but initialize the RNG key in info
        state = super().reset(rng)
        new_info = {**state.info, "rng": rng, "cost": jnp.array(0.0)}
        return state.replace(info=new_info)


envs.register_environment("fragile_ant", FragileAnt)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing FragileAnt with random policy...")

    # Run N episodes in parallel
    n_episodes = 1000
    episode_length = 1000

    # Create environment with better limit
    env = BraxToGymnaxWrapper(env_name="fragile_ant", episode_length=episode_length)

    def single_episode(rng):
        """Run a single episode with random actions."""
        reset_rng, rollout_rng = jax.random.split(rng)
        initial_state = env._env.reset(reset_rng)

        def step_fn(carry, _):
            state, rng = carry
            action_rng, rng = jax.random.split(rng)
            action = env.action_space().sample(key=action_rng)
            next_state = env._env.step(state, action)
            max_vel = jnp.max(jnp.abs(next_state.pipeline_state.xd.vel[1:]))
            cost = next_state.info.get("cost", 0.0)
            reward = next_state.reward
            return (next_state, rng), (cost, reward, max_vel)

        final_carry, (costs, rewards, max_vels) = jax.lax.scan(
            step_fn, (initial_state, rollout_rng), None, length=episode_length
        )
        return costs, rewards, jnp.sum(costs), max_vels

    # Vectorize over episodes and JIT compile
    batched_rollout = jax.jit(jax.vmap(single_episode))

    print(f"\nRunning {n_episodes} episodes (JIT compiling on first run)...")

    master_rng = jax.random.PRNGKey(42)
    episode_rngs = jax.random.split(master_rng, n_episodes)

    all_costs, all_rewards, total_costs, max_vels = batched_rollout(episode_rngs)

    # Print statistics
    print(f"\nResults:")
    print(
        f"  Mean episode max velocity: {jnp.mean(max_vels):.2f} ± {jnp.std(max_vels):.2f}"
    )
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
