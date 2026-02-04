#!/usr/bin/env python3
"""
Training and evaluation script for IQN-MPC portfolio optimization.

This script:
1. Creates a GARCH portfolio environment
2. Collects transition data from random/heuristic policies
3. Trains the IQN dynamics model
4. Evaluates the MPC solver against baselines
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
import argparse
from typing import Tuple
import matplotlib.pyplot as plt

from safenax.portfolio_optimization.po_garch import (
    PortfolioOptimizationGARCH,
    GARCHParams,
    EnvState,
)
from safenax.planning.iqn_dynamics import IQNDynamicsModel, IQNConfig
from safenax.planning.mpc_solver import MPCSolver, MPCConfig


def collect_transitions(
    env: PortfolioOptimizationGARCH,
    rng: jax.Array,
    num_episodes: int = 100,
    policy: str = "random",
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Collect (state, action, next_state) transitions from environment.
    
    Args:
        env: Portfolio environment
        rng: Random key
        num_episodes: Number of episodes to collect
        policy: "random" or "uniform" (equal weights)
    
    Returns:
        states: (N, state_dim)
        actions: (N, action_dim)
        next_states: (N, state_dim)
    """
    params = env.default_params
    max_steps = params.max_steps
    
    all_states = []
    all_actions = []
    all_next_states = []
    
    for ep in tqdm(range(num_episodes), desc="Collecting data"):
        rng, reset_rng, ep_rng = random.split(rng, 3)
        obs, state = env.reset(reset_rng, params)
        
        for t in range(max_steps - 1):
            rng, action_rng, step_rng = random.split(rng, 3)
            
            # Generate action based on policy
            if policy == "random":
                action = random.normal(action_rng, (env.num_assets + 1,))
            elif policy == "uniform":
                action = jnp.ones(env.num_assets + 1)  # Equal weights after softmax
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            # Store state and action
            all_states.append(np.array(obs))
            all_actions.append(np.array(jax.nn.softmax(action)))
            
            # Step environment
            next_obs, next_state, reward, done, info = env.step_env(
                step_rng, state, action, params
            )
            
            # Store next state
            all_next_states.append(np.array(next_obs))
            
            if done:
                break
            
            obs = next_obs
            state = next_state
    
    return (
        jnp.array(all_states),
        jnp.array(all_actions),
        jnp.array(all_next_states),
    )


def train_iqn(
    model: IQNDynamicsModel,
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    batch_size: int = 256,
    num_epochs: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Train IQN dynamics model on collected transitions.
    
    Args:
        model: IQN model to train
        states: Training states
        actions: Training actions
        next_states: Training next states
        batch_size: Batch size
        num_epochs: Number of epochs
        verbose: Print progress
    
    Returns:
        training_history: Dictionary with loss curves
    """
    num_samples = len(states)
    num_batches = num_samples // batch_size
    
    history = {"loss": [], "mae": []}
    
    for epoch in range(num_epochs):
        # Shuffle data
        perm = np.random.permutation(num_samples)
        epoch_losses = []
        epoch_maes = []
        
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            
            metrics = model.train_step(
                states[idx],
                actions[idx],
                next_states[idx],
            )
            
            epoch_losses.append(float(metrics["loss"]))
            epoch_maes.append(float(metrics["mae"]))
        
        avg_loss = np.mean(epoch_losses)
        avg_mae = np.mean(epoch_maes)
        history["loss"].append(avg_loss)
        history["mae"].append(avg_mae)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, mae={avg_mae:.6f}")
    
    return history


def evaluate_quantile_calibration(
    model: IQNDynamicsModel,
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    num_quantiles: int = 9,
) -> dict:
    """
    Evaluate how well the learned quantiles are calibrated.
    
    For well-calibrated quantiles, the fraction of actual values
    below the predicted τ-quantile should be approximately τ.
    
    Args:
        model: Trained IQN model
        states: Test states
        actions: Test actions
        next_states: Actual next states
        num_quantiles: Number of quantile levels to test
    
    Returns:
        calibration_results: Dictionary with calibration metrics
    """
    taus = jnp.linspace(0.1, 0.9, num_quantiles)
    
    calibration = {}
    for tau in taus:
        tau_arr = jnp.ones((len(states), 1)) * tau
        predicted = model.predict_quantiles(states, actions, tau_arr)
        predicted = predicted.squeeze(1)  # (N, state_dim)
        
        # Fraction of actual values below predicted quantile
        below = (next_states < predicted).mean(axis=0)  # Per dimension
        calibration[f"tau_{tau:.1f}"] = {
            "expected": float(tau),
            "actual_mean": float(below.mean()),
            "actual_per_dim": np.array(below),
        }
    
    return calibration


def evaluate_mpc(
    env: PortfolioOptimizationGARCH,
    mpc_solver: MPCSolver,
    rng: jax.Array,
    num_episodes: int = 10,
    baseline_policy: str = "uniform",
) -> dict:
    """
    Evaluate MPC solver against baseline.
    
    Args:
        env: Environment
        mpc_solver: Trained MPC solver
        rng: Random key
        num_episodes: Number of evaluation episodes
        baseline_policy: Baseline to compare against
    
    Returns:
        results: Dictionary with MPC and baseline results
    """
    params = env.default_params
    max_steps = min(50, params.max_steps)  # Limit for evaluation speed
    
    mpc_returns = []
    baseline_returns = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        rng, reset_rng = random.split(rng)
        obs, state = env.reset(reset_rng, params)
        
        # Initial weights (all cash)
        current_weights = jnp.zeros(env.num_assets + 1).at[0].set(1.0)
        portfolio_value = params.initial_cash
        
        mpc_total = 0.0
        baseline_total = 0.0
        
        for t in range(max_steps):
            rng, mpc_rng, step_rng1, step_rng2 = random.split(rng, 4)
            
            # MPC action
            try:
                mpc_weights, _ = mpc_solver.solve(
                    obs, portfolio_value, current_weights, mpc_rng, verbose=False
                )
                mpc_action = jnp.log(mpc_weights + 1e-8)  # Convert to logits
            except:
                mpc_action = jnp.zeros(env.num_assets + 1)
            
            # Baseline action
            if baseline_policy == "uniform":
                baseline_action = jnp.ones(env.num_assets + 1)
            elif baseline_policy == "cash":
                baseline_action = jnp.array([10.0] + [-10.0] * env.num_assets)
            else:
                baseline_action = jnp.zeros(env.num_assets + 1)
            
            # Step MPC
            _, mpc_state, mpc_reward, done1, _ = env.step_env(
                step_rng1, state, mpc_action, params
            )
            mpc_total += float(mpc_reward)
            
            # Step baseline (from same state)
            _, baseline_state, baseline_reward, done2, _ = env.step_env(
                step_rng2, state, baseline_action, params
            )
            baseline_total += float(baseline_reward)
            
            if done1 or done2:
                break
            
            # Update for next step (use MPC trajectory)
            obs, _ = env.get_obs_easy(mpc_state, params), None
            state = mpc_state
            current_weights = jax.nn.softmax(mpc_action)
            portfolio_value = float(mpc_state.total_value)
        
        mpc_returns.append(mpc_total)
        baseline_returns.append(baseline_total)
    
    return {
        "mpc_returns": np.array(mpc_returns),
        "baseline_returns": np.array(baseline_returns),
        "mpc_mean": np.mean(mpc_returns),
        "mpc_std": np.std(mpc_returns),
        "baseline_mean": np.mean(baseline_returns),
        "baseline_std": np.std(baseline_returns),
    }


def plot_results(
    training_history: dict,
    calibration: dict,
    eval_results: dict,
    save_path: str = None,
):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    axes[0, 0].plot(training_history["loss"], label="Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("IQN Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training MAE
    axes[0, 1].plot(training_history["mae"], label="MAE", color="orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].set_title("IQN Training MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calibration plot
    taus = []
    actuals = []
    for key, val in calibration.items():
        taus.append(val["expected"])
        actuals.append(val["actual_mean"])
    
    axes[1, 0].scatter(taus, actuals, s=100, label="Actual")
    axes[1, 0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[1, 0].set_xlabel("Expected quantile (τ)")
    axes[1, 0].set_ylabel("Actual fraction below")
    axes[1, 0].set_title("Quantile Calibration")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Evaluation comparison
    x = np.arange(2)
    means = [eval_results["mpc_mean"], eval_results["baseline_mean"]]
    stds = [eval_results["mpc_std"], eval_results["baseline_std"]]
    
    axes[1, 1].bar(x, means, yerr=stds, capsize=5, color=["blue", "gray"])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["MPC", "Baseline"])
    axes[1, 1].set_ylabel("Total Log Return")
    axes[1, 1].set_title("MPC vs Baseline Performance")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train IQN-MPC for portfolio optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=100, help="Data collection episodes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--save_fig", type=str, default=None, help="Path to save figure")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IQN-MPC Portfolio Optimization Training")
    print("=" * 60)
    
    # Initialize
    rng = random.PRNGKey(args.seed)
    
    # Create GARCH environment
    print("\n1. Creating GARCH environment...")
    rng, env_rng = random.split(rng)
    garch_params = {
        "BTC": GARCHParams(
            mu=5e-3,
            omega=1e-4,
            alpha=jnp.array([0.165]),
            beta=jnp.array([0.8]),
            initial_price=1.0,
        ),
        "APPL": GARCHParams(
            mu=3e-3,
            omega=1e-5,
            alpha=jnp.array([0.15]),
            beta=jnp.array([0.5]),
            initial_price=1.0,
        ),
    }
    
    env = PortfolioOptimizationGARCH(
        env_rng,
        garch_params,
        num_trajectories=1000,
        num_samples=500,
    )
    print(f"   Assets: {env.asset_names}")
    print(f"   State dim: {env.observation_space(env.default_params).shape}")
    print(f"   Action dim: {env.action_space(env.default_params).shape}")
    
    # Collect data
    print(f"\n2. Collecting {args.num_episodes} episodes of transition data...")
    rng, collect_rng = random.split(rng)
    states, actions, next_states = collect_transitions(
        env, collect_rng, num_episodes=args.num_episodes, policy="random"
    )
    print(f"   Collected {len(states)} transitions")
    
    # Split train/test
    split_idx = int(0.9 * len(states))
    train_states, test_states = states[:split_idx], states[split_idx:]
    train_actions, test_actions = actions[:split_idx], actions[split_idx:]
    train_next_states, test_next_states = next_states[:split_idx], next_states[split_idx:]
    print(f"   Train: {len(train_states)}, Test: {len(test_states)}")
    
    # Create IQN model
    print("\n3. Creating IQN dynamics model...")
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    iqn_config = IQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=64,
        num_cosine_features=64,
        num_layers=3,
        num_quantile_samples=32,
        learning_rate=1e-3,
        huber_kappa=1.0,
    )
    
    rng, model_rng = random.split(rng)
    iqn_model = IQNDynamicsModel(iqn_config, model_rng)
    print(f"   Config: {iqn_config}")
    
    # Train IQN
    print(f"\n4. Training IQN for {args.num_epochs} epochs...")
    training_history = train_iqn(
        iqn_model,
        train_states,
        train_actions,
        train_next_states,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        verbose=args.verbose,
    )
    print(f"   Final loss: {training_history['loss'][-1]:.6f}")
    print(f"   Final MAE: {training_history['mae'][-1]:.6f}")
    
    # Evaluate calibration
    print("\n5. Evaluating quantile calibration...")
    calibration = evaluate_quantile_calibration(
        iqn_model, test_states, test_actions, test_next_states
    )
    
    cal_errors = []
    for key, val in calibration.items():
        error = abs(val["expected"] - val["actual_mean"])
        cal_errors.append(error)
        print(f"   τ={val['expected']:.1f}: expected={val['expected']:.2f}, "
              f"actual={val['actual_mean']:.2f}, error={error:.3f}")
    print(f"   Mean calibration error: {np.mean(cal_errors):.4f}")
    
    # Create MPC solver
    print("\n6. Creating MPC solver...")
    mpc_config = MPCConfig(
        horizon=args.horizon,
        num_scenarios=32,
        num_optimization_steps=50,
        learning_rate=0.1,
        var_alpha=0.05,
        var_threshold=None,  # No VaR constraint for now
        linear_cost=0.001,
        entropy_weight=0.01,
    )
    
    mpc_solver = MPCSolver(
        config=mpc_config,
        dynamics_model=iqn_model,
        num_assets=env.num_assets + 1,
    )
    print(f"   Horizon: {mpc_config.horizon}")
    print(f"   Scenarios: {mpc_config.num_scenarios}")
    
    # Evaluate MPC
    print(f"\n7. Evaluating MPC ({args.eval_episodes} episodes)...")
    rng, eval_rng = random.split(rng)
    eval_results = evaluate_mpc(
        env, mpc_solver, eval_rng,
        num_episodes=args.eval_episodes,
        baseline_policy="uniform",
    )
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"   MPC Return:      {eval_results['mpc_mean']:.4f} ± {eval_results['mpc_std']:.4f}")
    print(f"   Baseline Return: {eval_results['baseline_mean']:.4f} ± {eval_results['baseline_std']:.4f}")
    improvement = (eval_results['mpc_mean'] - eval_results['baseline_mean']) / abs(eval_results['baseline_mean']) * 100
    print(f"   Improvement:     {improvement:.1f}%")
    
    # Plot results
    print("\n8. Plotting results...")
    plot_results(training_history, calibration, eval_results, save_path=args.save_fig)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
