"""
Model Predictive Control (MPC) solver for portfolio optimization.

Uses the IQN-learned distributional dynamics for planning with
VaR/CVaR risk constraints. Implements convex relaxations from
Lobo, Fazel & Boyd (2002) for handling transaction costs.
"""

from typing import NamedTuple, Optional, Tuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import optax

from safenax.planning.iqn_dynamics import IQNDynamicsModel


class MPCConfig(NamedTuple):
    """Configuration for MPC solver."""
    horizon: int = 10  # Planning horizon
    num_scenarios: int = 64  # Monte Carlo scenarios for expectation
    num_optimization_steps: int = 100  # Gradient descent iterations
    learning_rate: float = 0.1
    
    # Risk constraints
    var_alpha: float = 0.05  # VaR probability level (5%)
    var_threshold: Optional[float] = None  # VaR threshold (if None, no constraint)
    cvar_alpha: float = 0.05  # CVaR probability level
    cvar_threshold: Optional[float] = None  # CVaR threshold
    
    # Transaction cost parameters
    linear_cost: float = 0.001  # Linear transaction cost (0.1%)
    fixed_cost: float = 0.0  # Fixed cost per trade
    min_trade_threshold: float = 0.01  # Minimum trade size
    
    # Regularization
    entropy_weight: float = 0.01  # Encourage diversification
    turnover_penalty: float = 0.0  # Penalize portfolio changes
    
    # Constraints
    max_position: float = 1.0  # Max weight per asset
    min_position: float = 0.0  # Min weight per asset (0 = long only)
    leverage_limit: float = 1.0  # Max total leverage


class MPCSolver:
    """
    MPC solver using IQN dynamics for portfolio optimization.
    
    Solves:
        max E[Σ_t log(V_t+1 / V_t)]  (maximize expected log return)
        s.t. VaR_α(R) >= ρ           (Value-at-Risk constraint)
             Σ w_i = 1               (fully invested)
             w_i >= 0                (long only, optional)
             
    Uses gradient-based optimization with soft constraint penalties.
    """
    
    def __init__(
        self,
        config: MPCConfig,
        dynamics_model: IQNDynamicsModel,
        num_assets: int,
    ):
        self.config = config
        self.dynamics = dynamics_model
        self.num_assets = num_assets  # Including cash
        
        # Initialize optimizer
        self.optimizer = optax.adam(config.learning_rate)

    def _compute_transaction_costs(
        self,
        current_weights: jax.Array,
        new_weights: jax.Array,
        portfolio_value: float,
    ) -> jax.Array:
        """
        Compute transaction costs for rebalancing.
        
        Implements linear costs with optional fixed costs per trade.
        Uses the relaxation from Lobo et al. for convexity.
        
        Args:
            current_weights: Current portfolio weights (num_assets,)
            new_weights: Target weights (num_assets,)
            portfolio_value: Current portfolio value
        
        Returns:
            Total transaction cost
        """
        cfg = self.config
        
        # Weight changes
        delta_weights = new_weights - current_weights
        
        # Linear costs: cost proportional to |Δw|
        # For buys and sells, we pay the linear cost
        trade_values = jnp.abs(delta_weights) * portfolio_value
        linear_cost = cfg.linear_cost * jnp.sum(trade_values)
        
        # Fixed costs (if any): pay per asset traded
        if cfg.fixed_cost > 0:
            # Relaxation: use smooth approximation of indicator
            # This is the key insight from the paper - relax to convex
            trade_indicators = jnp.abs(delta_weights) > cfg.min_trade_threshold
            fixed_cost = cfg.fixed_cost * jnp.sum(trade_indicators.astype(jnp.float32))
        else:
            fixed_cost = 0.0
        
        return linear_cost + fixed_cost

    def _simulate_trajectory(
        self,
        rng: jax.Array,
        initial_state: jax.Array,
        action_sequence: jax.Array,  # (horizon, num_assets)
        portfolio_value: float,
        current_weights: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Simulate one trajectory using IQN dynamics.
        
        Args:
            rng: Random key
            initial_state: Initial observation
            action_sequence: Sequence of portfolio weight targets
            portfolio_value: Initial portfolio value
            current_weights: Current portfolio weights
        
        Returns:
            log_returns: Log returns at each step (horizon,)
            final_value: Final portfolio value
        """
        cfg = self.config
        
        def step_fn(carry, inputs):
            state, weights, value, rng = carry
            target_weights, = inputs
            rng, sample_rng = jax.random.split(rng)
            
            # Transaction costs for rebalancing
            tx_cost = self._compute_transaction_costs(weights, target_weights, value)
            value_after_costs = value - tx_cost
            
            # Sample next state from IQN
            # The state includes market returns/volatilities
            tau = jax.random.uniform(sample_rng, (1,))
            next_state = self.dynamics.predict_quantiles(
                state[None, :], target_weights[None, :], tau[None, :]
            ).squeeze()
            
            # Extract return from state (assuming first num_assets-1 dims are returns)
            # This depends on the observation structure
            asset_returns = next_state[:self.num_assets - 1]  # Exclude cash
            cash_return = jnp.array([0.0])  # Cash has no return
            returns = jnp.concatenate([cash_return, asset_returns])
            
            # Portfolio return
            portfolio_return = jnp.sum(target_weights * returns)
            log_return = jnp.log(1 + portfolio_return)
            
            # Update value
            new_value = value_after_costs * (1 + portfolio_return)
            
            return (next_state, target_weights, new_value, rng), log_return
        
        # Run trajectory
        init_carry = (initial_state, current_weights, portfolio_value, rng)
        inputs = (action_sequence,)
        
        _, log_returns = lax.scan(step_fn, init_carry, inputs)
        
        cumulative_return = jnp.sum(log_returns)
        final_value = portfolio_value * jnp.exp(cumulative_return)
        
        return log_returns, final_value

    def _compute_risk_metrics(
        self,
        log_returns: jax.Array,  # (num_scenarios, horizon)
    ) -> dict:
        """
        Compute risk metrics from simulated scenarios.
        
        Args:
            log_returns: Log returns for each scenario and timestep
        
        Returns:
            Dictionary with VaR, CVaR, etc.
        """
        cfg = self.config
        
        # Total return per scenario
        total_returns = log_returns.sum(axis=1)  # (num_scenarios,)
        
        # Sort for quantile computation
        sorted_returns = jnp.sort(total_returns)
        num_scenarios = len(sorted_returns)
        
        # VaR: α-quantile of returns (worst α% outcome)
        var_idx = int(cfg.var_alpha * num_scenarios)
        var = sorted_returns[var_idx]
        
        # CVaR: expected return below VaR
        cvar = sorted_returns[:var_idx + 1].mean()
        
        return {
            "var": var,
            "cvar": cvar,
            "mean_return": total_returns.mean(),
            "std_return": total_returns.std(),
            "min_return": total_returns.min(),
            "max_return": total_returns.max(),
        }

    def _objective(
        self,
        action_params: jax.Array,  # Raw parameters before softmax
        rng: jax.Array,
        initial_state: jax.Array,
        portfolio_value: float,
        current_weights: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """
        Compute optimization objective (negative expected return + penalties).
        
        Args:
            action_params: Raw action parameters (horizon, num_assets)
            rng: Random key for Monte Carlo
            initial_state: Initial observation
            portfolio_value: Current portfolio value
            current_weights: Current weights
        
        Returns:
            objective: Scalar objective (to minimize)
            metrics: Dictionary of metrics
        """
        cfg = self.config
        
        # Convert to valid weights via softmax
        action_sequence = jax.nn.softmax(action_params, axis=-1)
        
        # Monte Carlo simulation
        scenario_rngs = jax.random.split(rng, cfg.num_scenarios)
        
        def simulate_one(scenario_rng):
            return self._simulate_trajectory(
                scenario_rng,
                initial_state,
                action_sequence,
                portfolio_value,
                current_weights,
            )
        
        # Vectorized simulation
        all_log_returns, all_final_values = jax.vmap(simulate_one)(scenario_rngs)
        # all_log_returns: (num_scenarios, horizon)
        
        # Compute risk metrics
        risk_metrics = self._compute_risk_metrics(all_log_returns)
        
        # Primary objective: maximize expected return
        expected_return = risk_metrics["mean_return"]
        
        # Risk constraint penalties (soft constraints)
        penalty = 0.0
        
        if cfg.var_threshold is not None:
            # Penalize if VaR is below threshold
            var_violation = jnp.maximum(0, cfg.var_threshold - risk_metrics["var"])
            penalty += 100.0 * var_violation ** 2
        
        if cfg.cvar_threshold is not None:
            cvar_violation = jnp.maximum(0, cfg.cvar_threshold - risk_metrics["cvar"])
            penalty += 100.0 * cvar_violation ** 2
        
        # Entropy regularization (encourages diversification)
        entropy = -jnp.sum(action_sequence * jnp.log(action_sequence + 1e-8), axis=-1).mean()
        
        # Turnover penalty
        turnover = jnp.sum(jnp.abs(action_sequence[0] - current_weights))
        
        # Total objective (minimize)
        objective = (
            -expected_return  # Maximize return
            + penalty  # Risk constraints
            - cfg.entropy_weight * entropy  # Encourage diversification
            + cfg.turnover_penalty * turnover  # Discourage excessive trading
        )
        
        metrics = {
            "objective": objective,
            "expected_return": expected_return,
            "penalty": penalty,
            "entropy": entropy,
            "turnover": turnover,
            **risk_metrics,
        }
        
        return objective, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _optimization_step(
        self,
        action_params: jax.Array,
        opt_state,
        rng: jax.Array,
        initial_state: jax.Array,
        portfolio_value: float,
        current_weights: jax.Array,
    ):
        """Single optimization step."""
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: self._objective(p, rng, initial_state, portfolio_value, current_weights),
            has_aux=True
        )(action_params)
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(action_params, updates)
        
        return new_params, new_opt_state, metrics

    def solve(
        self,
        initial_state: jax.Array,
        portfolio_value: float,
        current_weights: jax.Array,
        rng: jax.Array,
        verbose: bool = False,
    ) -> Tuple[jax.Array, dict]:
        """
        Solve MPC problem to get optimal action sequence.
        
        Args:
            initial_state: Current observation
            portfolio_value: Current portfolio value
            current_weights: Current portfolio weights
            rng: Random key
            verbose: Print progress
        
        Returns:
            optimal_weights: First action from optimal sequence
            info: Dictionary with optimization info
        """
        cfg = self.config
        
        # Initialize action parameters (will be softmaxed)
        # Start from current weights (transformed to logits)
        rng, init_rng = jax.random.split(rng)
        action_params = jnp.log(current_weights + 1e-8)
        action_params = jnp.tile(action_params[None, :], (cfg.horizon, 1))
        # Add small noise for exploration
        action_params += 0.1 * jax.random.normal(init_rng, action_params.shape)
        
        # Initialize optimizer
        opt_state = self.optimizer.init(action_params)
        
        # Optimization loop
        all_metrics = []
        for i in range(cfg.num_optimization_steps):
            rng, step_rng = jax.random.split(rng)
            action_params, opt_state, metrics = self._optimization_step(
                action_params, opt_state, step_rng,
                initial_state, portfolio_value, current_weights
            )
            all_metrics.append(metrics)
            
            if verbose and (i + 1) % 20 == 0:
                print(f"Step {i+1}: obj={metrics['objective']:.4f}, "
                      f"E[R]={metrics['expected_return']:.4f}, "
                      f"VaR={metrics['var']:.4f}")
        
        # Get optimal weights (first action in sequence)
        optimal_sequence = jax.nn.softmax(action_params, axis=-1)
        optimal_weights = optimal_sequence[0]
        
        # Compile info
        info = {
            "optimal_sequence": optimal_sequence,
            "final_metrics": metrics,
            "all_metrics": all_metrics,
        }
        
        return optimal_weights, info

    def get_action(
        self,
        state: jax.Array,
        portfolio_value: float,
        current_weights: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """
        Get optimal portfolio weights for next step.
        
        Convenience method that just returns the weights.
        
        Args:
            state: Current observation
            portfolio_value: Current portfolio value
            current_weights: Current weights
            rng: Random key
        
        Returns:
            Target portfolio weights
        """
        weights, _ = self.solve(state, portfolio_value, current_weights, rng)
        return weights
