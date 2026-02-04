"""
Implicit Quantile Network (IQN) for learning distributional state transitions.

This module implements an IQN-based model that learns the conditional quantile
function Q_τ(s'|s,a) for state transitions, enabling distributional planning.

Reference: Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
"""

from typing import Tuple, NamedTuple, Optional
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


class IQNConfig(NamedTuple):
    """Configuration for IQN dynamics model."""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64  # Dimension for cosine embedding
    num_cosine_features: int = 64  # Number of cosine basis functions
    num_layers: int = 3
    num_quantile_samples: int = 32  # τ samples during training
    learning_rate: float = 1e-3
    huber_kappa: float = 1.0  # Huber loss threshold (0 = pinball, >0 = quantile huber)


class CosineEmbedding(nn.Module):
    """
    Cosine basis embedding for quantile levels τ ∈ [0,1].
    
    Maps scalar τ to high-dimensional embedding using:
    Φ_j(τ) = ReLU(Σ_i cos(πiτ) * w_ij + b_j)
    """
    embedding_dim: int
    num_cosine_features: int = 64

    @nn.compact
    def __call__(self, tau: jax.Array) -> jax.Array:
        """
        Args:
            tau: Quantile levels, shape (..., num_quantiles) or (...)
        Returns:
            Embedding, shape (..., embedding_dim)
        """
        # Ensure tau has at least 1 dimension
        tau = jnp.atleast_1d(tau)
        original_shape = tau.shape
        
        # Flatten for processing
        tau_flat = tau.reshape(-1, 1)  # (batch, 1)
        
        # Cosine features: cos(π * i * τ) for i = 0, 1, ..., n-1
        i = jnp.arange(self.num_cosine_features, dtype=jnp.float32)  # (n,)
        cos_features = jnp.cos(jnp.pi * tau_flat * i)  # (batch, n)
        
        # Linear projection + ReLU
        embedding = nn.Dense(self.embedding_dim)(cos_features)
        embedding = nn.relu(embedding)
        
        # Reshape back
        new_shape = original_shape + (self.embedding_dim,)
        return embedding.reshape(new_shape)


class IQNDynamicsNetwork(nn.Module):
    """
    IQN network for learning state transition quantiles.
    
    Architecture:
    1. State-action encoder: Ψ(s,a) → R^d
    2. Quantile embedding: Φ(τ) → R^d  
    3. Hadamard product: H = Ψ(s,a) ⊙ Φ(τ)
    4. Output layers: H → ŝ' (predicted next state quantile)
    """
    config: IQNConfig

    def setup(self):
        cfg = self.config
        
        # State-action encoder
        self.state_action_encoder = [
            nn.Dense(cfg.hidden_dim) for _ in range(cfg.num_layers - 1)
        ]
        self.state_action_final = nn.Dense(cfg.embedding_dim)
        
        # Quantile embedding
        self.tau_embedding = CosineEmbedding(
            embedding_dim=cfg.embedding_dim,
            num_cosine_features=cfg.num_cosine_features
        )
        
        # Output network (after Hadamard product)
        self.output_layers = [
            nn.Dense(cfg.hidden_dim) for _ in range(cfg.num_layers - 1)
        ]
        self.output_final = nn.Dense(cfg.state_dim)

    def __call__(
        self, 
        state: jax.Array, 
        action: jax.Array, 
        tau: jax.Array
    ) -> jax.Array:
        """
        Predict next state quantiles.
        
        Args:
            state: Current state, shape (batch, state_dim)
            action: Action taken, shape (batch, action_dim)
            tau: Quantile levels, shape (batch, num_quantiles) or (batch,)
        
        Returns:
            Predicted next state quantiles, shape (batch, num_quantiles, state_dim)
            or (batch, state_dim) if tau is 1D
        """
        # Encode state-action pair
        sa = jnp.concatenate([state, action], axis=-1)  # (batch, state_dim + action_dim)
        
        psi = sa
        for layer in self.state_action_encoder:
            psi = nn.relu(layer(psi))
        psi = self.state_action_final(psi)  # (batch, embedding_dim)
        
        # Handle different tau shapes
        tau_squeeze = tau.ndim == 1
        if tau_squeeze:
            tau = tau[:, None]  # (batch, 1)
        
        # Embed quantile levels
        # tau: (batch, num_quantiles) → phi: (batch, num_quantiles, embedding_dim)
        phi = self.tau_embedding(tau)
        
        # Expand psi for broadcasting: (batch, 1, embedding_dim)
        psi_expanded = psi[:, None, :]
        
        # Hadamard product
        h = psi_expanded * phi  # (batch, num_quantiles, embedding_dim)
        
        # Output network
        for layer in self.output_layers:
            h = nn.relu(layer(h))
        next_state_quantiles = self.output_final(h)  # (batch, num_quantiles, state_dim)
        
        if tau_squeeze:
            next_state_quantiles = next_state_quantiles.squeeze(1)
        
        return next_state_quantiles

    def sample_transitions(
        self,
        state: jax.Array,
        action: jax.Array,
        rng: jax.Array,
        num_samples: int = 100
    ) -> jax.Array:
        """
        Sample next states from the learned distribution.
        
        Args:
            state: Current state, shape (batch, state_dim) or (state_dim,)
            action: Action, shape (batch, action_dim) or (action_dim,)
            rng: Random key
            num_samples: Number of samples to draw
        
        Returns:
            Sampled next states, shape (batch, num_samples, state_dim)
        """
        # Handle unbatched inputs
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action = action[None, :]
        
        batch_size = state.shape[0]
        
        # Sample uniform τ values
        tau = jax.random.uniform(rng, (batch_size, num_samples))
        
        # Get quantile predictions
        samples = self.__call__(state, action, tau)
        
        if squeeze:
            samples = samples.squeeze(0)
        
        return samples


class IQNDynamicsModel:
    """
    High-level interface for IQN dynamics model training and inference.
    """
    
    def __init__(self, config: IQNConfig, rng: jax.Array):
        self.config = config
        self.network = IQNDynamicsNetwork(config)
        
        # Initialize parameters
        dummy_state = jnp.zeros((1, config.state_dim))
        dummy_action = jnp.zeros((1, config.action_dim))
        dummy_tau = jnp.zeros((1, config.num_quantile_samples))
        
        rng, init_rng = jax.random.split(rng)
        params = self.network.init(init_rng, dummy_state, dummy_action, dummy_tau)
        
        # Create optimizer and training state
        tx = optax.adam(config.learning_rate)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx
        )
        self.rng = rng

    @partial(jax.jit, static_argnums=(0,))
    def _compute_quantile_loss(
        self,
        params,
        state: jax.Array,
        action: jax.Array,
        next_state: jax.Array,
        tau: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """
        Compute quantile regression loss (pinball loss).
        
        Args:
            params: Network parameters
            state: Current states, (batch, state_dim)
            action: Actions, (batch, action_dim)
            next_state: Actual next states, (batch, state_dim)
            tau: Sampled quantile levels, (batch, num_quantiles)
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Predict quantiles: (batch, num_quantiles, state_dim)
        pred = self.network.apply(params, state, action, tau)
        
        # Expand next_state for broadcasting: (batch, 1, state_dim)
        target = next_state[:, None, :]
        
        # Residuals: (batch, num_quantiles, state_dim)
        delta = target - pred
        
        # Quantile Huber loss (reduces to pinball if kappa=0)
        kappa = self.config.huber_kappa
        if kappa > 0:
            # Huber loss
            abs_delta = jnp.abs(delta)
            huber = jnp.where(
                abs_delta <= kappa,
                0.5 * delta**2,
                kappa * (abs_delta - 0.5 * kappa)
            )
        else:
            huber = jnp.abs(delta)
        
        # Pinball weighting: τ if δ > 0, (1-τ) if δ < 0
        # tau shape: (batch, num_quantiles) → (batch, num_quantiles, 1)
        tau_expanded = tau[:, :, None]
        pinball_weight = jnp.where(delta > 0, tau_expanded, 1 - tau_expanded)
        
        # Weighted loss
        loss_per_element = pinball_weight * huber
        
        # Average over all dimensions
        loss = loss_per_element.mean()
        
        # Metrics
        metrics = {
            "loss": loss,
            "mae": jnp.abs(delta).mean(),
            "pred_mean": pred.mean(),
            "pred_std": pred.std(),
        }
        
        return loss, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        train_state: train_state.TrainState,
        state: jax.Array,
        action: jax.Array,
        next_state: jax.Array,
        rng: jax.Array,
    ) -> Tuple[train_state.TrainState, dict]:
        """Single training step."""
        batch_size = state.shape[0]
        num_quantiles = self.config.num_quantile_samples
        
        # Sample random quantile levels
        tau = jax.random.uniform(rng, (batch_size, num_quantiles))
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            self._compute_quantile_loss, has_aux=True
        )(train_state.params, state, action, next_state, tau)
        
        # Update parameters
        new_train_state = train_state.apply_gradients(grads=grads)
        
        return new_train_state, metrics

    def train_step(
        self,
        state: jax.Array,
        action: jax.Array,
        next_state: jax.Array,
    ) -> dict:
        """
        Perform a single training step.
        
        Args:
            state: Batch of current states
            action: Batch of actions
            next_state: Batch of next states
        
        Returns:
            metrics: Training metrics
        """
        self.rng, step_rng = jax.random.split(self.rng)
        self.train_state, metrics = self._train_step(
            self.train_state, state, action, next_state, step_rng
        )
        return metrics

    def predict_quantiles(
        self,
        state: jax.Array,
        action: jax.Array,
        tau: jax.Array,
    ) -> jax.Array:
        """
        Predict next state quantiles for given τ values.
        
        Args:
            state: Current state(s)
            action: Action(s)
            tau: Quantile level(s)
        
        Returns:
            Predicted quantiles
        """
        return self.network.apply(self.train_state.params, state, action, tau)

    def sample_next_states(
        self,
        state: jax.Array,
        action: jax.Array,
        num_samples: int = 100,
    ) -> jax.Array:
        """
        Sample next states from the learned distribution.
        
        Args:
            state: Current state(s)
            action: Action(s)
            num_samples: Number of samples
        
        Returns:
            Sampled next states
        """
        self.rng, sample_rng = jax.random.split(self.rng)
        
        # Add batch dim if needed
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action = action[None, :]
        
        batch_size = state.shape[0]
        tau = jax.random.uniform(sample_rng, (batch_size, num_samples))
        
        samples = self.predict_quantiles(state, action, tau)
        
        if squeeze:
            samples = samples.squeeze(0)
        
        return samples

    def get_distribution_stats(
        self,
        state: jax.Array,
        action: jax.Array,
        quantiles: jax.Array = None,
    ) -> dict:
        """
        Get distribution statistics (mean, std, quantiles) for next state.
        
        Args:
            state: Current state
            action: Action
            quantiles: Specific quantile levels to compute (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        
        Returns:
            Dictionary with distribution statistics
        """
        if quantiles is None:
            quantiles = jnp.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action = action[None, :]
        
        # Compute quantiles
        tau = jnp.tile(quantiles[None, :], (state.shape[0], 1))
        q_values = self.predict_quantiles(state, action, tau)
        
        # Estimate mean and std from samples
        samples = self.sample_next_states(state, action, num_samples=1000)
        
        stats = {
            "mean": samples.mean(axis=-2),
            "std": samples.std(axis=-2),
            "quantiles": {f"q{int(q*100)}": q_values[:, i, :] for i, q in enumerate(quantiles)},
        }
        
        if squeeze:
            stats["mean"] = stats["mean"].squeeze(0)
            stats["std"] = stats["std"].squeeze(0)
            stats["quantiles"] = {k: v.squeeze(0) for k, v in stats["quantiles"].items()}
        
        return stats
