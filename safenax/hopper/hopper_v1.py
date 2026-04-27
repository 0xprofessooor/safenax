"""Hopper extension exposing differentiable transition_fn / reward_fn for Sobolev PPO."""

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.hopper import Hopper


class HopperV1(Hopper):
    """Vanilla brax Hopper with analytic, differentiable transition_fn and reward_fn.

    Observation layout (11-dim, x-position excluded):
        obs[0]   : torso z-position (height)
        obs[1]   : torso angle (rooty)
        obs[2:5] : thigh, leg, foot joint angles
        obs[5]   : x-velocity
        obs[6]   : z-velocity
        obs[7]   : torso angular velocity (wy)
        obs[8:11]: thigh, leg, foot joint angular velocities
    """

    @property
    def name(self) -> str:
        return "Hopper-v1"

    def reward_fn(
        self, obs: jax.Array, action: jax.Array, next_obs: jax.Array
    ) -> jax.Array:
        """Differentiable reward matching brax Hopper.

        forward_reward uses the average x-velocity (reduces aliasing between
        the analytic transition_fn and the brax pipeline step), healthy_reward
        is a constant when terminate_when_unhealthy is True (the default),
        ctrl_cost is the standard quadratic.
        """
        forward_reward = (
            self._forward_reward_weight * 0.5 * (obs[..., 5] + next_obs[..., 5])
        )

        if self._terminate_when_unhealthy:
            healthy_reward = jnp.full_like(forward_reward, self._healthy_reward)
        else:
            z = next_obs[..., 0]
            angle = next_obs[..., 1]
            min_z, max_z = self._healthy_z_range
            min_a, max_a = self._healthy_angle_range
            is_healthy = (z > min_z) & (z < max_z) & (angle > min_a) & (angle < max_a)
            healthy_reward = self._healthy_reward * is_healthy.astype(
                forward_reward.dtype
            )

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        return forward_reward + healthy_reward - ctrl_cost

    def transition_fn(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Differentiable f(obs, action) -> next_obs.

        Forward Euler with n_frames substeps (matching brax's pipeline_step
        substepping schedule) and a soft ground-contact spring-damper.
        Deterministic, no noise.
        """
        action = jnp.clip(action, -1.0, 1.0)
        gear = self.sys.actuator.gear  # (3,)
        damping = self.sys.dof.damping[3:]  # (3,) — last 3 DOFs are hinges
        gravity = self.sys.gravity
        # _n_frames is a Python int (constant at construction); avoids tracing.
        n_substeps = self._n_frames
        sub_dt = self.dt / n_substeps

        # --- Unpack obs into mutable state ---
        z = obs[0]
        rooty = obs[1]
        q_joints = obs[2:5]
        vx = obs[5]
        vz = obs[6]
        wy = obs[7]
        qd_joints = obs[8:11]

        nominal_z = 1.25  # brax docstring init state

        for _ in range(n_substeps):
            # Joint dynamics
            joint_acc = gear * action - damping * qd_joints
            qd_joints = qd_joints + joint_acc * sub_dt
            q_joints = q_joints + qd_joints * sub_dt  # semi-implicit (use new qd)

            # Soft ground contact
            delta_z = nominal_z - z
            contact = jax.nn.sigmoid(20.0 * (delta_z + 0.25))
            k_spring = 100.0
            spring_accel = k_spring * jnp.maximum(delta_z, 0.0)
            ground_accel_z = contact * (-gravity[2] + spring_accel)

            horiz_damp = 1.0 - contact * 5.0 * sub_dt
            vx = vx * horiz_damp
            vz = vz + (gravity[2] + ground_accel_z) * sub_dt
            z = z + vz * sub_dt

            ang_damp = 1.0 - contact * 10.0 * sub_dt
            wy = wy * ang_damp
            rooty = rooty + wy * sub_dt

        return jnp.concatenate(
            [
                jnp.array([z, rooty]),
                q_joints,
                jnp.array([vx, vz, wy]),
                qd_joints,
            ]
        )


envs.register_environment(HopperV1.name, HopperV1)
