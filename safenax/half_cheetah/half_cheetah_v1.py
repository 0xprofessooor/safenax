"""HalfCheetah extension exposing differentiable transition_fn / reward_fn for Sobolev PPO."""

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.half_cheetah import Halfcheetah


class HalfCheetahV1(Halfcheetah):
    """Vanilla brax HalfCheetah with analytic, differentiable transition_fn and reward_fn.

    Observation layout (17-dim, x-position excluded):
        obs[0]    : torso z-position (rootz)
        obs[1]    : torso angle (rooty)
        obs[2:8]  : bthigh, bshin, bfoot, fthigh, fshin, ffoot joint angles
        obs[8]    : x-velocity
        obs[9]    : z-velocity
        obs[10]   : torso angular velocity (wy)
        obs[11:17]: 6 joint angular velocities (same order as obs[2:8])
    """

    @property
    def name(self) -> str:
        return "HalfCheetah-v1"

    def reward_fn(
        self, obs: jax.Array, action: jax.Array, next_obs: jax.Array
    ) -> jax.Array:
        """Differentiable reward — biased approximation of brax HalfCheetah's reward.

        brax computes forward_reward = weight * (next_x - prev_x) / dt from world
        x-positions, but x is excluded from obs. The cheetah is fast enough that
        averaging vx samples (as we do for Hopper/EcoAnt) is materially biased,
        so we use next_vx alone — the Sobolev gradient signal then flows entirely
        through transition_fn, which is the cleaner formulation.
        """
        forward_reward = self._forward_reward_weight * next_obs[..., 8]
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        return forward_reward - ctrl_cost

    def transition_fn(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Differentiable f(obs, action) -> next_obs.

        Forward Euler with n_frames substeps (matching brax's pipeline_step
        substepping schedule) and a soft ground-contact spring-damper.
        """
        action = jnp.clip(action, -1.0, 1.0)
        gear = self.sys.actuator.gear  # (6,)
        damping = self.sys.dof.damping[3:]  # (6,) — last 6 DOFs are hinges
        gravity = self.sys.gravity
        # _n_frames is a Python int (constant at construction); avoids tracing.
        n_substeps = self._n_frames
        sub_dt = self.dt / n_substeps

        # --- Unpack obs into mutable state ---
        z = obs[0]
        rooty = obs[1]
        q_joints = obs[2:8]
        vx = obs[8]
        vz = obs[9]
        wy = obs[10]
        qd_joints = obs[11:17]

        nominal_z = 0.0  # cheetah body sits near z=0 at rest

        for _ in range(n_substeps):
            joint_acc = gear * action - damping * qd_joints
            qd_joints = qd_joints + joint_acc * sub_dt
            q_joints = q_joints + qd_joints * sub_dt

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


envs.register_environment(HalfCheetahV1.name, HalfCheetahV1)
