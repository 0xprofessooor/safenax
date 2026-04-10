import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import State
from brax.envs.ant import Ant


class EcoAntV2(Ant):
    """
    Ant with a 'low battery' constraint.

    Modifications:
    1. Stochasticity: Adds Gaussian noise to actions to simulate motor imperfection.
    2. Cost Signal: Returns energy used per step.
    """

    def __init__(
        self, battery_limit: float = 500.0, noise_scale: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.battery_limit = battery_limit
        self.noise_scale = noise_scale

    @property
    def name(self) -> str:
        return "EcoAnt-v2"

    def cost_fn(
        self, obs: jax.Array, action: jax.Array, next_obs: jax.Array
    ) -> jax.Array:
        """
        Cost function that returns the energy used at each step.
        """
        energy_used = jnp.sum(jnp.square(action)) * 0.5
        return energy_used

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

    def is_done(self, obs: jax.Array) -> jax.Array:
        """
        Termination condition based on torso height and battery depletion.
        """
        z_pos = obs[..., 0]
        min_z, max_z = self._healthy_z_range
        is_unhealthy = (z_pos < min_z) | (z_pos > max_z)
        battery_dead = obs[..., -1] <= 0.0
        return (is_unhealthy | battery_dead).astype(jnp.float32)

    def transition_fn(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Differentiable state transition function f(s, a) -> s'.

        Implements a forward Euler integration of the EcoAnt dynamics.
        Deterministic (no noise) for use in gradient-based MBRL.

        Observation layout (27 base Ant + 1 battery = 28 total):
            obs[0]     : torso z-position
            obs[1:5]   : torso quaternion (w, x, y, z)
            obs[5:13]  : joint angles (8)
            obs[13:16] : torso linear velocity (vx, vy, vz)
            obs[16:19] : torso angular velocity (ωx, ωy, ωz)
            obs[19:27] : joint angular velocities (8)
            obs[27]    : battery percentage
        """
        dt = self.dt

        # Deterministic clip — no noise for differentiability
        action = jnp.clip(action, -1.0, 1.0)

        # --- Unpack observation ---
        z_pos = obs[0:1]
        quat = obs[1:5]  # (w, x, y, z)
        q_joints = obs[5:13]
        lin_vel = obs[13:16]  # (vx, vy, vz)
        ang_vel = obs[16:19]  # (ωx, ωy, ωz)
        qd_joints = obs[19:27]
        battery_pct = obs[27]

        # --- Joint dynamics (forward Euler) ---
        # First 6 DOFs are the torso free joint; remainder are the 8 revolute joints
        gear = self.sys.actuator.gear  # (8,)
        damping = self.sys.dof.damping[6:]  # (8,)
        joint_acc = gear * action - damping * qd_joints
        next_qd_joints = qd_joints + joint_acc * dt
        next_q_joints = q_joints + qd_joints * dt

        # --- Torso dynamics ---
        # Integrate z-position with z-velocity; x/y not in obs so omitted
        next_z_pos = z_pos + lin_vel[2:3] * dt

        # Linear velocity: gravity + approximate ground contact
        gravity = self.sys.gravity  # (3,) e.g. [0, 0, -9.81]

        # Soft ground contact: legs act as a virtual spring-damper supporting
        # the torso near the nominal standing height (~0.75 m).
        nominal_z = 0.75
        delta_z = nominal_z - z_pos[0]  # positive when below nominal

        # Smooth contact activation centered below nominal so that at nominal
        # height the support nearly equals gravity (sigmoid(5) ≈ 0.993).
        contact = jax.nn.sigmoid(20.0 * (delta_z + 0.25))

        # Spring restoring force (only pushes up when below nominal)
        k_spring = 100.0  # effective spring stiffness / mass  [1/s^2]
        spring_accel = k_spring * jnp.maximum(delta_z, 0.0)

        # Ground reaction: balances gravity when in contact, plus spring
        ground_accel_z = contact * (-gravity[2] + spring_accel)
        lin_accel = gravity + jnp.array([0.0, 0.0, ground_accel_z])

        # Horizontal friction when in contact (prevents drift)
        horiz_damp = 1.0 - contact * 5.0 * dt
        next_lin_vel = jnp.array(
            [
                lin_vel[0] * horiz_damp,
                lin_vel[1] * horiz_damp,
                lin_vel[2] + lin_accel[2] * dt,
            ]
        )

        # Angular velocity: damp via contact friction and internal dissipation
        ang_damp = 1.0 - contact * 10.0 * dt
        next_ang_vel = ang_vel * ang_damp
        # Quaternion: first-order integration  q̇ = 0.5 · Ω(ω) · q
        wx, wy, wz = ang_vel
        qw, qx, qy, qz = quat
        qdot = 0.5 * jnp.array(
            [
                -wx * qx - wy * qy - wz * qz,
                wx * qw + wz * qy - wy * qz,
                wy * qw - wz * qx + wx * qz,
                wz * qw + wy * qx - wx * qy,
            ]
        )
        next_quat = quat + qdot * dt
        next_quat = next_quat / jnp.linalg.norm(next_quat)

        # --- Battery update (analytical) ---
        energy_used = jnp.sum(jnp.square(action)) * 0.5
        new_battery = jnp.maximum(battery_pct * self.battery_limit - energy_used, 0.0)
        new_battery_pct = new_battery / self.battery_limit

        # --- Assemble next observation ---
        return jnp.concatenate(
            [
                next_z_pos,
                next_quat,
                next_q_joints,
                next_lin_vel,
                next_ang_vel,
                next_qd_joints,
                jnp.array([new_battery_pct]),
            ]
        )

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

        # Observation: Append the new battery level to the observation vector
        new_obs = jnp.concatenate([next_state.obs, jnp.array([new_battery_pct])])

        new_info = {
            **next_state.info,
            "rng": noise_key,
            "cost": energy_used,
            "battery": new_battery,
        }

        return next_state.replace(obs=new_obs, done=new_done, info=new_info)

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        # Append initial battery to observation
        new_obs = jnp.concatenate([state.obs, jnp.array([1.0])])

        # Initialize info
        new_info = {
            **state.info,
            "rng": rng,
            "cost": jnp.array(0.0),
            "battery": jnp.array(self.battery_limit),
        }

        return state.replace(obs=new_obs, info=new_info)


envs.register_environment(EcoAntV2.name, EcoAntV2)
