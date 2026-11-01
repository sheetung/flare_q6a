"""
场景 2：刚性缆绳自由飞行环境（RigidHoverEnv）

与 HoverEnv 类似，但缆绳为刚性单连杆模型（2 自由度：俯仰 + 滚转），
使用 URDF `real_sus_exp_rigid.urdf`。缆绳角度直接从仿真关节读取。

观测空间（26 维）：与 HoverEnv 相同。
奖励函数：与 HoverEnv 类似，但 cable_angle_safety 使用连续奖励而非二值惩罚。
"""

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, inv_quat, transform_quat_by_quat, quat_to_R
import flare_q6a.utils.server_s2_mixer as mixer


def gs_rand_float(lower, upper, shape, device):
    """在 [lower, upper] 范围内生成均匀分布随机数"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RigidHoverEnv:
    """刚性缆绳悬停环境：单连杆刚性缆绳 + 随机航点跟踪"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.hover_throttle = self.env_cfg.get("hover_throttle")
        self.throttle_range = self.env_cfg.get("throttle_range")

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # 创建 Genesis 仿真场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(1)),
                background_color=(0.9, 0.9, 0.9),
                ambient_light=tuple(
                    env_cfg.get("visual_ambient_light", (0.1, 0.1, 0.1))
                ),
                lights=env_cfg.get(
                    "visual_lights",
                    [
                        {
                            "type": "directional",
                            "dir": (-1.0, -1.0, -1.0),
                            "color": (1.0, 1.0, 1.0),
                            "intensity": 5.0,
                        }
                    ],
                ),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=False,
                enable_joint_limit=False,
            ),
            show_viewer=show_viewer,
        )

        # 添加地面平面
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        # 添加目标点可视化（可选）
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.07,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.3, 0.3, 0.9),
                    ),
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1920, 1080),
                pos=(0.0, 10.0, 3.0),
                lookat=(0.0, 0.0, 1.5),
                fov=30,
                GUI=True,
            )

        # 初始位置和姿态
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.payload_init_pos = self.base_init_pos - torch.tensor([0, 0, self.env_cfg["cable_length"]], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.default_dofs_pos = torch.zeros((self.num_envs, self.env_cfg['num_of_dofs']), device=self.device, dtype=gs.tc_float)

        # 添加无人机实体（刚性缆绳 URDF 模型）
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file=self.env_cfg.get("urdf_file", "assets/suspended_system/real_sus_exp_rigid.urdf"),
                merge_fixed_links=False,  # Keep center_of_mass_link for rigid cable
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.2, 0.2),
                ),
            ),
        )

        # Task-specific static geometry must be registered before scene.build().
        # Navigation subclasses use this hook without changing legacy S2 scenes.
        self._add_task_entities()

        # 构建场景（批量创建 num_envs 个并行环境）
        self.scene.build(n_envs=num_envs)

        # 获取刚性缆绳关节引用（场景构建后才能获取，关节名从配置读取，默认兼容旧 URDF）
        if self.env_cfg['num_of_dofs'] > 0:
            pitch_joint_name = self.env_cfg.get("cable_pitch_joint_name", "center_of_mass_link_to_cable_sphere")
            roll_joint_name = self.env_cfg.get("cable_roll_joint_name", "cable_sphere_to_rigid_cable")
            self.cable_pitch_joint = self.drone.get_joint(name=pitch_joint_name)
            self.cable_roll_joint = self.drone.get_joint(name=roll_joint_name)
        else:
            self.cable_pitch_joint = None
            self.cable_roll_joint = None

        roll_pid = self.env_cfg.get("mixer_roll_pid", [0.1, 0.05, 0.0001])
        pitch_pid = self.env_cfg.get("mixer_pitch_pid", [0.1, 0.05, 0.0001])
        yaw_pid = self.env_cfg.get("mixer_yaw_pid", [0.1, 0.05, 0.0001])
        self.quad_mixer = mixer.QuadXMixer(
            roll_p=roll_pid[0], roll_i=roll_pid[1], roll_d=roll_pid[2],
            pitch_p=pitch_pid[0], pitch_i=pitch_pid[1], pitch_d=pitch_pid[2],
            yaw_p=yaw_pid[0], yaw_i=yaw_pid[1], yaw_d=yaw_pid[2],
            i_limit=self.env_cfg.get("mixer_i_limit", None),
            d_alpha=self.env_cfg.get("mixer_d_alpha", None),
            device=self.device,
        )

        # 准备奖励函数，并将奖励权重乘以 dt（积分近似）
        self.reward_functions, self.episode_sums = dict(), dict()
        self.episode_abs_sums = dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            self.episode_abs_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.next_commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # quadrotor related states
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # payload related states
        self.payload_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_payload_pos = torch.zeros_like(self.payload_pos)
        self.payload_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cable_length = self.env_cfg["cable_length"]
        self.cable_angle = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)  # [pitch, roll] angles
        self.last_cable_angle = torch.zeros_like(self.cable_angle)
        self.cable_angle_vel = torch.zeros_like(self.cable_angle)
        self.cable_body_angle = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.max_safe_cable_angle = math.radians(self.env_cfg.get("max_safe_cable_angle_deg", 80))
        self.safety_violation_count = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.max_safety_violations = self.env_cfg.get("max_safety_violations", 10)

        self.extras = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx] = self.next_commands[envs_idx]

        self.next_commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.next_commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.next_commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def step(self, actions):
        """执行一步仿真：动作 → 电机控制 → 物理仿真 → 更新状态 → 计算奖励"""
        # Episode data is only valid on steps that actually reset an environment.
        # Keeping the previous value makes the runner repeatedly log stale data.
        self.extras.pop("episode", None)
        # Pre-physics NaN guard: subclasses may sanitise incoming actions
        # before they reach the mixer and physics engine.  Default is identity.
        actions = self._sanitise_prephysics_actions(actions)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 将动作映射到实际控制范围：油门 [0,1]，角速率 [-max, max]。
        # 默认钩子保持 S1/S2/S7 的历史解码完全不变；S8 只覆盖该钩子。
        self.throttle_cmd = self._decode_collective_thrust(exec_actions)
        self.roll_rate_cmd = exec_actions[:, 1] * self.env_cfg["max_roll_rate"]
        self.pitch_rate_cmd = exec_actions[:, 2] * self.env_cfg["max_pitch_rate"]
        self.yaw_rate_cmd = exec_actions[:, 3] * self.env_cfg["max_yaw_rate"]

        # mixer computes motor RPM
        ang_vel = self.drone.get_ang()
        roll_pid, pitch_pid, yaw_pid = self.quad_mixer.pid_attitude_command_for_mix(
            roll_vel_r=ang_vel[:, 0], pitch_vel_r=ang_vel[:, 1], yaw_vel_r=ang_vel[:, 2],
            roll_vel_d=self.roll_rate_cmd, pitch_vel_d=self.pitch_rate_cmd, yaw_vel_d=self.yaw_rate_cmd,
            dt=self.dt
        )
        motor_outputs = self.quad_mixer.mix(
            throttle=self.throttle_cmd,
            roll_pid=roll_pid,
            pitch_pid=pitch_pid,
            yaw_pid=yaw_pid
        )
        self.motor_outputs = motor_outputs
        # RPM = motor × √TWR × rpm_constant，rpm_constant 由模型 kf 和 mass 决定
        self.motor_rpm = self._compute_motor_rpm(motor_outputs)
        self.drone.set_propellels_rpm(self.motor_rpm)

        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True, envs_idx=list(range(self.num_envs)))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.next_rel_pos = self.next_commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.base_lin_vel[:] = self.drone.get_vel()
        self.base_ang_vel[:] = self.drone.get_ang()
        self.last_payload_pos[:] = self.payload_pos
        self.payload_pos[:] = self.drone.get_link(name="end_sphere").get_pos()
        self.payload_lin_vel[:] = self.drone.get_link(name="end_sphere").get_vel()
        self.rel_pos_payload = self.commands - self.payload_pos
        self.last_rel_pos_payload = self.commands - self.last_payload_pos

        # update safety-related states
        self._update_safety_states()
        # Optional deployment/domain-adaptation observer update.  This hook is
        # deliberately after current collision/safety state is available and
        # before reset or next-observation construction.
        self._after_physics_control_update()

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # Navigation tasks can change commands discontinuously. For those
        # tasks, terminate against a fixed world workspace rather than the
        # distance to the newly sampled command.
        termination_position_error = self.rel_pos
        if self.env_cfg.get("termination_position_frame") == "world":
            termination_position_error = self.base_pos - self.base_init_pos

        # ---- Authoritative termination masks ---------------------------------
        # Every downstream consumer (reward, success, diagnostics, eval) MUST
        # read these two fields.  Subclasses MUST NOT recompute them.
        self.timeout_condition = (
            self.episode_length_buf > self.max_episode_length
        )
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(termination_position_error[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(termination_position_error[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(termination_position_error[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.safety_violation_count > self.max_safety_violations)
            | (self.cable_body_angle > self.env_cfg.get("termination_cable_angle_threshold", math.radians(90)))
            | self._task_crash_condition()
        )
        self.reset_buf = self.timeout_condition | self.crash_condition

        self.extras["time_outs"] = self.timeout_condition.float()

        reset_envs = self.reset_buf.nonzero(as_tuple=False).flatten()

        # ---- Lifecycle hooks ---------------------------------------------------
        # Phase 1 — pre-reset state snapshot (stability window, non-reward metrics)
        self._before_episode_reset(reset_envs)

        # Phase 2 — rewards + post-reward snapshot (reward components, episode return)
        if self.env_cfg.get("reward_before_reset", False):
            self._compute_rewards()
            self._after_reward_before_reset(reset_envs)
            self.reset_idx(reset_envs)
        else:
            # Preserve the historical S2 behavior for existing checkpoints.
            self.reset_idx(reset_envs)
            self._compute_rewards()

        # compute observations
        rotmat = quat_to_R(self.base_quat)
        rotmat_flat = rotmat.reshape(self.num_envs, 9)
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos[:, 0:2] * self.obs_scales["rel_pos_xy"], -1, 1),
                torch.clip(self.rel_pos[:, 2:3] * self.obs_scales["rel_pos_z"], -1, 1),
                torch.clip(self.next_rel_pos[:, 0:2] * self.obs_scales["rel_pos_xy"], -1, 1),
                torch.clip(self.next_rel_pos[:, 2:3] * self.obs_scales["rel_pos_z"], -1, 1),
                torch.clip(self.base_lin_vel[:, 0:2] * self.obs_scales["lin_vel_xy"], -1, 1),
                torch.clip(self.base_lin_vel[:, 2:3] * self.obs_scales["lin_vel_z"], -1, 1),
                rotmat_flat,
                self.last_actions,
                torch.clip(self.cable_angle * self.obs_scales.get("cable_angle", 1.0), -1, 1),
                torch.clip(self.cable_angle_vel * self.obs_scales.get("cable_angle_vel", 0.1), -1, 1),
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _add_task_entities(self):
        """Hook for task geometry that must exist before scene construction."""

    def _decode_collective_thrust(self, exec_actions):
        """Decode the collective channel without changing legacy semantics."""
        if self.hover_throttle is not None and self.throttle_range is not None:
            return torch.clamp(
                self.hover_throttle + exec_actions[:, 0] * self.throttle_range,
                0.0,
                1.0,
            )
        return torch.sqrt(exec_actions[:, 0] * 0.5 + 0.5)

    def _apply_motor_strength_scale(self, motor_outputs):
        """Map clamped mixer outputs to effective motor-speed commands.

        The identity default is the historical S1/S2/S7 behavior.  Scenario 8
        overrides this hook to apply per-environment motor-strength domains.
        """
        return motor_outputs

    def _compute_motor_rpm(self, motor_outputs):
        effective_outputs = self._apply_motor_strength_scale(motor_outputs)
        rpm_constant = self.env_cfg.get(
            "rpm_constant", 68146.3
        )  # 默认兼容旧 URDF (real_sus_exp_rigid)
        return (
            effective_outputs
            * torch.sqrt(torch.tensor(self.env_cfg["TWR_max"]))
            * rpm_constant
        )

    def _after_physics_control_update(self):
        """Hook after the applied command's physical transition is observed."""

    def _reset_control_adaptation(self, envs_idx):
        """Reset only the selected environments' optional control adaptation."""

    def _sanitise_prephysics_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Hook before ``torch.clip`` — subclasses may replace NaN/Inf actions.

        Default identity (S1--S8 unchanged).  S9 replaces NaN rows with zeros
        and flags those environments for termination via ``_task_crash_condition``.
        """
        return actions

    def _before_episode_reset(self, reset_env_ids: torch.Tensor):
        """Hook after termination but before rewards.

        Called while pre-reset state buffers are still valid.  Use for
        stability-window updates, non-reward terminal metrics, and
        success-condition snapshots.  *Do not* read ``rew_buf`` or
        ``episode_sums`` here — they may not yet include the current step.

        Default no-op (S1--S8 unchanged).
        """
        pass

    def _after_reward_before_reset(self, reset_env_ids: torch.Tensor):
        """Hook after ``_compute_rewards()`` but before ``reset_idx()``.

        Called when ``self.rew_buf`` already includes the current step's
        reward (including any terminal crash penalty).  Use for:
        - accumulating ``episode_return_buf += self.rew_buf``,
        - validating the reward contract (crash penalty assertions),
        - snapshotting actual reward components.

        Default no-op (S1--S8 unchanged).
        """
        pass

    def _task_crash_condition(self):
        """Hook for task-specific terminal conditions evaluated after physics."""
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_abs_sums[name] += torch.abs(rew)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset cable and payload
        self.payload_pos[envs_idx] = self.payload_init_pos
        self.last_payload_pos[envs_idx] = self.payload_init_pos
        self.rel_pos_payload = self.commands - self.payload_pos
        self.last_rel_pos_payload = self.commands - self.last_payload_pos
        self.payload_lin_vel[envs_idx] = 0

        # Q6A Genesis partial-reset compatibility: set complete pose, then
        # refresh all link transforms. Partial FK in the pinned backend may
        # otherwise leave a reset payload at its previous world position.
        qpos = torch.zeros((len(envs_idx), self.drone.n_qs), device=self.device, dtype=gs.tc_float)
        qpos[:, :3] = self.base_init_pos
        qpos[:, 3:7] = self.base_init_quat
        self.drone.set_qpos(qpos, zero_velocity=True, envs_idx=envs_idx)
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.drone.solver._kernel_forward_kinematics_links_geoms(all_ids)

        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.quad_mixer.reset(envs_idx)
        self._reset_control_adaptation(envs_idx)

        # reset safety states
        self.cable_angle[envs_idx] = 0.0
        self.last_cable_angle[envs_idx] = 0.0
        self.cable_angle_vel[envs_idx] = 0.0
        self.cable_body_angle[envs_idx] = 0.0
        self.safety_violation_count[envs_idx] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.extras["episode"]["rew_abs_sum_" + key] = torch.mean(
                self.episode_abs_sums[key][envs_idx]
            ).item()
            self.episode_sums[key][envs_idx] = 0.0
            self.episode_abs_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ 奖励函数 ----------------
    def _reward_target(self):
        """目标到达奖励（势能差）：距离减小则给正奖励"""
        target_rew = torch.square(self.last_rel_pos) - torch.square(self.rel_pos)
        target_rew = torch.sum(target_rew, dim=1)
        return target_rew

    def _reward_smooth(self):
        """动作平滑惩罚：相邻步动作差异越大惩罚越大"""
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        """偏航角控制奖励：偏航角越小奖励越高"""
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        """角速度惩罚：角速度越小越好"""
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        """碰撞惩罚：发生终止条件时给惩罚"""
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_cable_angle_safety(self):
        """缆绳角度安全奖励：连续奖励，角度越小越好，超过安全角度额外惩罚"""
        cable_body_angle = self.cable_body_angle
        # 连续奖励：角度越小越好，归一化到[0,1]，0角度得1，max_safe_angle得0
        normalized_angle = cable_body_angle / self.max_safe_cable_angle
        angle_safety_reward = torch.clamp(1.0 - normalized_angle, 0.0, 1.0)
        # 超过安全角度的额外惩罚
        unsafe_mask = cable_body_angle > self.max_safe_cable_angle
        angle_safety_reward[unsafe_mask] = -self.reward_cfg.get("unsafe_angle_penalty", 0.001)
        return angle_safety_reward

    def _update_safety_states(self):
        """更新安全相关状态变量：缆绳角度、角度速度、安全违规计数"""
        self.last_payload_pos[:] = self.payload_pos[:]
        self.last_cable_angle[:] = self.cable_angle[:]

        if self.cable_pitch_joint is not None:
            # Read actual joint angles from simulation
            all_dofs_pos = self.drone.get_dofs_position()
            pitch_idx = self.cable_pitch_joint.dof_idx_local
            roll_idx = self.cable_roll_joint.dof_idx_local
            self.cable_angle[:, 0] = all_dofs_pos[:, pitch_idx]
            self.cable_angle[:, 1] = all_dofs_pos[:, roll_idx]
        else:
            # Fallback: compute from geometry
            cable_vector = self.payload_pos - self.base_pos
            self.cable_angle[:, 0] = torch.atan2(cable_vector[:, 0], -cable_vector[:, 2])
            self.cable_angle[:, 1] = torch.atan2(cable_vector[:, 1], -cable_vector[:, 2])

        self.cable_angle_vel[:] = (self.cable_angle - self.last_cable_angle) / self.dt

        cable_vector = self.payload_pos - self.base_pos
        cable_vector_normalized = cable_vector / torch.norm(cable_vector, dim=1, keepdim=True)
        rotmat = quat_to_R(self.base_quat)
        body_z_axis = -rotmat[:, :, 2]

        dot_product = torch.sum(cable_vector_normalized * body_z_axis, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        self.cable_body_angle = torch.acos(dot_product)

        extreme_angle_violation = self.cable_body_angle > self.max_safe_cable_angle
        self.safety_violation_count += extreme_angle_violation.int()
