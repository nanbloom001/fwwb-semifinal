# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — custom reward processor (lite baseline).
RewardProcess — 自定义奖励处理器（lite baseline）。

This file only ships two example rewards:
    1. _reward_reach_goal       — goal-reaching judgment (0.6 m)
    2. _reward_forward_velocity — forward velocity reward (dense, demonstrates reward writing style)
本文件仅预置两个示例 reward：
    1. _reward_reach_goal       — 赛题到达判定（0.6 m）
    2. _reward_forward_velocity — 前向速度奖励（dense，展示 reward 写法）

Other generic locomotion rewards (track_lin_vel_xy / joint_acc / action_rate, etc.)
are inherited from RewardProcessBase (see tools/base_env/base_reward.py).
Players only need to activate them in the TOML; no need to re-implement them here.
其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
选手在 TOML 中激活即可，无需在此重复实现。

If players need to train a navigation policy, please add more rewards in this file.
选手若需训练导航策略，请在本文件自行添加更多 reward。
"""

import logging

import torch

from agent_ppo.feature.height_scan_features import maze_wall_stair_features
from tools.base_env.base_reward import RewardProcessBase

LOGGER = logging.getLogger(__name__)


class RewardProcess(RewardProcessBase):

    # -----------------------------------------------------------------------
    # Locomotion quality rewards
    # 运动质量奖励
    # -----------------------------------------------------------------------

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).

        奖励长步幅（移动时脚部滞空时间超过阈值）。
        Ref: Rudin et al., "Learning to Walk in Minutes", RSS 2022 (legged_gym).
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        使用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        Ref: Miki et al., Science Robotics 2022.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    def _reward_feet_clearance(
        self,
        command_name: str = "base_velocity",
        target_height: float = 0.08,
        std: float = 0.05,
        terrain_height_scale: float = 0.6,
        max_terrain_extra_height: float = 0.08,
        speed_height_scale: float = 0.01,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
    ):
        """Reward terrain-aware swing-foot clearance to reduce stair-edge trips."""
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        command = self.env.command_manager.get_command(command_name)
        command_speed = torch.norm(command[:, :2], dim=1)

        terrain_extra = torch.zeros(self.env.num_envs, device=self.env.device)
        grid = self._height_scan_grid()
        if grid is not None:
            forward_window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
            if forward_window.shape[-1] > 1 and forward_window.shape[1] > 0:
                step_deltas = torch.abs(forward_window[:, :, 1:] - forward_window[:, :, :-1])
                step_like = (step_deltas > 0.03) & (step_deltas < 0.24)
                wall_score, stair_score = self._height_scan_wall_stair_scores(grid)
                step_samples = torch.where(step_like, step_deltas, torch.zeros_like(step_deltas)).flatten(1)
                local_step = torch.quantile(step_samples, delta_quantile, dim=1)
                stair_gate = (stair_score > 0.03) & (wall_score < 0.08)
                terrain_extra = torch.clamp(
                    terrain_height_scale * local_step,
                    0.0,
                    max_terrain_extra_height,
                ) * stair_gate.float()

        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        dynamic_target_height = target_height + terrain_extra + speed_extra
        height_error = (foot_height - dynamic_target_height.unsqueeze(1)) / max(std, 1.0e-6)
        clearance_reward = torch.exp(-torch.square(height_error))
        is_moving = command_speed > 0.1
        return torch.sum(clearance_reward * swing.float(), dim=1) * is_moving.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_joint_position_penalty(
        self,
        stand_still_scale: float = 5.0,
        velocity_threshold: float = 0.1,
        cmd_threshold: float = 0.1,
        ang_cmd_threshold: float = 0.2,
    ):
        """Penalize joint position deviation from default pose.

        惩罚关节位置偏离默认姿态；静止时惩罚倍增（鼓励站立保持默认姿势）。
        Ref: Kumar et al., "RMA: Rapid Motor Adaptation", RSS 2021.

        Bug fix: original code used `cmd > 0.0` (exact-zero check).
        Since velocity commands are sampled from continuous distributions, the
        probability of all three components being EXACTLY 0.0 simultaneously is
        essentially zero — stand_still_scale never fired in practice.
        Fixed to use `cmd > cmd_threshold` (default 0.1 m/s equivalent) so the
        scale activates whenever the robot is genuinely commanded to stand still.

        原始实现使用 `cmd > 0.0`（精确零判断）。由于速度指令来自连续分布，
        三个分量同时精确等于 0 的概率几乎为零，stand_still_scale 实际上从不生效。
        修复为 `cmd > cmd_threshold`（默认 0.1），让真正的零速指令时才放大惩罚。

        Args:
            stand_still_scale: Penalty multiplier when standing still (cmd ≈ 0).
                               静止（cmd ≈ 0）时的惩罚倍数。
            velocity_threshold: Body velocity threshold to confirm robot is not moving (m/s).
                                机体速度阈值，低于此值判定为静止 (m/s)。
            cmd_threshold: Command norm threshold below which robot is "standing still" (m/s).
                           速度指令模长阈值，低于此值视为零速指令 (m/s)。
            ang_cmd_threshold: Yaw-rate command threshold below which robot is treated as not turning.
                               偏航角速度指令阈值，低于此值视为未被要求转向。
        """
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command("base_velocity")
        cmd_xy = torch.linalg.norm(cmd[:, :2], dim=1)
        cmd_yaw = torch.abs(cmd[:, 2])
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        deviation = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        is_moving = torch.logical_or(
            torch.logical_or(cmd_xy > cmd_threshold, cmd_yaw > ang_cmd_threshold),
            body_vel > velocity_threshold,
        )
        return torch.where(
            is_moving,
            deviation,
            stand_still_scale * deviation,
        )

    def _reward_stand_still_motion(
        self,
        command_name: str = "base_velocity",
        lin_cmd_threshold: float = 0.15,
        ang_cmd_threshold: float = 0.2,
        vertical_vel_scale: float = 0.5,
        ang_vel_scale: float = 0.5,
        joint_vel_scale: float = 0.1,
    ):
        """Penalize body oscillation and leg fidgeting under near-zero commands.

        连续命令采样几乎不会精确落在 0 速度，因此“站立不动”往往训练不足。
        本奖励在近零线速度/角速度命令带内激活，直接惩罚机身上下晃动、
        pitch/roll 摇摆以及小腿频繁抬放造成的关节速度。

        Args:
            lin_cmd_threshold: Near-zero threshold for XY linear command norm (m/s).
            ang_cmd_threshold: Near-zero threshold for yaw command magnitude (rad/s).
            vertical_vel_scale: Weight for vertical body velocity in the penalty.
            ang_vel_scale: Weight for pitch/roll angular velocity in the penalty.
            joint_vel_scale: Weight for mean absolute joint velocity in the penalty.
        """
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)

        near_zero_cmd = (
            torch.linalg.norm(cmd[:, :2], dim=1) < lin_cmd_threshold
        ) & (
            torch.abs(cmd[:, 2]) < ang_cmd_threshold
        )

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b
        mean_abs_joint_vel = torch.mean(torch.abs(asset.data.joint_vel), dim=1)

        motion_penalty = (
            torch.linalg.norm(base_lin_vel[:, :2], dim=1)
            + vertical_vel_scale * torch.abs(base_lin_vel[:, 2])
            + ang_vel_scale * torch.linalg.norm(base_ang_vel[:, :2], dim=1)
            + joint_vel_scale * mean_abs_joint_vel
        )
        return near_zero_cmd.float() * motion_penalty

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).

        惩罚脚撞到垂直面（台阶边缘、墙壁）。水平力 > 5× 垂直力时触发。
        阈值 5× 对齐 legged_gym 原版。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        wall_score_threshold: float = 0.08,
        wall_vs_stair_margin: float = 1.6,
        wall_height_threshold: float = 0.18,
        body_clearance_threshold: float = 0.30,
        wall_jump_threshold: float = 0.16,
        stair_min_delta: float = 0.03,
        stair_max_delta: float = 0.24,
        turn_std: float = 0.45,
        min_forward_cmd: float = 0.05,
        height_scan_weight: float = 1.0,
        nav_scan_weight: float = 0.0,
        nav_threshold: float = 0.5,
    ):
        """Penalize driving into maze-like walls, not stair-like height changes.

        Maze walls are tall, laterally continuous blockers. Stairs are lower,
        forward-progressive height discontinuities. This reward uses that
        distinction so stair edges do not receive the maze wall penalty.
        """
        penalty = torch.zeros(self.env.num_envs, device=self.env.device)
        cmd = self.env.command_manager.get_command(command_name)
        forward_cmd = torch.clamp(cmd[:, 0], min=0.0)
        has_forward_cmd = forward_cmd > min_forward_cmd
        yaw_rate = torch.abs(self._get_robot_asset().data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / max(turn_std, 1.0e-6))

        if hasattr(self.env, "scene") and hasattr(self.env.scene, "sensors"):
            nav_scanner = self.env.scene.sensors.get("nav_scanner")
            if nav_scanner is not None and hasattr(nav_scanner.data, "ray_hits_w"):
                ray_vec = nav_scanner.data.ray_hits_w - nav_scanner.data.pos_w.unsqueeze(1)
                dist = torch.linalg.norm(ray_vec[..., :2], dim=-1)
                valid = torch.isfinite(dist)
                close = torch.clamp(nav_threshold - torch.where(valid, dist, nav_threshold), min=0.0)
                penalty = penalty + nav_scan_weight * torch.mean(close, dim=1) * not_evading

            grid = self._height_scan_grid()
            if grid is not None:
                wall_score, stair_score = self._height_scan_wall_stair_scores(
                    grid,
                    wall_height_threshold=wall_height_threshold,
                    body_clearance_threshold=body_clearance_threshold,
                    wall_jump_threshold=wall_jump_threshold,
                    stair_min_delta=stair_min_delta,
                    stair_max_delta=stair_max_delta,
                )
                wall_like = (wall_score > wall_score_threshold) & (
                    wall_score > stair_score * wall_vs_stair_margin
                )
                penalty = penalty + height_scan_weight * wall_score * wall_like.float() * not_evading

        return penalty * has_forward_cmd.float()

    def _height_scan_grid(self):
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return None
        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is None or not hasattr(height_scanner.data, "ray_hits_w"):
            return None
        scan = height_scanner.data.pos_w[:, 2:3] - height_scanner.data.ray_hits_w[..., 2]
        if scan.shape[-1] < 256:
            return None
        return scan[:, :256].view(self.env.num_envs, 16, 16)

    def _height_scan_wall_stair_scores(
        self,
        grid,
        y_start: int = 4,
        y_end: int = 12,
        x_start: int = 1,
        x_end: int = 10,
        wall_height_threshold: float = 0.18,
        body_clearance_threshold: float = 0.30,
        wall_jump_threshold: float = 0.16,
        stair_min_delta: float = 0.03,
        stair_max_delta: float = 0.24,
    ):
        """Return geometry scores that separate maze walls from stair edges.

        `grid = scanner_z - hit_z`: negative values mean the ray hit something
        above the scanner plane. Maze walls create tall, laterally continuous
        negative bands. Stairs create moderate forward height discontinuities.
        """
        y0 = max(0, min(int(y_start), 15))
        y1 = max(y0 + 1, min(int(y_end), 16))
        x0 = max(0, min(int(x_start), 15))
        x1 = max(x0 + 2, min(int(x_end), 16))
        features = maze_wall_stair_features(
            grid,
            y_bands=((y0, y1), (y0, y1), (y0, y1)),
            x_start=x0,
            x_end=x1,
            wall_height_threshold=wall_height_threshold,
            body_clearance_threshold=body_clearance_threshold,
            wall_jump_threshold=wall_jump_threshold,
            stair_min_delta=stair_min_delta,
            stair_max_delta=stair_max_delta,
        )
        return features["front_wall_score"], features["stair_score"]

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out).

        惩罚真正的失败（被终止 且 非超时截断），对应 legged_gym `reset_buf * ~time_out_buf` 逻辑。
        防止策略学会"倒地不起"以规避其他惩罚。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        goal_reached = self._get_goal_reached_mask(term_mgr)
        if goal_reached is not None:
            failure = failure & ~goal_reached
        return failure.float()

    def _reward_action_smoothness(self):
        """2nd-order action smoothness penalty (squared action acceleration).

        动作二阶平滑惩罚（动作加速度的平方和）。
        a_t - 2*a_{t-1} + a_{t-2} 的平方和，比一阶 action_rate 更能抑制抖动。
        在 env 上缓存 prev_prev_action 以跨调用维持状态。
        """
        curr = self.env.action_manager.action
        prev = self.env.action_manager.prev_action
        if not hasattr(self.env, "_smooth_prev_prev"):
            self.env._smooth_prev_prev = prev.clone()
        accel = curr - 2.0 * prev + self.env._smooth_prev_prev
        self.env._smooth_prev_prev = prev.clone()
        return torch.sum(torch.square(accel), dim=1)

    def _reward_energy(self):
        """Energy penalty: sum of |torque × joint_velocity|.

        能耗惩罚：扭矩绝对值与关节角速度的乘积之和。
        对应赛题 energy 评分项，鼓励高效步态。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)

    def _reward_correct_base_height(
        self,
        target_height: float = 0.38,
        use_height_scan: bool = False,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_end: int = 4,
    ):
        """Penalize deviation of base height from target (squared).

        惩罚机身高度偏离目标高度（平方误差）。
        对应赛题 posture 评分项。Go2 标准站立高度 ≈ 0.38 m。

        Args:
            target_height: Target base height in meters. / 目标机身高度（米）。
        """
        asset = self._get_robot_asset()
        base_height = asset.data.root_pos_w[:, 2]
        if use_height_scan:
            scan_height = self._estimate_base_height_from_scan(
                asset,
                body_y_start=body_y_start,
                body_y_end=body_y_end,
                near_x_end=near_x_end,
            )
            if scan_height is not None:
                base_height = scan_height
        return torch.square(base_height - target_height)

    def _reward_feet_regulation(
        self,
        height_scale: float = 0.01,
        use_height_scan: bool = True,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_end: int = 4,
    ):
        """Penalize horizontal foot speed when feet are close to the terrain."""
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]

        foot_xy_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        root_z = asset.data.root_pos_w[:, 2]

        ground_z = None
        if use_height_scan:
            ground_z = self._estimate_ground_z_from_scan(
                body_y_start=body_y_start,
                body_y_end=body_y_end,
                near_x_end=near_x_end,
            )
        if ground_z is None:
            ground_z = torch.zeros_like(root_z)

        foot_height = torch.clamp(foot_z - ground_z.unsqueeze(1), min=0.0)
        low_foot_gate = torch.exp(-foot_height / max(float(height_scale), 1.0e-6))
        foot_xy_speed_sq = torch.sum(torch.square(foot_xy_vel), dim=-1)
        return torch.sum(foot_xy_speed_sq * low_foot_gate, dim=1)

    def _estimate_base_height_from_scan(
        self,
        asset,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_end: int = 4,
    ):
        ground_z = self._estimate_ground_z_from_scan(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_end=near_x_end,
        )
        if ground_z is None:
            return None
        return asset.data.root_pos_w[:, 2] - ground_z

    def _estimate_ground_z_from_scan(
        self,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_end: int = 4,
    ):
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return None

        sensor = self.env.scene.sensors.get("height_scanner")
        if sensor is None or not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
            return None

        hits_z = sensor.data.ray_hits_w[..., 2]
        if hits_z.dim() != 2:
            hits_z = hits_z.view(self.env.num_envs, -1)
        valid = torch.isfinite(hits_z)

        if hits_z.shape[1] >= 256:
            grid = hits_z[:, :256].view(self.env.num_envs, 16, 16)
            valid_grid = valid[:, :256].view(self.env.num_envs, 16, 16)
            y0 = max(0, min(int(body_y_start), 15))
            y1 = max(y0 + 1, min(int(body_y_end), 16))
            x1 = max(1, min(int(near_x_end), 16))
            sample = grid[:, y0:y1, :x1]
            sample_valid = valid_grid[:, y0:y1, :x1]
            count = sample_valid.float().sum(dim=(1, 2))
            summed = torch.where(sample_valid, sample, torch.zeros_like(sample)).sum(dim=(1, 2))
        else:
            count = valid.float().sum(dim=1)
            summed = torch.where(valid, hits_z, torch.zeros_like(hits_z)).sum(dim=1)

        if not torch.any(count > 0.0):
            return None
        return summed / count.clamp_min(1.0)

    def _reward_hip_to_default(self):
        """Penalize hip joint deviation from default angle (squared sum).

        惩罚髋关节偏离默认角度（平方和）。
        Go2 髋关节为每腿第 0 个关节：FL=0, FR=3, RL=6, RR=9。
        辅助保持自然站姿，防止外八或内八步态。
        """
        asset = self._get_robot_asset()
        hip_idx = [0, 3, 6, 9]
        hip_dev = asset.data.joint_pos[:, hip_idx] - asset.data.default_joint_pos[:, hip_idx]
        return torch.sum(torch.square(hip_dev), dim=1)

    # -----------------------------------------------------------------------
    # Gait quality rewards (Round-4 addition)
    # 步态质量奖励（第四轮新增）
    # -----------------------------------------------------------------------

    def _reward_dof_vel(self):
        """Penalize large joint velocities (L2 norm of joint velocity vector).

        惩罚过大的关节速度（关节速度向量的 L2 范数）。
        关节速度无约束是步态"鬼畜"的根本原因之一；该惩罚鼓励流畅、低速摆腿。
        Ref: agent_diy/feature/reward_process.py
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_base_lateral_vel(self, command_name: str = "base_velocity"):
        """Penalize untracked lateral (Y-axis) velocity to prevent crab walking.

        惩罚**未被指令覆盖**的侧向速度，防止螃蟹步或横向侧滑。

        注意：不能无条件惩罚 root_lin_vel_b[:,1]，因为速度指令中含有 lin_vel_y
        分量（[-0.3, 0.3]），机器人有时被要求侧向行走。无条件惩罚会与
        track_lin_vel_xy 正向奖励产生梯度冲突。

        正确做法：惩罚 (实际侧向速度 - 指令侧向速度)²，即"横移误差"。
        机器人照指令侧走时误差≈0，不罚；无指令自发侧漂时才受惩罚。
        Ref: custom_rewards.py penalize_base_lat_vel_l2 (error-based variant)
        """
        asset = self._get_robot_asset()
        cmd_vy = self.env.command_manager.get_command(command_name)[:, 1]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        return torch.square(actual_vy - cmd_vy)

    def _reward_air_time_variance_penalty(self):
        """Penalize variance in per-foot air time to enforce gait rhythm symmetry.

        惩罚各脚滞空时间的方差，促进对称步态节律（如 trot）。
        对称步态更稳定、更节能，有助于提高 posture 与 energy 评分。
        使用 clamp(max=0.5s) 避免单脚异常值主导方差计算。
        Ref: agent_diy/feature/reward_process.py
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clamp(last_air_time, max=0.5), dim=1)

    def _reward_pivot_turning(
        self,
        lin_vel_threshold: float = 0.2,
        ang_vel_threshold: float = 0.5,
    ):
        """Penalize pivoting: rotating on the spot without lifting feet.

        惩罚原地蹭脚转弯——线速度低（< lin_vel_threshold）但偏航角速度高
        （> ang_vel_threshold）时，仍有多脚接触地面则给予惩罚。
        鼓励通过迈步（lift & place）完成转弯，而非用脚蹭地旋转，减少关节磨损和
        步态不自然。
        Ref: custom_rewards.py penalize_pivot_turning

        Args:
            lin_vel_threshold: Max horizontal speed (m/s) for pivoting detection.
                               判定原地旋转的最大水平速度阈值 (m/s)。
            ang_vel_threshold: Min yaw rate (rad/s) for pivoting detection.
                               判定原地旋转的最小偏航角速度阈值 (rad/s)。
        """
        asset = self._get_robot_asset()
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b

        horizontal_speed = torch.norm(base_lin_vel[:, :2], dim=1)
        is_pivoting = (horizontal_speed < lin_vel_threshold) & (
            torch.abs(base_ang_vel[:, 2]) > ang_vel_threshold
        )

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        feet_in_contact = contact_forces > 1.0
        num_contacting_feet = torch.sum(feet_in_contact.float(), dim=1)

        return num_contacting_feet * is_pivoting.float()

    # -----------------------------------------------------------------------
    # Goal-reaching rewards (activated only in track terrain)
    # 目标到达奖励（仅 track 地形时激活）
    # -----------------------------------------------------------------------

    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the maze exit (returns 1.0 when distance < 0.6 m).
        到达迷宫出口奖励（distance < 0.6 m 时返回 1.0）。

        Note:
            The threshold must match the threshold of _goal_reached_termination
            in tools/unitree_rl_lab/.../velocity_env_cfg.py (currently 0.6 m),
            otherwise a "termination-reward dead zone" will appear.
            threshold 必须与 tools/unitree_rl_lab/.../velocity_env_cfg.py 中
            _goal_reached_termination 的 threshold 一致（当前 0.6 m），
            否则会产生"终止-奖励死区"。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_goal_progress(self, scale: float = 1.0):
        """Reward one-step progress toward the active track goal."""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)

        if not hasattr(self.env, "_prev_goal_distance"):
            self.env._prev_goal_distance = dist.detach()
            return torch.zeros_like(dist)

        progress = self.env._prev_goal_distance - dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        progress = torch.where(reset_mask, torch.zeros_like(progress), progress)
        self.env._prev_goal_distance = dist.detach()
        return scale * progress

    def _reward_heading_to_goal(self, std: float = 0.75):
        """Reward aligning body x-axis with the current goal direction."""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_vec = self.env.goal_positions[:, :2] - robot_pos
        goal_yaw = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
        yaw = self._robot_yaw_w(robot)
        if yaw is None:
            self._warn_reward_once(
                "heading_to_goal_missing_yaw",
                "[heading_to_goal] robot data has neither heading_w nor root_quat_w; reward disabled",
            )
            return torch.zeros(self.env.num_envs, device=self.env.device)
        yaw_error = torch.atan2(torch.sin(goal_yaw - yaw), torch.cos(goal_yaw - yaw))
        return torch.exp(-torch.square(yaw_error / max(std, 1.0e-6)))

    def _reward_navigation_time(self):
        """Dense time penalty for track/navigation stages."""
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_navigation_termination(self):
        """Penalty for navigation failures, excluding successful goal reach."""
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        goal_reached = self._get_goal_reached_mask(term_mgr)
        if goal_reached is not None:
            failure = failure & ~goal_reached
        return failure.float()

    @staticmethod
    def _get_goal_reached_mask(term_mgr):
        goal_reached = getattr(term_mgr, "goal_reached", None)
        if goal_reached is not None:
            return goal_reached
        active_terms = getattr(term_mgr, "active_terms", None)
        if active_terms is not None and "goal_reached" in active_terms and hasattr(term_mgr, "get_term"):
            return term_mgr.get_term("goal_reached")
        return None

    def _robot_yaw_w(self, robot):
        heading = getattr(robot.data, "heading_w", None)
        if heading is not None:
            return heading

        quat = getattr(robot.data, "root_quat_w", None)
        if quat is None:
            return None

        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def _warn_reward_once(self, key: str, message: str):
        if not hasattr(self.env, "_reward_warning_keys"):
            self.env._reward_warning_keys = set()
        if key in self.env._reward_warning_keys:
            return
        self.env._reward_warning_keys.add(key)
        self._log_reward_warning(message)

    def _log_reward_warning(self, message: str, *args):
        formatted = message % args if args else message
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = getattr(self.env, "logger", None)
        if logger is not None:
            logger.warning(formatted)
        else:
            LOGGER.warning(formatted)

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame (the larger the better).
        前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）。

        This is an example reward that demonstrates how to read the robot state and
        build a dense signal.
        示例性 reward，展示如何读取机器人状态并构造 dense signal。
        """
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]
