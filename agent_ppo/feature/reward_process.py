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

import math

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    # -----------------------------------------------------------------------
    # Locomotion quality rewards
    # 运动质量奖励
    # -----------------------------------------------------------------------

    def _reward_track_lin_vel_xy(
        self,
        std: float = 0.40,
        command_name: str = "base_velocity",
        min_tracking_vx: float = 0.0,
    ):
        """Track XY velocity while optionally lifting low vx commands."""
        asset = self._get_robot_asset()
        command = self.env.command_manager.get_command(command_name)
        effective_command_xy = command[:, :2].clone()
        if min_tracking_vx > 0.0:
            effective_command_xy[:, 0] = torch.clamp(effective_command_xy[:, 0], min=min_tracking_vx)
        lin_vel_error = torch.sum(torch.square(effective_command_xy - asset.data.root_lin_vel_b[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / max(std * std, 1.0e-6))

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
        """Reward terrain-aware swing-foot clearance to reduce stair-edge tripping.

        Active only for moving commands and swing feet. The Gaussian target keeps
        the reward bounded: it encourages enough clearance to step over edges,
        but does not reward unnecessarily high, energy-wasting leg lifts.

        The dynamic target is based on local adjacent height-scan deltas, not a
        full-window max-min range, so several stair levels in the scan window are
        less likely to be accumulated into one exaggerated step height.
        """
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
        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is not None:
            scan = height_scanner.data.pos_w[:, 2:3] - height_scanner.data.ray_hits_w[..., 2]
            grid = scan.view(self.env.num_envs, 16, 16)
            forward_window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
            if forward_window.shape[-1] > 1 and forward_window.shape[1] > 0:
                step_deltas = torch.abs(forward_window[:, :, 1:] - forward_window[:, :, :-1]).flatten(1)
                local_step = torch.quantile(step_deltas, delta_quantile, dim=1)
                terrain_extra = torch.clamp(
                    terrain_height_scale * local_step,
                    0.0,
                    max_terrain_extra_height,
                )

        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        dynamic_target_height = target_height + terrain_extra + speed_extra
        height_error = (foot_height - dynamic_target_height.unsqueeze(1)) / max(std, 1e-6)
        clearance_reward = torch.exp(-torch.square(height_error))
        is_moving = command_speed > 0.1
        return torch.sum(clearance_reward * swing.float(), dim=1) * is_moving.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_feet_swing_forward(
        self,
        command_name: str = "base_velocity",
        target_forward: float = 0.10,
        std: float = 0.08,
        min_command: float = 0.10,
    ):
        """Reward swing feet moving forward enough to follow the body on stairs.

        This complements feet_clearance: clearance helps the foot avoid stair
        edges vertically, while this term encourages the swing foot—especially
        the rear feet—to reach forward instead of taking a short high step that
        still lands on the stair edge.
        """
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
        foot_rel_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - asset.data.root_pos_w[:, :2].unsqueeze(1)
        foot_forward, foot_lateral = self._project_world_xy_to_body_xy(asset, foot_rel_xy)

        command = self.env.command_manager.get_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        fallback = torch.zeros_like(command_dir)
        fallback[:, 0] = 1.0
        command_dir = torch.where((command_speed > 1.0e-6).unsqueeze(1), command_dir, fallback)
        foot_command_forward = (
            foot_forward * command_dir[:, 0].unsqueeze(1)
            + foot_lateral * command_dir[:, 1].unsqueeze(1)
        )
        shortfall = torch.clamp(target_forward - foot_command_forward, min=0.0)
        forward_reward = torch.exp(-torch.square(shortfall / max(std, 1e-6)))

        has_command = command_speed > min_command
        return torch.sum(forward_reward * swing.float(), dim=1) * has_command.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_height_scan_feet_clearance(
        self,
        command_name: str = "base_velocity",
        base_target_height: float = 0.08,
        step_height_scale: float = 0.75,
        max_extra_height: float = 0.12,
        speed_height_scale: float = 0.01,
        std: float = 0.05,
        min_command_speed: float = 0.10,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        up_min_height: float = 0.025,
        up_max_height: float = 0.20,
        wall_min_height: float = 0.24,
        min_step_score: float = 0.02,
        max_wall_score: float = 0.08,
        log_interval: int = 50,
    ):
        """Reward swing-foot clearance only when height scan sees a climbable up-step.

        This is deliberately scan-gated rather than terrain-label-gated. It does
        not encode a world/stair heading, so it should not create the previous
        diagonal-path dependency by itself.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        gate = self._height_scan_semantic_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
            up_min_height=up_min_height,
            up_max_height=up_max_height,
            wall_min_height=wall_min_height,
            min_step_score=min_step_score,
            max_wall_score=max_wall_score,
            log_interval=log_interval,
        )

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        command = self.env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        up_gate = gate["up_step"].float()
        step_extra = torch.clamp(step_height_scale * torch.clamp(gate["step_delta"], min=0.0), 0.0, max_extra_height)
        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        target_height = base_target_height + step_extra + speed_extra
        height_error = (foot_height - target_height.unsqueeze(1)) / max(std, 1.0e-6)
        reward = torch.exp(-torch.square(height_error))
        active = (command_speed > min_command_speed).float() * up_gate
        value = torch.sum(reward * swing.float(), dim=1) / max(len(asset_cfg.body_ids), 1)
        value = value * active

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["height_scan_feet_clearance_reward_mean"] = float(value.detach().float().mean().item()) if value.numel() else 0.0
        debug["height_scan_feet_clearance_active_ratio"] = float((active.detach().float() > 0.0).float().mean().item()) if active.numel() else 0.0
        self.env._stair_gate_debug = debug
        return value

    def _reward_height_scan_wall_reject(
        self,
        command_name: str = "base_velocity",
        min_forward_speed: float = 0.05,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        wall_min_height: float = 0.24,
        min_step_score: float = 0.02,
        max_wall_score: float = 0.08,
        log_interval: int = 50,
    ):
        """Penalty magnitude for driving forward into too-tall scan obstacles."""
        asset = self._get_robot_asset()
        gate = self._height_scan_semantic_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
            wall_min_height=wall_min_height,
            min_step_score=min_step_score,
            max_wall_score=max_wall_score,
            log_interval=log_interval,
        )
        command = self.env.command_manager.get_command(command_name)
        commanded_forward = torch.clamp(command[:, 0], min=0.0)
        actual_forward = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
        pushing_forward = torch.maximum(commanded_forward, actual_forward)
        value = gate["wall"].float() * torch.clamp(pushing_forward - min_forward_speed, min=0.0)

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["height_scan_wall_reject_reward_mean"] = float(value.detach().float().mean().item()) if value.numel() else 0.0
        debug["height_scan_wall_reject_active_ratio"] = float((value.detach().float() > 0.0).float().mean().item()) if value.numel() else 0.0
        self.env._stair_gate_debug = debug
        return value

    def _reward_stair_relax_ang_vel_xy(
        self,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        log_interval: int = 50,
    ):
        """Positive compensation for pitch/roll angular-velocity penalty on steps."""
        asset = self._get_robot_asset()
        gate = self._height_scan_semantic_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
            log_interval=log_interval,
        )
        active = (gate["up_step"] | gate["down_step"]).float()
        value = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * active
        self._set_stair_relax_debug("stair_relax_ang_vel_xy", value, active)
        return value

    def _reward_stair_relax_base_height(
        self,
        target_height: float = 0.38,
        use_height_scan: bool = False,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        log_interval: int = 50,
    ):
        """Positive compensation for base-height penalty on up/down steps."""
        asset = self._get_robot_asset()
        gate = self._height_scan_semantic_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
            log_interval=log_interval,
        )
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
        active = (gate["up_step"] | gate["down_step"]).float()
        value = torch.square(base_height - target_height) * active
        self._set_stair_relax_debug("stair_relax_base_height", value, active)
        return value

    def _reward_stair_relax_joint_position(
        self,
        command_name: str = "base_velocity",
        stand_still_scale: float = 2.0,
        velocity_threshold: float = 0.1,
        cmd_threshold: float = 0.1,
        ang_cmd_threshold: float = 0.2,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        log_interval: int = 50,
    ):
        """Positive compensation for default-joint-pose penalty on steps."""
        asset = self._get_robot_asset()
        gate = self._height_scan_semantic_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
            log_interval=log_interval,
        )
        cmd = self.env.command_manager.get_command(command_name)
        cmd_xy = torch.linalg.norm(cmd[:, :2], dim=1)
        cmd_yaw = torch.abs(cmd[:, 2])
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        deviation = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        is_moving = torch.logical_or(
            torch.logical_or(cmd_xy > cmd_threshold, cmd_yaw > ang_cmd_threshold),
            body_vel > velocity_threshold,
        )
        raw_penalty = torch.where(is_moving, deviation, stand_still_scale * deviation)
        active = (gate["up_step"] | gate["down_step"]).float()
        value = raw_penalty * active
        self._set_stair_relax_debug("stair_relax_joint_position", value, active)
        return value

    def _set_stair_relax_debug(self, name: str, value, active):
        debug = getattr(self.env, "_stair_gate_debug", {})
        debug[f"{name}_reward_mean"] = float(value.detach().float().mean().item()) if value.numel() else 0.0
        debug[f"{name}_active_ratio"] = float((active.detach().float() > 0.0).float().mean().item()) if active.numel() else 0.0
        self.env._stair_gate_debug = debug

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

    def _reward_commanded_still_penalty(
        self,
        command_name: str = "base_velocity",
        cmd_threshold: float = 0.20,
        still_speed_threshold: float = 0.12,
    ):
        """Penalize staying nearly still when an XY velocity command is present."""
        asset = self._get_robot_asset()
        command = self.env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

        commanded_to_move = command_speed > cmd_threshold
        stillness = torch.clamp(
            (still_speed_threshold - body_speed) / max(still_speed_threshold, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        return commanded_to_move.float() * stillness

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

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out).

        惩罚真正的失败（被终止 且 非超时截断），对应 legged_gym `reset_buf * ~time_out_buf` 逻辑。
        防止策略学会"倒地不起"以规避其他惩罚。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
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

    def _height_scan_grid(self):
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return None
        sensor = self.env.scene.sensors.get("height_scanner")
        if sensor is None or not hasattr(sensor, "data"):
            return None
        ray_hits = getattr(sensor.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.shape[-2] < 256:
            return None
        scan = sensor.data.pos_w[:, 2:3] - ray_hits[..., 2]
        return scan[:, :256].view(self.env.num_envs, 16, 16)

    def _ground_z_window_from_scan(
        self,
        body_y_start: int = 5,
        body_y_end: int = 11,
        x_start: int = 0,
        x_end: int = 4,
    ):
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return None
        sensor = self.env.scene.sensors.get("height_scanner")
        if sensor is None or not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
            return None

        hits_z = sensor.data.ray_hits_w[..., 2]
        if hits_z.dim() != 2:
            hits_z = hits_z.view(self.env.num_envs, -1)
        if hits_z.shape[1] < 256:
            return None

        grid = hits_z[:, :256].view(self.env.num_envs, 16, 16)
        valid = torch.isfinite(grid)
        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(x_start), 15))
        x1 = max(x0 + 1, min(int(x_end), 16))
        sample = grid[:, y0:y1, x0:x1]
        sample_valid = valid[:, y0:y1, x0:x1]
        count = sample_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        return torch.where(sample_valid, sample, torch.zeros_like(sample)).sum(dim=(1, 2)) / count

    def _height_scan_wall_stair_scores(
        self,
        grid,
        y_start: int = 5,
        y_end: int = 11,
        x_start: int = 0,
        x_end: int = 10,
        stair_min_delta: float = 0.03,
        stair_max_delta: float = 0.24,
    ):
        y0 = max(0, min(int(y_start), 15))
        y1 = max(y0 + 1, min(int(y_end), 16))
        x0 = max(0, min(int(x_start), 15))
        x1 = max(x0 + 2, min(int(x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        if window.shape[1] < 1 or window.shape[2] < 2:
            return zeros, zeros
        deltas = torch.abs(window[:, :, 1:] - window[:, :, :-1])
        stair_like = (deltas > stair_min_delta) & (deltas < stair_max_delta)
        wall_like = deltas >= stair_max_delta
        stair_score = stair_like.float().mean(dim=(1, 2))
        wall_score = wall_like.float().mean(dim=(1, 2))
        return wall_score, stair_score

    def _stair_gate_from_scan(
        self,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        stair_min_delta: float = 0.03,
        stair_max_delta: float = 0.24,
    ):
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        grid = self._height_scan_grid()
        if grid is None:
            self._set_stair_gate_debug(
                "stair_scan",
                available=zeros,
                stair_gate=zeros,
                stair_score=zeros,
                wall_score=zeros,
                local_step=zeros,
            )
            return zeros, zeros

        wall_score, stair_score = self._height_scan_wall_stair_scores(
            grid,
            y_start=body_y_start,
            y_end=body_y_end,
            x_start=near_x_start,
            x_end=near_x_end,
            stair_min_delta=stair_min_delta,
            stair_max_delta=stair_max_delta,
        )

        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(near_x_start), 15))
        x1 = max(x0 + 2, min(int(near_x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 1 or window.shape[2] < 2:
            self._set_stair_gate_debug(
                "stair_scan",
                available=zeros,
                stair_gate=zeros,
                stair_score=stair_score,
                wall_score=wall_score,
                local_step=zeros,
            )
            return zeros, zeros

        step_deltas = torch.abs(window[:, :, 1:] - window[:, :, :-1])
        step_like = (step_deltas > stair_min_delta) & (step_deltas < stair_max_delta)
        local_step = torch.where(step_like, step_deltas, torch.zeros_like(step_deltas)).amax(dim=(1, 2))
        gate = (stair_score > min_stair_score) & (wall_score < max_wall_score) & (local_step > 0.0)
        self._set_stair_gate_debug(
            "stair_scan",
            available=torch.ones_like(local_step),
            stair_gate=gate.float(),
            stair_score=stair_score,
            wall_score=wall_score,
            local_step=local_step,
        )
        return gate.float(), local_step

    def _height_scan_semantic_gate(
        self,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
        up_min_height: float = 0.025,
        up_max_height: float = 0.20,
        down_min_height: float = 0.025,
        down_max_height: float = 0.22,
        wall_min_height: float = 0.24,
        min_step_score: float = 0.02,
        max_wall_score: float = 0.08,
        log_interval: int = 100,
    ):
        """Classify local height-scan geometry without terrain labels.

        The sign comes from world ground height: front_z - near_z.
        Positive means the target foothold is higher than the current nearby
        ground (up-step), negative means a down-step. Large discontinuities are
        treated as walls/too-tall obstacles and excluded from stair rewards.
        """
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        near_z = self._ground_z_window_from_scan(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            x_start=near_x_start,
            x_end=near_x_end,
        )
        front_z = self._ground_z_window_from_scan(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            x_start=front_x_start,
            x_end=front_x_end,
        )
        grid = self._height_scan_grid()
        if near_z is None or front_z is None or grid is None:
            self._set_stair_gate_debug(
                "height_scan",
                available=zeros,
                up_step=zeros,
                down_step=zeros,
                wall=zeros,
                flat=zeros,
                step_delta=zeros,
                step_score=zeros,
                wall_score=zeros,
            )
            self._log_height_scan_gate_status(
                available=zeros,
                up_step=zeros,
                down_step=zeros,
                wall=zeros,
                flat=zeros,
                step_delta=zeros,
                step_score=zeros,
                wall_score=zeros,
                reason="height_scan_unavailable",
                log_interval=log_interval,
            )
            return {
                "available": zeros,
                "up_step": zeros.bool(),
                "down_step": zeros.bool(),
                "wall": zeros.bool(),
                "flat": torch.ones_like(zeros).bool(),
                "step_delta": zeros,
                "step_score": zeros,
                "wall_score": zeros,
            }

        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(near_x_start), 15))
        x1 = max(x0 + 2, min(int(front_x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 1 or window.shape[2] < 2:
            step_score = zeros
            wall_score = zeros
        else:
            adjacent_delta = torch.abs(window[:, :, 1:] - window[:, :, :-1])
            step_like = (adjacent_delta >= min(up_min_height, down_min_height)) & (adjacent_delta <= max(up_max_height, down_max_height))
            wall_like = adjacent_delta >= wall_min_height
            step_score = step_like.float().mean(dim=(1, 2))
            wall_score = wall_like.float().mean(dim=(1, 2))

        step_delta = front_z - near_z
        wall = (
            (wall_score > max_wall_score)
            | (step_delta > wall_min_height)
            | torch.isnan(step_delta)
        )
        stair_like = (step_score > min_step_score) & (~wall)
        up_step = stair_like & (step_delta >= up_min_height) & (step_delta <= up_max_height)
        down_step = stair_like & (step_delta <= -down_min_height) & (step_delta >= -down_max_height)
        flat = ~(up_step | down_step | wall)
        available = torch.ones_like(step_delta)

        self._set_stair_gate_debug(
            "height_scan",
            available=available,
            up_step=up_step.float(),
            down_step=down_step.float(),
            wall=wall.float(),
            flat=flat.float(),
            step_delta=step_delta,
            step_score=step_score,
            wall_score=wall_score,
        )
        self._log_height_scan_gate_status(
            available=available,
            up_step=up_step.float(),
            down_step=down_step.float(),
            wall=wall.float(),
            flat=flat.float(),
            step_delta=step_delta,
            step_score=step_score,
            wall_score=wall_score,
            reason="ok",
            log_interval=log_interval,
        )
        return {
            "available": available,
            "up_step": up_step,
            "down_step": down_step,
            "wall": wall,
            "flat": flat,
            "step_delta": step_delta,
            "step_score": step_score,
            "wall_score": wall_score,
        }

    def _log_height_scan_gate_status(
        self,
        available,
        up_step,
        down_step,
        wall,
        flat,
        step_delta,
        step_score,
        wall_score,
        reason: str,
        log_interval: int = 100,
    ):
        if not hasattr(self.env, "_height_scan_gate_log_count"):
            self.env._height_scan_gate_log_count = 0
        self.env._height_scan_gate_log_count += 1
        count = self.env._height_scan_gate_log_count
        interval = max(int(log_interval), 1)
        if count != 1 and count % interval != 0:
            return

        def ratio(value):
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return 0.0
            return float((value.detach().float() > 0.0).float().mean().item())

        def mean(value):
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return 0.0
            return float(value.detach().float().mean().item())

        self._log_reward_warning(
            "[height_scan_gate] call=%d reason=%s available=%.3f up=%.3f down=%.3f wall=%.3f flat=%.3f "
            "step_delta_mean=%.4f step_score=%.4f wall_score=%.4f",
            count,
            reason,
            ratio(available),
            ratio(up_step),
            ratio(down_step),
            ratio(wall),
            ratio(flat),
            mean(step_delta),
            mean(step_score),
            mean(wall_score),
        )

    def _set_stair_gate_debug(self, prefix: str, **values):
        debug = getattr(self.env, "_stair_gate_debug", {})
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                detached = value.detach().float()
                debug[f"{prefix}_{key}_mean"] = float(detached.mean().item()) if detached.numel() else 0.0
                debug[f"{prefix}_{key}_active_ratio"] = (
                    float((detached > 0.0).float().mean().item()) if detached.numel() else 0.0
                )
            else:
                debug[f"{prefix}_{key}"] = float(value)
        self.env._stair_gate_debug = debug

    def _standard_terrain_mask(self, target_names):
        target_names = tuple(target_names)
        false_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_gen_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
        if terrain is None or terrain_types is None or terrain_gen_cfg is None or not sub_terrains:
            self._log_standard_terrain_mask_status(
                target_names,
                false_mask,
                reason="missing_terrain_or_types_or_subterrains",
                terrain=terrain,
                terrain_types=terrain_types,
                terrain_gen_cfg=terrain_gen_cfg,
                sub_terrains=sub_terrains,
            )
            return false_mask
        if getattr(terrain_gen_cfg, "track_length", None) is not None:
            self._log_standard_terrain_mask_status(
                target_names,
                false_mask,
                reason="track_terrain_generator",
                terrain=terrain,
                terrain_types=terrain_types,
                terrain_gen_cfg=terrain_gen_cfg,
                sub_terrains=sub_terrains,
            )
            return false_mask

        terrain_names = list(sub_terrains.keys())
        proportions = torch.tensor(
            [float(getattr(sub_terrains[name], "proportion", 0.0)) for name in terrain_names],
            dtype=torch.float32,
            device=self.env.device,
        )
        if torch.sum(proportions) <= 1.0e-9:
            self._log_standard_terrain_mask_status(
                target_names,
                false_mask,
                reason="zero_subterrain_proportions",
                terrain=terrain,
                terrain_types=terrain_types,
                terrain_gen_cfg=terrain_gen_cfg,
                sub_terrains=sub_terrains,
                terrain_names=terrain_names,
            )
            return false_mask
        proportions = proportions / torch.sum(proportions)
        cumulative = torch.cumsum(proportions, dim=0)
        num_cols = int(getattr(terrain_gen_cfg, "num_cols", 0) or 0)
        if num_cols <= 0:
            terrain_origins = getattr(terrain, "terrain_origins", None)
            if terrain_origins is not None and getattr(terrain_origins, "ndim", 0) >= 2:
                num_cols = int(terrain_origins.shape[1])
        if num_cols <= 0:
            self._log_standard_terrain_mask_status(
                target_names,
                false_mask,
                reason="num_cols_unavailable",
                terrain=terrain,
                terrain_types=terrain_types,
                terrain_gen_cfg=terrain_gen_cfg,
                sub_terrains=sub_terrains,
                terrain_names=terrain_names,
            )
            return false_mask

        cols = terrain_types.long().clamp(0, num_cols - 1)
        choices = cols.float() / max(num_cols, 1) + 0.001
        terrain_idx = torch.searchsorted(cumulative, choices).clamp(0, len(terrain_names) - 1)
        mask = torch.zeros_like(false_mask)
        for idx, name in enumerate(terrain_names):
            if name in target_names:
                mask |= terrain_idx == idx

        self._log_standard_terrain_mask_status(
            target_names,
            mask,
            reason="ok",
            terrain=terrain,
            terrain_types=terrain_types,
            terrain_gen_cfg=terrain_gen_cfg,
            sub_terrains=sub_terrains,
            terrain_names=terrain_names,
            proportions=proportions,
            num_cols=num_cols,
            cols=cols,
            cumulative=cumulative,
        )
        return mask

    def _log_standard_terrain_mask_status(
        self,
        target_names,
        mask,
        reason: str,
        terrain=None,
        terrain_types=None,
        terrain_gen_cfg=None,
        sub_terrains=None,
        terrain_names=None,
        proportions=None,
        num_cols: int = 0,
        cols=None,
        cumulative=None,
        log_interval: int = 200,
    ):
        log_key = "standard_terrain_mask_" + "_".join(target_names)
        counter_name = "_standard_terrain_mask_log_counts"
        if not hasattr(self.env, counter_name):
            setattr(self.env, counter_name, {})
        counts = getattr(self.env, counter_name)
        count_key = f"{log_key}:{reason}"
        count = int(counts.get(count_key, 0)) + 1
        counts[count_key] = count
        interval = max(int(log_interval), 1)
        if count != 1 and count % interval != 0:
            return

        names_text = list(terrain_names) if terrain_names is not None else (
            list(sub_terrains.keys()) if sub_terrains else []
        )
        proportions_text = []
        if proportions is not None:
            proportions_text = [float(x) for x in proportions.detach().cpu().tolist()]
        elif sub_terrains:
            total = sum(float(getattr(sub_terrains[name], "proportion", 0.0)) for name in names_text)
            if total > 1.0e-9:
                proportions_text = [
                    float(getattr(sub_terrains[name], "proportion", 0.0)) / total for name in names_text
                ]

        col_ranges = []
        if cumulative is not None and num_cols > 0 and names_text:
            col_ranges = []
            start = 0
            cumulative_cpu = cumulative.detach().cpu().tolist()
            for idx, name in enumerate(names_text):
                end = min(num_cols - 1, max(start, int(math.ceil(cumulative_cpu[idx] * num_cols) - 1)))
                col_ranges.append(f"{name}:{start}-{end}")
                start = end + 1

        level_summary = "unavailable"
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is not None:
            levels = terrain_levels.long()
            level_summary = (
                f"min={int(levels.min().detach().item())} "
                f"max={int(levels.max().detach().item())} "
                f"mean={float(levels.float().mean().detach().item()):.2f}"
            )

        if cols is not None and cols.numel() > 0:
            terrain_type_min = int(cols.min().detach().item())
            terrain_type_max = int(cols.max().detach().item())
        elif terrain_types is not None and getattr(terrain_types, "numel", lambda: 0)() > 0:
            terrain_type_min = int(terrain_types.long().min().detach().item())
            terrain_type_max = int(terrain_types.long().max().detach().item())
        else:
            terrain_type_min = -1
            terrain_type_max = -1

        self._log_reward_warning(
            "[standard_terrain_mask] call=%d reason=%s targets=%s active=%d/%d "
            "terrain_present=%s terrain_types_present=%s terrain_gen_present=%s subterrains_present=%s "
            "names=%s proportions=%s num_cols=%d col_ranges=%s terrain_type_minmax=(%d,%d) terrain_levels=%s",
            count,
            reason,
            target_names,
            int(mask.sum().detach().item()),
            int(self.env.num_envs),
            terrain is not None,
            terrain_types is not None,
            terrain_gen_cfg is not None,
            bool(sub_terrains),
            names_text,
            proportions_text,
            int(num_cols or 0),
            col_ranges,
            terrain_type_min,
            terrain_type_max,
            level_summary,
        )

    def _reset_mask(self):
        term_mgr = getattr(self.env, "termination_manager", None)
        if term_mgr is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        terminated = getattr(term_mgr, "terminated", None)
        time_outs = getattr(term_mgr, "time_outs", None)
        if terminated is None and time_outs is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if terminated is None:
            return time_outs.bool()
        if time_outs is None:
            return terminated.bool()
        return torch.logical_or(terminated.bool(), time_outs.bool())

    def _command_xy_speed_dir(self, command_name: str = "base_velocity"):
        command = self.env.command_manager.get_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        fallback = torch.zeros_like(command_dir)
        fallback[:, 0] = 1.0
        command_dir = torch.where((command_speed > 1.0e-6).unsqueeze(1), command_dir, fallback)
        return command, command_xy, command_speed, command_dir

    def _robot_yaw_w(self, asset):
        if hasattr(asset.data, "heading_w"):
            return asset.data.heading_w
        if hasattr(asset.data, "root_quat_w"):
            quat = asset.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            return torch.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def _world_xy_velocity(self, asset):
        if hasattr(asset.data, "root_lin_vel_w"):
            return asset.data.root_lin_vel_w[:, :2]
        vel_b = asset.data.root_lin_vel_b[:, :2]
        yaw = self._robot_yaw_w(asset)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.stack(
            (
                cos_yaw * vel_b[:, 0] - sin_yaw * vel_b[:, 1],
                sin_yaw * vel_b[:, 0] + cos_yaw * vel_b[:, 1],
            ),
            dim=1,
        )

    def _body_xy_to_world_xy(self, asset, body_xy):
        yaw = self._robot_yaw_w(asset)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.stack(
            (
                cos_yaw * body_xy[:, 0] - sin_yaw * body_xy[:, 1],
                sin_yaw * body_xy[:, 0] + cos_yaw * body_xy[:, 1],
            ),
            dim=1,
        )

    def _wrap_to_pi(self, angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _goal_direction_w(self, asset):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return None, None
        root_xy = asset.data.root_pos_w[:, :2]
        goal_xy = self.env.goal_positions[:, :2].to(device=root_xy.device, dtype=root_xy.dtype)
        delta = goal_xy - root_xy
        distance = torch.linalg.norm(delta, dim=1)
        fallback = torch.zeros_like(delta)
        fallback[:, 0] = 1.0
        direction = delta / distance.unsqueeze(1).clamp_min(1.0e-6)
        direction = torch.where((distance > 1.0e-6).unsqueeze(1), direction, fallback)
        return direction, distance

    def _command_anchor_direction_w(
        self,
        asset,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.05,
        command_change_threshold: float = 1.0e-4,
    ):
        command, command_xy, command_speed, _ = self._command_xy_speed_dir(command_name)
        current_dir_w = self._body_xy_to_world_xy(asset, command_xy)
        current_dir_w = current_dir_w / torch.linalg.norm(current_dir_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        fallback_dir_w = self._body_xy_to_world_xy(asset, torch.tensor([[1.0, 0.0]], device=self.env.device).repeat(self.env.num_envs, 1))
        current_dir_w = torch.where((command_speed > min_command_speed).unsqueeze(1), current_dir_w, fallback_dir_w)

        yaw = self._robot_yaw_w(asset)
        reset_mask = self._reset_mask()
        needs_init = (
            not hasattr(self.env, "_stair_desired_anchor_dir_w")
            or self.env._stair_desired_anchor_dir_w.shape != current_dir_w.shape
            or not hasattr(self.env, "_stair_desired_anchor_cmd")
            or self.env._stair_desired_anchor_cmd.shape != command.shape
        )
        if needs_init:
            self.env._stair_desired_anchor_dir_w = current_dir_w.detach().clone()
            self.env._stair_desired_anchor_heading_w = yaw.detach().clone()
            self.env._stair_desired_anchor_cmd = command.detach().clone()
            return self.env._stair_desired_anchor_dir_w, self.env._stair_desired_anchor_heading_w, command, command_speed

        prev_cmd = self.env._stair_desired_anchor_cmd.to(command.device)
        command_changed = torch.linalg.norm(command - prev_cmd, dim=1) > command_change_threshold
        refresh = reset_mask | command_changed
        anchor_dir = torch.where(
            refresh.unsqueeze(1),
            current_dir_w.detach(),
            self.env._stair_desired_anchor_dir_w.to(command.device),
        )
        anchor_heading = torch.where(
            refresh,
            yaw.detach(),
            self.env._stair_desired_anchor_heading_w.to(command.device),
        )
        anchor_cmd = torch.where(
            refresh.unsqueeze(1),
            command.detach(),
            prev_cmd,
        )
        self.env._stair_desired_anchor_dir_w = anchor_dir.detach().clone()
        self.env._stair_desired_anchor_heading_w = anchor_heading.detach().clone()
        self.env._stair_desired_anchor_cmd = anchor_cmd.detach().clone()
        return anchor_dir, anchor_heading, command, command_speed

    def _desired_direction_w(
        self,
        asset,
        command_name: str = "base_velocity",
        direction_source: str = "command_anchor",
        min_command_speed: float = 0.05,
    ):
        if direction_source == "goal":
            goal_dir, goal_distance = self._goal_direction_w(asset)
            if goal_dir is not None:
                command = self.env.command_manager.get_command(command_name)
                command_speed = torch.linalg.norm(command[:, :2], dim=1)
                return goal_dir, None, command, command_speed, goal_distance
        anchor_dir, anchor_heading, command, command_speed = self._command_anchor_direction_w(
            asset,
            command_name=command_name,
            min_command_speed=min_command_speed,
        )
        return anchor_dir, anchor_heading, command, command_speed, None

    def _stair_desired_gate(
        self,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        terrain_fallback_weight: float = 0.35,
    ):
        scan_gate, local_step = self._stair_gate_from_scan(
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
        )
        terrain_gate = self._standard_terrain_mask(("pyramid_stairs", "pyramid_stairs_inv")).float()
        fallback_gate = torch.clamp(terrain_gate * max(float(terrain_fallback_weight), 0.0), 0.0, 1.0)
        gate = torch.maximum(scan_gate, fallback_gate)
        return gate, scan_gate, terrain_gate, local_step

    def _set_stair_desired_debug(
        self,
        reward,
        gate,
        scan_gate,
        terrain_gate,
        command_gate,
        projected_vel,
        orthogonal_vel,
        heading_error,
    ):
        def mean_value(value):
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return 0.0
            return float(value.detach().float().mean().item())

        def active_ratio(value):
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return 0.0
            return float((value.detach().float() > 0.0).float().mean().item())

        self.env._stair_desired_debug = {
            "stair_desired_reward_mean": mean_value(reward),
            "stair_desired_gate_ratio": active_ratio(gate),
            "stair_desired_scan_gate_ratio": active_ratio(scan_gate),
            "stair_desired_terrain_gate_ratio": active_ratio(terrain_gate),
            "stair_desired_command_gate_ratio": active_ratio(command_gate),
            "stair_desired_proj_vel_mean": mean_value(projected_vel),
            "stair_desired_orth_vel_mean": mean_value(torch.abs(orthogonal_vel)),
            "stair_desired_heading_error_mean": mean_value(torch.abs(heading_error)),
        }

    def _log_stair_desired_activation(
        self,
        gate,
        scan_gate,
        terrain_gate,
        command_gate,
        projected_vel,
        orthogonal_vel,
        heading_error,
        reason: str,
        log_interval: int = 200,
    ):
        if not hasattr(self.env, "_stair_desired_log_count"):
            self.env._stair_desired_log_count = 0
        self.env._stair_desired_log_count += 1
        count = self.env._stair_desired_log_count
        interval = max(int(log_interval), 1)
        if count != 1 and count % interval != 0:
            return
        active = gate > 0.0
        if torch.any(active):
            mean_proj = float(projected_vel[active].mean().detach().item())
            mean_orth = float(torch.abs(orthogonal_vel[active]).mean().detach().item())
            mean_head = float(torch.abs(heading_error[active]).mean().detach().item())
        else:
            mean_proj = 0.0
            mean_orth = 0.0
            mean_head = 0.0
        self._log_reward_warning(
            "[stair_desired_direction] call=%d reason=%s active=%d/%d scan=%d terrain=%d command=%d "
            "mean_proj=%.3f mean_abs_orth=%.3f mean_abs_heading=%.3f",
            count,
            reason,
            int(active.sum().detach().item()),
            int(self.env.num_envs),
            int((scan_gate > 0.0).sum().detach().item()),
            int((terrain_gate > 0.0).sum().detach().item()),
            int((command_gate > 0.0).sum().detach().item()),
            mean_proj,
            mean_orth,
            mean_head,
        )

    def _project_world_xy_to_body_xy(self, asset, vectors_w_xy):
        """Project world XY vectors into the robot yaw frame."""
        if hasattr(asset.data, "heading_w"):
            heading = asset.data.heading_w
            forward_w = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        elif hasattr(asset.data, "root_quat_w"):
            quat = asset.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            forward_w = torch.stack(
                (
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + z * w),
                ),
                dim=1,
            )
        else:
            forward_w = torch.zeros((self.env.num_envs, 2), device=self.env.device)
            forward_w[:, 0] = 1.0
        forward_w = forward_w / torch.linalg.norm(forward_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        left_w = torch.stack((-forward_w[:, 1], forward_w[:, 0]), dim=1)
        body_x = torch.sum(vectors_w_xy * forward_w.unsqueeze(1), dim=-1)
        body_y = torch.sum(vectors_w_xy * left_w.unsqueeze(1), dim=-1)
        return body_x, body_y

    def _log_reward_warning(self, message: str, *args):
        formatted = message % args if args else message
        logger = getattr(self, "logger", None)
        if logger is None:
            logger = getattr(self.env, "logger", None)
        if logger is not None:
            logger.warning(formatted)
        else:
            print(formatted)

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

    def _reward_forward_stall(
        self,
        command_name: str = "base_velocity",
        cmd_threshold: float = 0.18,
        progress_ratio: float = 0.35,
    ):
        """Penalize very low forward speed when a forward command is active."""
        asset = self._get_robot_asset()
        cmd_vx = self.env.command_manager.get_command(command_name)[:, 0]
        actual_vx = asset.data.root_lin_vel_b[:, 0]
        active_forward = cmd_vx > cmd_threshold
        min_expected_vx = progress_ratio * cmd_vx
        stall = torch.clamp(min_expected_vx - actual_vx, min=0.0)
        return torch.square(stall) * active_forward.float()

    def _reward_yaw_drift(
        self,
        command_name: str = "base_velocity",
        lin_cmd_threshold: float = 0.10,
        ang_cmd_threshold: float = 0.08,
        deadzone: float = 0.08,
    ):
        """Penalize unintended yaw while the command asks for straight climbing."""
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)
        straight_forward_cmd = (cmd[:, 0] > lin_cmd_threshold) & (torch.abs(cmd[:, 2]) < ang_cmd_threshold)
        yaw_excess = torch.clamp(torch.abs(asset.data.root_ang_vel_b[:, 2]) - deadzone, min=0.0)
        return torch.square(yaw_excess) * straight_forward_cmd.float()

    def _reward_stair_ray_alignment(
        self,
        command_name: str = "base_velocity",
        min_forward_speed: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        x1_limit_deg: float = 12.0,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 10,
        min_step_delta: float = 0.025,
    ):
        """Penalize sideways velocity when stair-like height changes are ahead."""
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is None or not hasattr(height_scanner, "data"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        ray_hits = getattr(height_scanner.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.shape[-2] < 256:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        scan = height_scanner.data.pos_w[:, 2:3] - ray_hits[..., 2]
        grid = scan[:, :256].view(self.env.num_envs, 16, 16)
        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(near_x_start), 15))
        x1 = max(x0 + 1, min(int(near_x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 1 or window.shape[2] < 2:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        forward_delta = torch.mean(torch.abs(window[:, :, 1:] - window[:, :, :-1]), dim=(1, 2))
        stair_gate = forward_delta > min_step_delta

        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)
        vel = asset.data.root_lin_vel_b
        forward_gate = (cmd[:, 0] > min_forward_speed) & (torch.abs(cmd[:, 2]) < yaw_cmd_threshold)

        x1_angle = torch.atan2(vel[:, 1], torch.clamp(vel[:, 0], min=1.0e-4))
        x1_limit = math.radians(float(x1_limit_deg))
        x1_excess = torch.clamp(torch.abs(x1_angle) - x1_limit, min=0.0)
        return torch.square(x1_excess) * stair_gate.float() * forward_gate.float()

    def _reward_stair_command_velocity_alignment(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        min_projected_vel: float = 0.03,
        std_orthogonal: float = 0.25,
        target_speed_ratio: float = 0.70,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
    ):
        """Reward stair motion aligned with the sampled body-frame command."""
        stair_gate, _ = self._stair_gate_from_scan(
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
        )
        asset = self._get_robot_asset()
        _, _, command_speed, command_dir = self._command_xy_speed_dir(command_name)
        vel_xy = asset.data.root_lin_vel_b[:, :2]
        projected_vel = torch.sum(vel_xy * command_dir, dim=1)
        orthogonal_vel = vel_xy[:, 0] * (-command_dir[:, 1]) + vel_xy[:, 1] * command_dir[:, 0]
        target_speed = torch.clamp(command_speed * target_speed_ratio, min=min_projected_vel)
        projected_score = torch.clamp(projected_vel / target_speed.clamp_min(1.0e-6), 0.0, 1.0)
        orthogonal_score = torch.exp(-torch.square(orthogonal_vel / max(std_orthogonal, 1.0e-6)))
        command_gate = command_speed > min_command_speed
        progress_gate = projected_vel > min_projected_vel
        return projected_score * orthogonal_score * stair_gate.float() * command_gate.float() * progress_gate.float()

    def _stair_desired_motion_components(
        self,
        command_name: str = "base_velocity",
        direction_source: str = "command_anchor",
        min_command_speed: float = 0.10,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        terrain_fallback_weight: float = 0.35,
        log_interval: int = 200,
    ):
        asset = self._get_robot_asset()
        desired_dir, anchor_heading, command, command_speed, _ = self._desired_direction_w(
            asset,
            command_name=command_name,
            direction_source=direction_source,
            min_command_speed=min_command_speed,
        )
        desired_dir = desired_dir / torch.linalg.norm(desired_dir, dim=1, keepdim=True).clamp_min(1.0e-6)
        left_dir = torch.stack((-desired_dir[:, 1], desired_dir[:, 0]), dim=1)
        vel_w = self._world_xy_velocity(asset)
        projected_vel = torch.sum(vel_w * desired_dir, dim=1)
        orthogonal_vel = torch.sum(vel_w * left_dir, dim=1)
        gate, scan_gate, terrain_gate, _ = self._stair_desired_gate(
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            terrain_fallback_weight=terrain_fallback_weight,
        )
        command_gate = (command_speed > min_command_speed).float()

        yaw = self._robot_yaw_w(asset)
        if anchor_heading is None:
            desired_yaw = torch.atan2(desired_dir[:, 1], desired_dir[:, 0])
            heading_error = self._wrap_to_pi(yaw - desired_yaw)
        else:
            heading_error = self._wrap_to_pi(yaw - anchor_heading.to(yaw.device))

        active_gate = gate * command_gate
        self._set_stair_desired_debug(
            reward=torch.zeros_like(projected_vel),
            gate=active_gate,
            scan_gate=scan_gate,
            terrain_gate=terrain_gate,
            command_gate=command_gate,
            projected_vel=projected_vel,
            orthogonal_vel=orthogonal_vel,
            heading_error=heading_error,
        )
        self._log_stair_desired_activation(
            gate=active_gate,
            scan_gate=scan_gate,
            terrain_gate=terrain_gate,
            command_gate=command_gate,
            projected_vel=projected_vel,
            orthogonal_vel=orthogonal_vel,
            heading_error=heading_error,
            reason="active" if torch.any(active_gate > 0.0) else "gate_or_command_zero",
            log_interval=log_interval,
        )
        return active_gate, command, command_speed, projected_vel, orthogonal_vel, heading_error

    def _reward_stair_desired_velocity_alignment(
        self,
        command_name: str = "base_velocity",
        direction_source: str = "command_anchor",
        min_command_speed: float = 0.10,
        min_projected_vel: float = 0.03,
        target_speed_ratio: float = 0.70,
        std_orthogonal: float = 0.25,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        terrain_fallback_weight: float = 0.35,
        log_interval: int = 200,
    ):
        """Reward stair movement along the desired world-frame direction."""
        gate, _, command_speed, projected_vel, orthogonal_vel, heading_error = self._stair_desired_motion_components(
            command_name=command_name,
            direction_source=direction_source,
            min_command_speed=min_command_speed,
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            terrain_fallback_weight=terrain_fallback_weight,
            log_interval=log_interval,
        )
        target_speed = torch.clamp(command_speed * target_speed_ratio, min=min_projected_vel)
        projected_score = torch.clamp(projected_vel / target_speed.clamp_min(1.0e-6), 0.0, 1.0)
        orthogonal_score = torch.exp(-torch.square(orthogonal_vel / max(std_orthogonal, 1.0e-6)))
        progress_gate = (projected_vel > min_projected_vel).float()
        reward = projected_score * orthogonal_score * gate * progress_gate
        debug = getattr(self.env, "_stair_desired_debug", {})
        debug["stair_desired_reward_mean"] = float(reward.detach().float().mean().item()) if reward.numel() else 0.0
        debug["stair_desired_heading_error_mean"] = float(torch.abs(heading_error).detach().float().mean().item()) if heading_error.numel() else 0.0
        self.env._stair_desired_debug = debug
        return reward

    def _reward_stair_desired_orthogonal_vel(
        self,
        command_name: str = "base_velocity",
        direction_source: str = "command_anchor",
        min_command_speed: float = 0.10,
        deadzone: float = 0.03,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        terrain_fallback_weight: float = 0.35,
        log_interval: int = 200,
    ):
        """Penalize stair-sideways velocity relative to the desired direction."""
        gate, _, _, _, orthogonal_vel, _ = self._stair_desired_motion_components(
            command_name=command_name,
            direction_source=direction_source,
            min_command_speed=min_command_speed,
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            terrain_fallback_weight=terrain_fallback_weight,
            log_interval=log_interval,
        )
        excess = torch.clamp(torch.abs(orthogonal_vel) - deadzone, min=0.0)
        return torch.square(excess) * gate

    def _reward_stair_uncommanded_heading_drift(
        self,
        command_name: str = "base_velocity",
        direction_source: str = "command_anchor",
        min_command_speed: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        deadzone_deg: float = 6.0,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 10,
        terrain_fallback_weight: float = 0.35,
        log_interval: int = 200,
    ):
        """Penalize pre-turning near stairs when yaw command is near zero."""
        gate, command, _, _, _, heading_error = self._stair_desired_motion_components(
            command_name=command_name,
            direction_source=direction_source,
            min_command_speed=min_command_speed,
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            terrain_fallback_weight=terrain_fallback_weight,
            log_interval=log_interval,
        )
        yaw_gate = (torch.abs(command[:, 2]) < yaw_cmd_threshold).float()
        deadzone = math.radians(float(deadzone_deg))
        excess = torch.clamp(torch.abs(heading_error) - deadzone, min=0.0)
        return torch.square(excess) * gate * yaw_gate

    def _reward_stair_heading_alignment(
        self,
        command_name: str = "base_velocity",
        min_forward_cmd: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        heading_limit_deg: float = 18.0,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 10,
        min_step_delta: float = 0.025,
    ):
        """Penalize body-heading drift away from the stair-forward axis."""
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is None or not hasattr(height_scanner, "data"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        ray_hits = getattr(height_scanner.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.shape[-2] < 256:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        scan = height_scanner.data.pos_w[:, 2:3] - ray_hits[..., 2]
        grid = scan[:, :256].view(self.env.num_envs, 16, 16)
        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(near_x_start), 15))
        x1 = max(x0 + 1, min(int(near_x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 1 or window.shape[2] < 2:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        forward_delta = torch.mean(torch.abs(window[:, :, 1:] - window[:, :, :-1]), dim=(1, 2))
        stair_gate = forward_delta > min_step_delta

        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)
        command_gate = (cmd[:, 0] > min_forward_cmd) & (torch.abs(cmd[:, 2]) < yaw_cmd_threshold)

        forward = torch.zeros((self.env.num_envs, 2), device=self.env.device)
        forward[:, 0] = 1.0
        if hasattr(asset.data, "heading_w"):
            heading = asset.data.heading_w
            forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        elif hasattr(asset.data, "root_quat_w"):
            quat = asset.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            forward = torch.stack(
                (
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + z * w),
                ),
                dim=1,
            )

        heading_angle = torch.atan2(forward[:, 1], torch.clamp(forward[:, 0], min=1.0e-4))
        heading_limit = math.radians(float(heading_limit_deg))
        heading_excess = torch.clamp(torch.abs(heading_angle) - heading_limit, min=0.0)
        return torch.square(heading_excess) * stair_gate.float() * command_gate.float()

    def _reward_down_stair_safety(
        self,
        command_name: str = "base_velocity",
        min_forward_cmd: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        max_forward_vel: float = 0.75,
        max_abs_vertical_vel: float = 0.35,
        max_ang_vel_xy: float = 0.75,
        speed_scale: float = 1.0,
        vertical_scale: float = 1.0,
        angular_scale: float = 0.6,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 5,
        far_x_start: int = 7,
        far_x_end: int = 10,
        min_drop: float = 0.025,
        max_drop_for_gate: float = 0.18,
    ):
        """Penalize lunging/impact motions only when the scan ahead goes downward."""
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is None or not hasattr(height_scanner, "data"):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        ray_hits = getattr(height_scanner.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.shape[-2] < 256:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        hits_z = ray_hits[..., 2]
        if hits_z.dim() != 2:
            hits_z = hits_z.view(self.env.num_envs, -1)
        if hits_z.shape[1] < 256:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        grid = hits_z[:, :256].view(self.env.num_envs, 16, 16)
        valid = torch.isfinite(grid)
        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        nx0 = max(0, min(int(near_x_start), 15))
        nx1 = max(nx0 + 1, min(int(near_x_end), 16))
        fx0 = max(0, min(int(far_x_start), 15))
        fx1 = max(fx0 + 1, min(int(far_x_end), 16))

        near_sample = grid[:, y0:y1, nx0:nx1]
        far_sample = grid[:, y0:y1, fx0:fx1]
        near_valid = valid[:, y0:y1, nx0:nx1]
        far_valid = valid[:, y0:y1, fx0:fx1]
        near_count = near_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        far_count = far_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        near_ground = torch.where(near_valid, near_sample, torch.zeros_like(near_sample)).sum(dim=(1, 2)) / near_count
        far_ground = torch.where(far_valid, far_sample, torch.zeros_like(far_sample)).sum(dim=(1, 2)) / far_count

        drop_ahead = torch.clamp(near_ground - far_ground - min_drop, min=0.0)
        drop_gate = torch.clamp(drop_ahead / max(max_drop_for_gate, 1.0e-6), 0.0, 1.0)

        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)
        command_gate = (cmd[:, 0] > min_forward_cmd) & (torch.abs(cmd[:, 2]) < yaw_cmd_threshold)

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b
        forward_excess = torch.clamp(base_lin_vel[:, 0] - max_forward_vel, min=0.0)
        vertical_excess = torch.clamp(torch.abs(base_lin_vel[:, 2]) - max_abs_vertical_vel, min=0.0)
        angular_excess = torch.clamp(torch.linalg.norm(base_ang_vel[:, :2], dim=1) - max_ang_vel_xy, min=0.0)
        safety_penalty = (
            speed_scale * torch.square(forward_excess)
            + vertical_scale * torch.square(vertical_excess)
            + angular_scale * torch.square(angular_excess)
        )
        return safety_penalty * drop_gate * command_gate.float()

    def _reward_stair_height_transition(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.15,
        min_stair_score: float = 0.03,
        max_wall_score: float = 0.08,
        min_tracking_ratio: float = 0.35,
        std: float = 0.38,
        max_transition: float = 0.18,
        max_vertical_vel: float = 0.60,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_end: int = 4,
    ):
        """Reward stable local ground-height transitions while tracking commands."""
        ground_z = self._estimate_ground_z_from_scan(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_end=near_x_end,
        )
        if ground_z is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()
        _, command_xy, command_speed, _ = self._command_xy_speed_dir(command_name)
        tracking_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - command_xy), dim=1)
        tracking_ratio = torch.exp(-tracking_error / max(std * std, 1.0e-6))
        stair_gate, _ = self._stair_gate_from_scan(
            min_stair_score=min_stair_score,
            max_wall_score=max_wall_score,
        )

        if (
            not hasattr(self.env, "_stair_prev_ground_z")
            or self.env._stair_prev_ground_z.shape != ground_z.shape
        ):
            self.env._stair_prev_ground_z = ground_z.detach().clone()
            return torch.zeros_like(ground_z)

        reset_mask = self._reset_mask()
        prev_ground_z = torch.where(reset_mask, ground_z, self.env._stair_prev_ground_z.to(ground_z.device))
        transition = torch.clamp(torch.abs(ground_z - prev_ground_z), min=0.0, max=max_transition)
        gate = (
            stair_gate
            * (command_speed > min_command_speed).float()
            * (tracking_ratio > min_tracking_ratio).float()
            * (torch.abs(asset.data.root_lin_vel_b[:, 2]) < max_vertical_vel).float()
            * (~reset_mask).float()
        )
        self.env._stair_prev_ground_z = ground_z.detach().clone()
        return (transition / max(max_transition, 1.0e-6)) * gate

    # Deprecated: inv_stair_climb_action was removed from active training.
    # The terrain-label gate and worker-side reward invocation could not be
    # verified reliably in the Kaiwu worker setup, so this shaping path is
    # intentionally unavailable for this fine-tune round.

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

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame (the larger the better).
        前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）。

        This is an example reward that demonstrates how to read the robot state and
        build a dense signal.
        示例性 reward，展示如何读取机器人状态并构造 dense signal。
        """
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]
