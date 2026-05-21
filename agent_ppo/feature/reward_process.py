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

from agent_ppo.feature.command_mix import get_mixed_command
from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    def _get_velocity_command(self, command_name: str = "base_velocity"):
        return get_mixed_command(self.env, command_name=command_name, site="reward")

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
        command = self._get_velocity_command(command_name)
        effective_command_xy = command[:, :2].clone()
        if min_tracking_vx > 0.0:
            effective_command_xy[:, 0] = torch.clamp(effective_command_xy[:, 0], min=min_tracking_vx)
        lin_vel_error = torch.sum(torch.square(effective_command_xy - asset.data.root_lin_vel_b[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / max(std * std, 1.0e-6))

    def _reward_track_ang_vel_z(self, std: float = 0.25, command_name: str = "base_velocity"):
        """Track yaw-rate against the same mixed command that the policy sees."""
        command = self._get_velocity_command(command_name)
        ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
        return torch.exp(-ang_vel_error / max(std * std, 1.0e-6))

    def _reward_command_direction_progress(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        target_speed_ratio: float = 0.80,
        min_target_speed: float = 0.08,
    ):
        """Reward forward progress along the sampled XY velocity command.

        This is intentionally different from exact velocity tracking.  It gives
        dense credit for moving in the commanded direction, which discourages
        stair hesitation/timeouts without binding the robot to a world axis.
        """
        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)

        actual_xy = asset.data.root_lin_vel_b[:, :2]
        projected_speed = torch.sum(actual_xy * command_dir, dim=1)
        target_speed = torch.clamp(command_speed * float(target_speed_ratio), min=float(min_target_speed))
        progress = torch.clamp(projected_speed / target_speed.clamp_min(1.0e-6), min=0.0, max=1.0)
        active = command_speed > float(min_command_speed)
        value = progress * active.float()

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["cmd_progress_active_ratio"] = self._tensor_ratio(active)
        debug["cmd_progress_vel_mean"] = self._tensor_mean(projected_speed)
        debug["cmd_progress_reward_mean"] = self._tensor_mean(value)
        self.env._stair_gate_debug = debug
        return value

    def _reward_command_direction_deviation(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        min_actual_speed: float = 0.10,
        angle_limit_deg: float = 0.0,
        max_angle_deg: float = 75.0,
        log_interval: int = 100,
    ):
        """Penalize moving at a large angle away from the XY velocity command.

        The penalty grows quadratically from ``angle_limit_deg`` to
        ``max_angle_deg``.  It only applies while actually moving, so standing
        still is handled by missing progress reward rather than by a noisy
        undefined direction penalty.
        """
        asset = self._get_robot_asset()
        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)

        actual_xy = asset.data.root_lin_vel_b[:, :2]
        actual_speed = torch.linalg.norm(actual_xy, dim=1)
        actual_dir = actual_xy / actual_speed.unsqueeze(1).clamp_min(1.0e-6)
        cos_angle = torch.clamp(torch.sum(actual_dir * command_dir, dim=1), min=-1.0, max=1.0)
        angle_deg = torch.rad2deg(torch.acos(cos_angle))

        active = (command_speed > float(min_command_speed)) & (actual_speed > float(min_actual_speed))
        span = max(float(max_angle_deg) - float(angle_limit_deg), 1.0e-6)
        deviation = torch.square(torch.clamp((angle_deg - float(angle_limit_deg)) / span, min=0.0, max=1.0))
        value = deviation * active.float()

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["cmd_dir_dev_active_ratio"] = self._tensor_ratio(active & (angle_deg > float(angle_limit_deg)))
        debug["cmd_dir_dev_angle_deg_mean"] = self._tensor_mean(angle_deg)
        debug["cmd_dir_dev_reward_mean"] = self._tensor_mean(value)
        self.env._stair_gate_debug = debug

        if not hasattr(self.env, "_cmd_dir_dev_log_count"):
            self.env._cmd_dir_dev_log_count = 0
        self.env._cmd_dir_dev_log_count += 1
        count = self.env._cmd_dir_dev_log_count
        interval = max(int(log_interval), 1)
        if count == 1 or count % interval == 0:
            self._log_reward_warning(
                "[CommandDirDev] call=%d active=%.4f angle_deg=%.3f reward_mean=%.6f",
                count,
                debug["cmd_dir_dev_active_ratio"],
                debug["cmd_dir_dev_angle_deg_mean"],
                debug["cmd_dir_dev_reward_mean"],
            )
        return value

    def _reward_command_path_progress(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        target_step_progress: float = 0.006,
        segment_progress_cap: float = 0.35,
        command_change_threshold: float = 1.0e-4,
        log_interval: int = 100,
    ):
        """Reward short-horizon displacement along the commanded path.

        For near-zero yaw commands, the desired direction is anchored at reset or
        command resampling time.  For full commands with yaw, the current command
        direction is used so legitimate commanded turns are not penalized.

        The reward is capped by command-segment progress.  It acts as an
        anti-stall signal near the start of a command segment, not as an
        always-on "rush forward" reward that can destabilize down-stairs.
        """
        asset = self._get_robot_asset()
        root_xy = asset.data.root_pos_w[:, :2]
        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        active = command_speed > float(min_command_speed)

        current_dir_w = self._body_xy_to_world_xy(asset, command_xy)
        current_dir_w = current_dir_w / torch.linalg.norm(current_dir_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        fallback_dir_w = self._body_xy_to_world_xy(
            asset,
            torch.tensor([[1.0, 0.0]], device=root_xy.device, dtype=root_xy.dtype).repeat(self.env.num_envs, 1),
        )
        current_dir_w = torch.where(active.unsqueeze(1), current_dir_w, fallback_dir_w)

        needs_init = (
            not hasattr(self.env, "_cmd_path_anchor_pos_w")
            or self.env._cmd_path_anchor_pos_w.shape != root_xy.shape
            or not hasattr(self.env, "_cmd_path_prev_pos_w")
            or self.env._cmd_path_prev_pos_w.shape != root_xy.shape
            or not hasattr(self.env, "_cmd_path_anchor_dir_w")
            or self.env._cmd_path_anchor_dir_w.shape != current_dir_w.shape
            or not hasattr(self.env, "_cmd_path_anchor_cmd")
            or self.env._cmd_path_anchor_cmd.shape != command.shape
        )
        if needs_init:
            self.env._cmd_path_anchor_pos_w = root_xy.detach().clone()
            self.env._cmd_path_prev_pos_w = root_xy.detach().clone()
            self.env._cmd_path_anchor_dir_w = current_dir_w.detach().clone()
            self.env._cmd_path_anchor_cmd = command.detach().clone()

        prev_cmd = self.env._cmd_path_anchor_cmd.to(command.device)
        command_changed = torch.linalg.norm(command - prev_cmd, dim=1) > float(command_change_threshold)
        refresh = self._reset_mask().to(command.device) | command_changed

        anchor_pos = torch.where(
            refresh.unsqueeze(1),
            root_xy.detach(),
            self.env._cmd_path_anchor_pos_w.to(root_xy.device),
        )
        prev_pos = torch.where(
            refresh.unsqueeze(1),
            root_xy.detach(),
            self.env._cmd_path_prev_pos_w.to(root_xy.device),
        )
        anchor_dir = torch.where(
            refresh.unsqueeze(1),
            current_dir_w.detach(),
            self.env._cmd_path_anchor_dir_w.to(current_dir_w.device),
        )
        anchor_cmd = torch.where(refresh.unsqueeze(1), command.detach(), prev_cmd)

        no_yaw = torch.abs(command[:, 2]) < float(yaw_cmd_threshold)
        desired_dir = torch.where(no_yaw.unsqueeze(1), anchor_dir, current_dir_w)
        step_delta = root_xy - prev_pos
        delta_progress = torch.sum(step_delta * desired_dir, dim=1)
        projected_dist = torch.sum((root_xy - anchor_pos) * desired_dir, dim=1)
        raw_value = torch.clamp(
            delta_progress / max(float(target_step_progress), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        segment_factor = torch.clamp(
            (float(segment_progress_cap) - torch.clamp(projected_dist, min=0.0))
            / max(float(segment_progress_cap), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        value = raw_value * segment_factor * active.float()

        self.env._cmd_path_anchor_pos_w = anchor_pos.detach().clone()
        self.env._cmd_path_prev_pos_w = root_xy.detach().clone()
        self.env._cmd_path_anchor_dir_w = anchor_dir.detach().clone()
        self.env._cmd_path_anchor_cmd = anchor_cmd.detach().clone()

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["cmd_path_progress_active_ratio"] = self._tensor_ratio(active)
        debug["cmd_path_full_active_ratio"] = self._tensor_ratio(active & (~no_yaw))
        debug["cmd_path_delta_progress_mean"] = self._tensor_mean(delta_progress)
        debug["cmd_path_projected_dist_mean"] = self._tensor_mean(projected_dist)
        debug["cmd_path_segment_factor_mean"] = self._tensor_mean(segment_factor)
        debug["cmd_path_reward_mean"] = self._tensor_mean(value)
        self.env._stair_gate_debug = debug

        if not hasattr(self.env, "_cmd_path_progress_log_count"):
            self.env._cmd_path_progress_log_count = 0
        self.env._cmd_path_progress_log_count += 1
        count = self.env._cmd_path_progress_log_count
        interval = max(int(log_interval), 1)
        if count == 1 or count % interval == 0:
            self._log_reward_warning(
                "[CommandPathProgress] call=%d active=%.4f full=%.4f delta=%.5f dist=%.3f cap_factor=%.3f reward_mean=%.6f",
                count,
                debug["cmd_path_progress_active_ratio"],
                debug["cmd_path_full_active_ratio"],
                debug["cmd_path_delta_progress_mean"],
                debug["cmd_path_projected_dist_mean"],
                debug["cmd_path_segment_factor_mean"],
                debug["cmd_path_reward_mean"],
            )
        return value

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
        is_moving = torch.norm(self._get_velocity_command(command_name)[:, :2], dim=1) > 0.1
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
        reward_scale: float = 1.0,
        max_reward: float = 1.0,
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
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        ground_z = self._ground_z_window_from_scan(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            x_start=0,
            x_end=near_x_end,
        )
        if ground_z is None:
            ground_z = asset.data.root_pos_w[:, 2] - 0.35
        foot_height = foot_z - ground_z.unsqueeze(1)
        command = self._get_velocity_command(command_name)
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
        value = torch.sum(clearance_reward * swing.float(), dim=1) * is_moving.float() / max(len(asset_cfg.body_ids), 1)
        value = torch.clamp(value * float(reward_scale), min=0.0, max=float(max_reward))

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["feet_clearance_reward_mean"] = self._tensor_mean(value)
        debug["feet_clearance_active_ratio"] = self._tensor_ratio(swing.float() * is_moving.float().unsqueeze(1))
        debug["feet_clearance_height_mean"] = self._tensor_mean(foot_height)
        debug["feet_clearance_target_mean"] = self._tensor_mean(dynamic_target_height)
        self.env._stair_gate_debug = debug
        return value

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

        command = self._get_velocity_command(command_name)
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
        base_clearance: float = 0.06,
        min_clearance: float = 0.03,
        max_clearance: float = 0.12,
        step_height_scale: float = 0.75,
        max_extra_height: float = 0.12,
        speed_height_scale: float = 0.01,
        std: float = 0.05,
        max_air_time: float = 0.45,
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
        reward_scale: float = 1.0,
        max_reward: float = 1.0,
        log_interval: int = 50,
    ):
        """Reward swing-foot clearance in a bounded height band over an up-step.

        This is deliberately scan-gated rather than terrain-label-gated. It does
        not encode a world/stair heading, so it should not create the previous
        diagonal-path dependency by itself.  The target is expressed relative to
        the scan-estimated upper step surface: enough clearance to avoid the
        edge, but no extra reward for throwing a foot far above the step.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        gate = self._height_scan_reward_stair_gate(
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
        current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        air_time_ok = current_air_time <= float(max_air_time)
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        command = self._get_velocity_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        up_gate = gate["up_step"].float()
        upper_surface_z = torch.maximum(gate["near_z"], gate["front_z"])
        # The target is measured above the upper surface itself, so the local
        # step height should not be added again. Keep legacy params accepted by
        # the function signature for TOML compatibility.
        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        target_clearance = (
            float(base_clearance)
            + speed_extra
        )
        target_clearance = torch.clamp(target_clearance, min=float(min_clearance), max=float(max_clearance))
        clearance = foot_z - upper_surface_z.unsqueeze(1)
        height_error = (clearance - target_clearance.unsqueeze(1)) / max(std, 1.0e-6)
        reward = torch.exp(-torch.square(height_error))
        clearance_valid = clearance >= float(min_clearance)
        active = (command_speed > min_command_speed).float() * up_gate
        value = torch.sum(
            reward * swing.float() * air_time_ok.float() * clearance_valid.float(),
            dim=1,
        ) / max(len(asset_cfg.body_ids), 1)
        value = value * active
        value = torch.clamp(value * float(reward_scale), min=0.0, max=float(max_reward))

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["height_scan_feet_clearance_reward_mean"] = float(value.detach().float().mean().item()) if value.numel() else 0.0
        debug["height_scan_feet_clearance_active_ratio"] = float((active.detach().float() > 0.0).float().mean().item()) if active.numel() else 0.0
        debug["height_scan_feet_clearance_clearance_mean"] = self._tensor_mean(clearance)
        debug["height_scan_feet_clearance_target_mean"] = self._tensor_mean(target_clearance)
        self.env._stair_gate_debug = debug
        return value

    def _reward_stair_forward_foot_placement(
        self,
        command_name: str = "base_velocity",
        target_forward: float = 0.12,
        max_forward: float = 0.30,
        forward_std: float = 0.08,
        overshoot_std: float = 0.12,
        height_std: float = 0.07,
        touchdown_speed_std: float = 0.35,
        foot_ground_offset: float = 0.02,
        min_air_time: float = 0.04,
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
        reward_scale: float = 1.0,
        max_reward: float = 1.0,
        log_interval: int = 50,
    ):
        """Reward touchdown that places swing feet forward onto an up-step.

        This complements ``height_scan_feet_clearance``.  Clearance rewards
        lifting to a suitable height during swing; this term is a sparse
        touchdown reward that prefers landing forward along the sampled velocity
        command and near the scan-estimated upper step surface.  It is positive
        only, so missed placements do not add an extra destabilizing penalty.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        gate = self._height_scan_reward_stair_gate(
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

        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        fallback = torch.zeros_like(command_dir)
        fallback[:, 0] = 1.0
        command_dir = torch.where((command_speed > 1.0e-6).unsqueeze(1), command_dir, fallback)

        foot_rel_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - asset.data.root_pos_w[:, :2].unsqueeze(1)
        foot_forward, foot_lateral = self._project_world_xy_to_body_xy(asset, foot_rel_xy)
        foot_command_forward = (
            foot_forward * command_dir[:, 0].unsqueeze(1)
            + foot_lateral * command_dir[:, 1].unsqueeze(1)
        )

        shortfall = torch.clamp(target_forward - foot_command_forward, min=0.0)
        overshoot = torch.clamp(foot_command_forward - max_forward, min=0.0)
        forward_score = torch.exp(-torch.square(shortfall / max(forward_std, 1.0e-6)))
        forward_score = forward_score * torch.exp(-torch.square(overshoot / max(overshoot_std, 1.0e-6)))

        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        target_z = torch.maximum(gate["near_z"], gate["front_z"]).unsqueeze(1)
        height_error = foot_z - (target_z + foot_ground_offset)
        height_score = torch.exp(-torch.square(height_error / max(height_std, 1.0e-6)))

        foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        downward_speed = torch.clamp(-foot_vel_z, min=0.0)
        touchdown_speed_score = torch.exp(
            -torch.square(downward_speed / max(float(touchdown_speed_std), 1.0e-6))
        )

        contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
        contact_now = contact_forces > 1.0
        first_contact = self._foot_first_contact(
            contact_sensor,
            sensor_cfg,
            contact_now,
            min_air_time=min_air_time,
            cache_name="stair_forward_foot_placement",
        )

        active = (command_speed > min_command_speed).float() * gate["up_step"].float()
        placement = forward_score * height_score * touchdown_speed_score * first_contact.float()
        value = torch.sum(placement, dim=1) / max(len(asset_cfg.body_ids), 1)
        value = value * active
        value = torch.clamp(value * float(reward_scale), min=0.0, max=float(max_reward))

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["stair_forward_foot_placement_reward_mean"] = float(value.detach().float().mean().item()) if value.numel() else 0.0
        debug["stair_forward_foot_placement_active_ratio"] = float((active.detach().float() > 0.0).float().mean().item()) if active.numel() else 0.0
        debug["stair_forward_foot_placement_first_contact_ratio"] = (
            float(first_contact.detach().float().mean().item()) if first_contact.numel() else 0.0
        )
        debug["stair_forward_foot_placement_forward_mean"] = (
            float(foot_command_forward.detach().float().mean().item()) if foot_command_forward.numel() else 0.0
        )
        debug["stair_forward_foot_placement_height_error_abs_mean"] = (
            float(torch.abs(height_error).detach().float().mean().item()) if height_error.numel() else 0.0
        )
        debug["stair_forward_foot_placement_down_speed_mean"] = self._tensor_mean(downward_speed)
        self.env._stair_gate_debug = debug
        return value

    def _reward_stair_over_clearance_penalty(
        self,
        max_clearance: float = 0.14,
        std: float = 0.07,
        min_air_time: float = 0.08,
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
        max_penalty: float = 1.0,
        log_interval: int = 50,
    ):
        """Penalize sustained swing-foot height far above the upper stair surface."""
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        gate = self._height_scan_reward_stair_gate(
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
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        sustained_swing = air_time >= float(min_air_time)
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        upper_surface_z = torch.maximum(gate["near_z"], gate["front_z"]).unsqueeze(1)
        clearance = foot_z - upper_surface_z
        over = torch.clamp(clearance - float(max_clearance), min=0.0)
        per_foot = torch.square(over / max(float(std), 1.0e-6))
        per_foot = torch.clamp(per_foot, min=0.0, max=float(max_penalty))
        active = gate["stair_gate"].float().unsqueeze(1) * swing.float() * sustained_swing.float()
        value = torch.sum(per_foot * active, dim=1) / max(len(asset_cfg.body_ids), 1)

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["stair_over_clearance_penalty_mean"] = self._tensor_mean(value)
        debug["stair_over_clearance_active_ratio"] = self._tensor_ratio(active)
        debug["stair_over_clearance_clearance_mean"] = self._tensor_mean(clearance)
        self.env._stair_gate_debug = debug
        return value

    def _reward_stair_base_clearance_penalty(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        min_clearance: float = 0.34,
        std: float = 0.08,
        max_penalty: float = 1.5,
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
        """Penalize a low base over an up-step without enforcing flat height globally."""
        asset = self._get_robot_asset()
        command = self._get_velocity_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        gate = self._height_scan_reward_stair_gate(
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

        upper_surface_z = torch.maximum(gate["near_z"], gate["front_z"])
        base_clearance = asset.data.root_pos_w[:, 2] - upper_surface_z
        deficit = torch.clamp(float(min_clearance) - base_clearance, min=0.0)
        value = torch.square(deficit / max(float(std), 1.0e-6))
        value = torch.clamp(value, min=0.0, max=float(max_penalty))
        active = (command_speed > float(min_command_speed)) & gate["up_step"]
        value = value * active.float()

        active_float = active.float()
        active_count = active_float.sum().clamp_min(1.0)
        active_clearance = (base_clearance * active_float).sum() / active_count
        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["stair_base_clearance_penalty_mean"] = self._tensor_mean(value)
        debug["stair_base_clearance_active_ratio"] = self._tensor_ratio(active)
        debug["stair_base_clearance_mean"] = float(active_clearance.detach().float().item())
        debug["stair_base_clearance_deficit_mean"] = self._tensor_mean(deficit * active_float)
        self.env._stair_gate_debug = debug

        if not hasattr(self.env, "_stair_base_clearance_log_count"):
            self.env._stair_base_clearance_log_count = 0
        self.env._stair_base_clearance_log_count += 1
        count = self.env._stair_base_clearance_log_count
        interval = max(int(log_interval), 1)
        if count == 1 or count % interval == 0:
            self._log_reward_warning(
                "[StairBaseClearance] call=%d active=%.4f clearance=%.4f deficit=%.4f penalty=%.6f",
                count,
                debug["stair_base_clearance_active_ratio"],
                debug["stair_base_clearance_mean"],
                debug["stair_base_clearance_deficit_mean"],
                debug["stair_base_clearance_penalty_mean"],
            )
        return value

    def _reward_stair_edge_normal_alignment(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        min_edge_strength: float = 0.04,
        max_penalty: float = 1.0,
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
        """Penalize moving along a stair edge instead of across its local normal."""
        grid = self._height_scan_ground_proxy_grid()
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        if grid is None:
            return zeros

        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        normal, edge_strength, lateral_ratio = self._height_scan_edge_normal_body(
            grid,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            x_start=near_x_start,
            x_end=front_x_end,
        )
        forward_dir = torch.zeros_like(command_dir)
        forward_dir[:, 0] = 1.0
        command_dir = torch.where((command_speed > 1.0e-6).unsqueeze(1), command_dir, forward_dir)

        gate = self._height_scan_reward_stair_gate(
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

        align_cos = torch.abs(torch.sum(normal * command_dir, dim=1)).clamp(0.0, 1.0)
        value = torch.clamp(1.0 - align_cos, min=0.0, max=float(max_penalty))
        active = (
            (command_speed > float(min_command_speed))
            & gate["stair_gate"]
            & (edge_strength > float(min_edge_strength))
        )
        value = value * active.float()

        active_float = active.float()
        active_count = active_float.sum().clamp_min(1.0)
        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["stair_edge_normal_alignment_penalty_mean"] = self._tensor_mean(value)
        debug["stair_edge_align_active_ratio"] = self._tensor_ratio(active)
        debug["stair_edge_align_cos_mean"] = float(((align_cos * active_float).sum() / active_count).detach().float().item())
        debug["stair_edge_lateral_ratio_mean"] = float(((lateral_ratio * active_float).sum() / active_count).detach().float().item())
        debug["stair_edge_strength_mean"] = float(((edge_strength * active_float).sum() / active_count).detach().float().item())
        self.env._stair_gate_debug = debug

        if not hasattr(self.env, "_stair_edge_align_log_count"):
            self.env._stair_edge_align_log_count = 0
        self.env._stair_edge_align_log_count += 1
        count = self.env._stair_edge_align_log_count
        interval = max(int(log_interval), 1)
        if count == 1 or count % interval == 0:
            self._log_reward_warning(
                "[StairEdgeNormalAlign] call=%d active=%.4f cos=%.4f lateral=%.4f strength=%.4f penalty=%.6f",
                count,
                debug["stair_edge_align_active_ratio"],
                debug["stair_edge_align_cos_mean"],
                debug["stair_edge_lateral_ratio_mean"],
                debug["stair_edge_strength_mean"],
                debug["stair_edge_normal_alignment_penalty_mean"],
            )
        return value

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
        cmd = self._get_velocity_command("base_velocity")
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
        cmd = self._get_velocity_command(command_name)

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
        command = self._get_velocity_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

        commanded_to_move = command_speed > cmd_threshold
        stillness = torch.clamp(
            (still_speed_threshold - body_speed) / max(still_speed_threshold, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        return commanded_to_move.float() * stillness

    def _reward_commanded_stall_penalty(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.15,
        min_projected_speed: float = 0.10,
        progress_ratio: float = 0.25,
        stall_time_s: float = 0.80,
        step_dt: float = 0.02,
        grace_after_contact_s: float = 0.20,
        min_contact_air_time: float = 0.04,
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
        log_interval: int = 100,
    ):
        """Penalize sustained failure to progress along the XY command.

        This is not a dense forward reward.  It only activates after sustained
        low projected velocity and is disabled during down-step probing and a
        short window after touchdown, so it does not force a stair descent rush.
        """
        asset = self._get_robot_asset()
        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        actual_xy = asset.data.root_lin_vel_b[:, :2]
        projected_speed = torch.sum(actual_xy * command_dir, dim=1)

        gate = self._height_scan_reward_stair_gate(
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

        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
        contact_now = contact_forces > 1.0
        first_contact = self._foot_first_contact(
            contact_sensor,
            sensor_cfg,
            contact_now,
            min_air_time=min_contact_air_time,
            cache_name="commanded_stall",
        )
        any_first_contact = torch.any(first_contact, dim=1)

        dt = max(float(step_dt), 1.0e-6)
        grace_steps = max(int(math.ceil(float(grace_after_contact_s) / dt)), 0)
        if (
            not hasattr(self.env, "_commanded_stall_time")
            or self.env._commanded_stall_time.shape != command_speed.shape
        ):
            self.env._commanded_stall_time = torch.zeros_like(command_speed)
        if (
            not hasattr(self.env, "_commanded_stall_contact_grace")
            or self.env._commanded_stall_contact_grace.shape != command_speed.shape
        ):
            self.env._commanded_stall_contact_grace = torch.zeros_like(command_speed)

        reset_mask = self._reset_mask().to(command_speed.device)
        grace = self.env._commanded_stall_contact_grace.to(command_speed.device)
        grace = torch.where(any_first_contact, torch.full_like(grace, float(grace_steps)), torch.clamp(grace - 1.0, min=0.0))
        grace = torch.where(reset_mask, torch.zeros_like(grace), grace)
        contact_grace_active = grace > 0.0

        threshold = torch.maximum(
            torch.full_like(command_speed, float(min_projected_speed)),
            float(progress_ratio) * command_speed,
        )
        below_threshold = projected_speed < threshold
        allowed_pause = gate["down_step"].to(command_speed.device) | contact_grace_active
        commanded = command_speed > float(min_command_speed)
        stalled = commanded & below_threshold & (~allowed_pause)

        stall_time = self.env._commanded_stall_time.to(command_speed.device)
        stall_time = torch.where(stalled, stall_time + dt, torch.zeros_like(stall_time))
        stall_time = torch.where(reset_mask, torch.zeros_like(stall_time), stall_time)
        self.env._commanded_stall_time = stall_time.detach().clone()
        self.env._commanded_stall_contact_grace = grace.detach().clone()

        severity = torch.clamp((threshold - projected_speed) / threshold.clamp_min(1.0e-6), min=0.0, max=1.0)
        time_factor = torch.clamp((stall_time - float(stall_time_s)) / max(float(stall_time_s), 1.0e-6), min=0.0, max=1.0)
        value = torch.square(severity) * torch.square(time_factor) * commanded.float()

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["commanded_stall_active_ratio"] = self._tensor_ratio(stalled)
        debug["commanded_stall_penalty_mean"] = self._tensor_mean(value)
        debug["commanded_stall_projected_speed_mean"] = self._tensor_mean(projected_speed)
        debug["commanded_stall_time_mean"] = self._tensor_mean(stall_time)
        self.env._stair_gate_debug = debug
        return value

    def _reward_uncommanded_yaw_rate(
        self,
        command_name: str = "base_velocity",
        yaw_cmd_threshold: float = 0.08,
        min_command_speed: float = 0.10,
        base_scale: float = 0.25,
        stair_scale: float = 1.0,
        log_interval: int = 100,
    ):
        """Penalize yawing while the command asks for translation, not rotation.

        This targets the observed "turn on the stair before climbing" behavior
        without binding the policy to a world/stair axis.  The stair multiplier
        uses the same height-scan structure gate as the stair foot rewards.
        """
        asset = self._get_robot_asset()
        command = self._get_velocity_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        no_yaw_command = torch.abs(command[:, 2]) < yaw_cmd_threshold
        moving_command = command_speed > min_command_speed

        gate = self._height_scan_reward_stair_gate(log_interval=log_interval)
        stair_gate = gate["stair_gate"].float()
        active = moving_command.float() * no_yaw_command.float()
        scale = float(base_scale) + float(stair_scale) * stair_gate
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        value = torch.square(yaw_rate) * active * scale

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["uncommanded_yaw_active_ratio"] = self._tensor_ratio(active)
        debug["uncommanded_yaw_stair_ratio"] = self._tensor_ratio(active * stair_gate)
        debug["uncommanded_yaw_abs_mean"] = self._tensor_mean(yaw_rate)
        debug["uncommanded_yaw_reward_mean"] = self._tensor_mean(value)
        self.env._stair_gate_debug = debug
        return value

    def _reward_uncommanded_heading_drift(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        yaw_cmd_threshold: float = 0.08,
        deadzone_deg: float = 6.0,
        max_error_deg: float = 45.0,
        command_change_threshold: float = 1.0e-4,
        log_interval: int = 100,
    ):
        """Penalize heading drift when the command asks for translation but no yaw.

        The anchor heading is refreshed on env reset or when the mixed velocity
        command changes.  This keeps the penalty command-relative instead of
        binding the policy to a world axis, stair axis, or terrain label.
        """
        asset = self._get_robot_asset()
        yaw = self._robot_yaw_w(asset)
        command = self._get_velocity_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        needs_init = (
            not hasattr(self.env, "_uncommanded_heading_anchor_yaw")
            or not hasattr(self.env, "_uncommanded_heading_anchor_cmd")
            or self.env._uncommanded_heading_anchor_yaw.shape != yaw.shape
            or self.env._uncommanded_heading_anchor_cmd.shape != command.shape
        )
        if needs_init:
            self.env._uncommanded_heading_anchor_yaw = yaw.detach().clone()
            self.env._uncommanded_heading_anchor_cmd = command.detach().clone()
        else:
            prev_cmd = self.env._uncommanded_heading_anchor_cmd.to(command.device)
            command_changed = torch.linalg.norm(command - prev_cmd, dim=1) > command_change_threshold
            refresh = self._reset_mask().to(command.device) | command_changed
            anchor_yaw = torch.where(
                refresh,
                yaw.detach(),
                self.env._uncommanded_heading_anchor_yaw.to(command.device),
            )
            anchor_cmd = torch.where(refresh.unsqueeze(1), command.detach(), prev_cmd)
            self.env._uncommanded_heading_anchor_yaw = anchor_yaw.detach().clone()
            self.env._uncommanded_heading_anchor_cmd = anchor_cmd.detach().clone()

        anchor_yaw = self.env._uncommanded_heading_anchor_yaw.to(command.device)
        heading_error = self._wrap_to_pi(yaw - anchor_yaw)
        abs_error = torch.abs(heading_error)
        deadzone = math.radians(float(deadzone_deg))
        max_error = max(math.radians(float(max_error_deg)), deadzone + 1.0e-6)
        excess = torch.clamp(abs_error - deadzone, min=0.0, max=max_error - deadzone)
        normalized = excess / max(max_error - deadzone, 1.0e-6)

        active = (command_speed > min_command_speed) & (torch.abs(command[:, 2]) < yaw_cmd_threshold)
        value = torch.square(normalized) * active.float()

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["heading_drift_active_ratio"] = self._tensor_ratio(active.float())
        debug["heading_drift_abs_deg_mean"] = self._tensor_mean(torch.rad2deg(abs_error))
        debug["heading_drift_reward_mean"] = self._tensor_mean(value)
        self.env._stair_gate_debug = debug

        if not hasattr(self.env, "_heading_drift_log_count"):
            self.env._heading_drift_log_count = 0
        self.env._heading_drift_log_count += 1
        count = self.env._heading_drift_log_count
        interval = max(int(log_interval), 1)
        if count == 1 or count % interval == 0:
            self._log_reward_warning(
                "[HeadingDrift] call=%d active=%.4f mean_abs_deg=%.3f reward_mean=%.6f",
                count,
                debug["heading_drift_active_ratio"],
                debug["heading_drift_abs_deg_mean"],
                debug["heading_drift_reward_mean"],
            )
        return value

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

    def _height_scan_ground_proxy_grid(self):
        """Return observation-compatible local ground-height proxy.

        The height scanner stores distance-to-ground.  The rollout probe that
        was validated in monitor data uses ``-height_scan`` as a local
        ground-height proxy, because only relative deltas matter for stair vs
        slope detection.  Reward-side stair gates must use the same semantics.
        """
        grid = self._height_scan_grid()
        if grid is None:
            return None
        return torch.nan_to_num(-grid, nan=0.0, posinf=0.0, neginf=0.0)

    def _height_scan_reward_stair_gate(
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
        direction_margin: float = 0.02,
        log_interval: int = 100,
    ):
        """Height-scan gate matching the validated rollout monitor probes.

        This replaces the old front-window mean-height gate.  It detects
        discrete stair edges by local edge concentration, rejects smooth slopes,
        and uses dominant up/down evidence so a reward can be tied to step-up
        actions without using terrain labels or world-axis headings.
        """
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        grid = self._height_scan_ground_proxy_grid()
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
        if grid is None or near_z is None or front_z is None:
            self._set_stair_gate_debug(
                "height_scan",
                available=zeros,
                up_step=zeros,
                down_step=zeros,
                wall=zeros,
                flat=torch.ones_like(zeros),
                step_delta=zeros,
                step_score=zeros,
                wall_score=zeros,
            )
            self._log_height_scan_gate_status(
                available=zeros,
                up_step=zeros,
                down_step=zeros,
                wall=zeros,
                flat=torch.ones_like(zeros),
                step_delta=zeros,
                step_score=zeros,
                wall_score=zeros,
                reason="height_scan_unavailable",
                log_interval=log_interval,
            )
            return {
                "available": zeros,
                "stair_gate": zeros.bool(),
                "up_step": zeros.bool(),
                "down_step": zeros.bool(),
                "wall": zeros.bool(),
                "flat": torch.ones_like(zeros).bool(),
                "step_delta": zeros,
                "step_height": zeros,
                "step_score": zeros,
                "wall_score": zeros,
                "near_z": zeros,
                "front_z": zeros,
            }

        thresholds = (
            min(float(up_min_height), float(down_min_height)),
            max(float(up_max_height), float(down_max_height)),
            float(wall_min_height),
            float(min_step_score),
            float(max_wall_score),
        )
        method_outputs = [
            self._height_scan_probe_method(
                method_grid,
                mode=mode,
                thresholds=thresholds,
                body_y_start=body_y_start,
                body_y_end=body_y_end,
                near_x_start=near_x_start,
                near_x_end=near_x_end,
                front_x_start=front_x_start,
                front_x_end=front_x_end,
            )
            for method_grid, mode in (
                (grid, "mean"),
                (grid.transpose(1, 2), "mean"),
                (grid, "edge"),
                (grid.transpose(1, 2), "edge"),
            )
        ]
        probe = {
            "up": torch.stack([out["up"].float() for out in method_outputs], dim=1).amax(dim=1).bool(),
            "down": torch.stack([out["down"].float() for out in method_outputs], dim=1).amax(dim=1).bool(),
            "wall": torch.stack([out["wall"].float() for out in method_outputs], dim=1).amax(dim=1).bool(),
            "step_delta": torch.stack([out["step_delta"] for out in method_outputs], dim=1).mean(dim=1),
            "step_score": torch.stack([out["step_score"] for out in method_outputs], dim=1).amax(dim=1),
            "wall_score": torch.stack([out["wall_score"] for out in method_outputs], dim=1).amax(dim=1),
            "up_evidence": torch.stack([out["up_evidence"] for out in method_outputs], dim=1).amax(dim=1),
            "down_evidence": torch.stack([out["down_evidence"] for out in method_outputs], dim=1).amax(dim=1),
        }
        structure = self._height_scan_structure_tensors(grid)
        wall = probe["wall"] | structure["wall_like"]
        base = (probe["step_score"] > float(min_step_score)) & (probe["wall_score"] < float(max_wall_score)) & (~wall)
        diff = probe["up_evidence"] - probe["down_evidence"]
        stair_gate = structure["stair_like"] & (~structure["slope_like"]) & (~wall)
        up_step = stair_gate & base & (diff > float(direction_margin))
        down_step = stair_gate & base & (diff < -float(direction_margin))
        flat = ~(up_step | down_step | wall)
        step_height = torch.clamp(structure["edge_sharpness"], 0.0, max(float(up_max_height), float(down_max_height)))
        available = torch.ones_like(step_height)

        self._set_stair_gate_debug(
            "height_scan",
            available=available,
            up_step=up_step.float(),
            down_step=down_step.float(),
            wall=wall.float(),
            flat=flat.float(),
            step_delta=probe["step_delta"],
            step_score=probe["step_score"],
            wall_score=probe["wall_score"],
        )
        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["g_stair_noslope_m"] = self._tensor_ratio(stair_gate)
        debug["g_dom_up_02"] = self._tensor_ratio(up_step)
        debug["g_dom_down_02"] = self._tensor_ratio(down_step)
        debug["g_dom_both_02"] = self._tensor_ratio(
            stair_gate
            & base
            & (probe["up_evidence"] > float(min_step_score))
            & (probe["down_evidence"] > float(min_step_score))
            & (torch.abs(diff) <= float(direction_margin))
        )
        debug["g_step_edge_sharpness"] = self._tensor_mean(structure["edge_sharpness"])
        debug["g_step_edge_locality"] = self._tensor_mean(structure["edge_locality"])
        debug["g_slope_smoothness"] = self._tensor_mean(structure["slope_smoothness"])
        debug["g_struct_wall"] = self._tensor_ratio(structure["wall_like"])
        self.env._stair_gate_debug = debug
        self._log_height_scan_gate_status(
            available=available,
            up_step=up_step.float(),
            down_step=down_step.float(),
            wall=wall.float(),
            flat=flat.float(),
            step_delta=probe["step_delta"],
            step_score=probe["step_score"],
            wall_score=probe["wall_score"],
            reason="ok",
            log_interval=log_interval,
        )
        return {
            "available": available,
            "stair_gate": stair_gate,
            "up_step": up_step,
            "down_step": down_step,
            "wall": wall,
            "flat": flat,
            "step_delta": probe["step_delta"],
            "step_height": step_height,
            "step_score": probe["step_score"],
            "wall_score": probe["wall_score"],
            "near_z": near_z,
            "front_z": front_z,
        }

    def _height_scan_structure_tensors(self, grid):
        num_envs = grid.shape[0]
        zeros = torch.zeros(num_envs, device=grid.device, dtype=grid.dtype)
        window = grid[:, 5:11, 0:10]
        if window.shape[1] < 2 or window.shape[2] < 2:
            return {
                "edge_sharpness": zeros,
                "edge_locality": zeros,
                "slope_smoothness": zeros,
                "wall_like": zeros.bool(),
                "stair_like": zeros.bool(),
                "slope_like": zeros.bool(),
            }
        dx = window[:, :, 1:] - window[:, :, :-1]
        dy = window[:, 1:, :] - window[:, :-1, :]
        abs_edges = torch.cat(
            [torch.abs(dx).reshape(num_envs, -1), torch.abs(dy).reshape(num_envs, -1)],
            dim=1,
        )
        if abs_edges.numel() == 0:
            return {
                "edge_sharpness": zeros,
                "edge_locality": zeros,
                "slope_smoothness": zeros,
                "wall_like": zeros.bool(),
                "stair_like": zeros.bool(),
                "slope_like": zeros.bool(),
            }
        edge_sharpness = abs_edges.amax(dim=1)
        edge_mean = abs_edges.mean(dim=1)
        slope_smoothness = torch.clamp(edge_mean / (edge_sharpness + 1.0e-6), min=0.0, max=1.0)
        edge_locality = torch.clamp(1.0 - slope_smoothness, min=0.0, max=1.0)
        wall_like = edge_sharpness > 0.30
        stair_like = (
            (edge_sharpness >= 0.040)
            & (edge_locality >= 0.30)
            & (slope_smoothness <= 0.45)
            & (~wall_like)
        )
        slope_like = (
            (edge_mean >= 0.006)
            & (slope_smoothness >= 0.60)
            & (edge_sharpness <= 0.070)
            & (~wall_like)
        )
        return {
            "edge_sharpness": edge_sharpness,
            "edge_locality": edge_locality,
            "slope_smoothness": slope_smoothness,
            "wall_like": wall_like,
            "stair_like": stair_like,
            "slope_like": slope_like,
        }

    def _height_scan_edge_normal_body(
        self,
        grid,
        body_y_start: int = 5,
        body_y_end: int = 11,
        x_start: int = 0,
        x_end: int = 10,
    ):
        """Estimate the local stair-edge normal in robot body XY coordinates.

        The 16x16 height scan is already robot-aligned.  Height gradients across
        x/y scan cells approximate the normal of a discrete stair edge; the edge
        itself is perpendicular to this vector.  The sign is not important for
        crossing a stair, so downstream rewards use an absolute dot product.
        """
        num_envs = grid.shape[0]
        zeros = torch.zeros(num_envs, device=grid.device, dtype=grid.dtype)
        forward = torch.zeros((num_envs, 2), device=grid.device, dtype=grid.dtype)
        forward[:, 0] = 1.0

        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(x_start), 15))
        x1 = max(x0 + 1, min(int(x_end), 16))
        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 2 or window.shape[2] < 2:
            return forward, zeros, zeros

        dx = window[:, :, 1:] - window[:, :, :-1]
        dy = window[:, 1:, :] - window[:, :-1, :]
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        dx_weight = abs_dx.sum(dim=(1, 2)).clamp_min(1.0e-6)
        dy_weight = abs_dy.sum(dim=(1, 2)).clamp_min(1.0e-6)
        grad_x = (dx * abs_dx).sum(dim=(1, 2)) / dx_weight
        grad_y = (dy * abs_dy).sum(dim=(1, 2)) / dy_weight
        normal = torch.stack((grad_x, grad_y), dim=1)
        norm = torch.linalg.norm(normal, dim=1, keepdim=True)
        normal = torch.where(norm > 1.0e-6, normal / norm.clamp_min(1.0e-6), forward)
        edge_strength = torch.maximum(abs_dx.amax(dim=(1, 2)), abs_dy.amax(dim=(1, 2)))
        lateral_ratio = abs_dy.sum(dim=(1, 2)) / (abs_dx.sum(dim=(1, 2)) + abs_dy.sum(dim=(1, 2)) + 1.0e-6)
        return normal, edge_strength, lateral_ratio

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
                "near_z": zeros,
                "front_z": zeros,
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
        self._probe_height_scan_gate_variants(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            front_x_start=front_x_start,
            front_x_end=front_x_end,
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
            "near_z": near_z,
            "front_z": front_z,
        }

    def _height_scan_hit_grid(self):
        if not hasattr(self.env, "scene") or not hasattr(self.env.scene, "sensors"):
            return None
        sensor = self.env.scene.sensors.get("height_scanner")
        if sensor is None or not hasattr(sensor, "data"):
            return None
        ray_hits = getattr(sensor.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.shape[-2] < 256:
            return None
        hits_z = ray_hits[..., 2]
        if hits_z.dim() != 2:
            hits_z = hits_z.view(self.env.num_envs, -1)
        return hits_z[:, :256].view(self.env.num_envs, 16, 16)

    def _probe_height_scan_gate_variants(
        self,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 0,
        near_x_end: int = 4,
        front_x_start: int = 4,
        front_x_end: int = 10,
    ):
        """Record multiple height-scan gate variants for diagnosis only.

        These metrics intentionally do not feed back into any reward.  They
        compare grid-axis assumptions and mean-window vs local-edge gates in a
        single short run.
        """
        # Historical versions wrote many g_* probe metrics for threshold tests.
        # The active training monitor is now intentionally compact, so keep this
        # hook silent unless a future experiment explicitly re-enables it.
        debug = getattr(self.env, "_stair_gate_debug", {})
        debug.setdefault("hs_probe_available", 0.0)
        debug.setdefault("hs_probe_reason_code", 0.0)
        self.env._stair_gate_debug = debug
        return

        hit_grid = self._height_scan_hit_grid()
        if hit_grid is None:
            self._set_empty_height_scan_probe_debug(reason="hit_grid_unavailable")
            self._log_height_scan_probe_status(reason="hit_grid_unavailable")
            return

        threshold_sets = {
            "loose": (0.010, 0.280, 0.320, 0.002, 0.250),
            "mild": (0.015, 0.260, 0.300, 0.005, 0.200),
            "current": (0.025, 0.200, 0.240, 0.020, 0.080),
            "strict": (0.030, 0.180, 0.220, 0.030, 0.060),
        }
        methods = {
            "mean_x": (0.0, hit_grid, "mean"),
            "mean_y": (1.0, hit_grid.transpose(1, 2), "mean"),
            "edge_x": (2.0, hit_grid, "edge"),
            "edge_y": (3.0, hit_grid.transpose(1, 2), "edge"),
        }

        method_scores = {}
        method_outputs = {}
        for method_name, (_method_id, grid, mode) in methods.items():
            output = self._height_scan_probe_method(
                grid,
                mode=mode,
                thresholds=threshold_sets["mild"],
                body_y_start=body_y_start,
                body_y_end=body_y_end,
                near_x_start=near_x_start,
                near_x_end=near_x_end,
                front_x_start=front_x_start,
                front_x_end=front_x_end,
            )
            method_outputs[method_name] = output
            method_scores[method_name] = output["step_score"]

        stacked_scores = torch.stack([method_scores[name] for name in methods], dim=1)
        best_indices = torch.argmax(stacked_scores, dim=1).float()
        best_score = torch.amax(stacked_scores, dim=1)

        # Threshold summaries use the union across all four methods.  This is a
        # diagnostic "can any measurement see stairs?" probe, not a reward gate.
        threshold_outputs = {}
        for name, thresholds in threshold_sets.items():
            outputs = [
                self._height_scan_probe_method(
                    grid,
                    mode=mode,
                    thresholds=thresholds,
                    body_y_start=body_y_start,
                    body_y_end=body_y_end,
                    near_x_start=near_x_start,
                    near_x_end=near_x_end,
                    front_x_start=front_x_start,
                    front_x_end=front_x_end,
                )
                for _, grid, mode in methods.values()
            ]
            threshold_outputs[name] = {
                "up": torch.stack([out["up"].float() for out in outputs], dim=1).amax(dim=1),
                "down": torch.stack([out["down"].float() for out in outputs], dim=1).amax(dim=1),
                "wall": torch.stack([out["wall"].float() for out in outputs], dim=1).amax(dim=1),
                "step_delta": torch.stack([out["step_delta"] for out in outputs], dim=1).mean(dim=1),
                "up_evidence": torch.stack([out["up_evidence"] for out in outputs], dim=1).amax(dim=1),
                "down_evidence": torch.stack([out["down_evidence"] for out in outputs], dim=1).amax(dim=1),
                "wall_score": torch.stack([out["wall_score"] for out in outputs], dim=1).amax(dim=1),
            }

        debug = getattr(self.env, "_stair_gate_debug", {})
        for method_name, output in method_outputs.items():
            debug[f"gx_{method_name}"] = self._tensor_mean(output["step_score"])
        debug["g_best_axis"] = self._tensor_mean(best_indices)
        debug["g_best_score"] = self._tensor_mean(best_score)

        mild_probe = threshold_outputs["mild"]
        debug["g_step_delta"] = self._tensor_mean(mild_probe["step_delta"])
        debug["g_edge_pos"] = self._tensor_mean(mild_probe["up_evidence"])
        debug["g_edge_neg"] = self._tensor_mean(mild_probe["down_evidence"])
        debug["g_wall_score"] = self._tensor_mean(mild_probe["wall_score"])

        for name, output in threshold_outputs.items():
            debug[f"g_{name}_up"] = self._tensor_ratio(output["up"])
            debug[f"g_{name}_down"] = self._tensor_ratio(output["down"])
            debug[f"g_{name}_wall"] = self._tensor_ratio(output["wall"])
        debug["g_probe_available"] = 1.0
        debug["g_probe_reason_code"] = 0.0
        self.env._stair_gate_debug = debug
        self._log_height_scan_probe_status(
            reason="ok",
            available=1.0,
            best_axis=self._tensor_mean(best_indices),
            best_score=self._tensor_mean(best_score),
            loose_up=debug.get("g_loose_up", 0.0),
            mild_up=debug.get("g_mild_up", 0.0),
            current_up=debug.get("g_current_up", 0.0),
            strict_up=debug.get("g_strict_up", 0.0),
            wall=debug.get("g_mild_wall", 0.0),
        )

    def _set_empty_height_scan_probe_debug(self, reason: str):
        reason_codes = {
            "ok": 0.0,
            "hit_grid_unavailable": 1.0,
            "window_invalid": 2.0,
        }
        debug = getattr(self.env, "_stair_gate_debug", {})
        zero_keys = (
            "gx_mean_x",
            "gx_mean_y",
            "gx_edge_x",
            "gx_edge_y",
            "g_best_axis",
            "g_best_score",
            "g_step_delta",
            "g_edge_pos",
            "g_edge_neg",
            "g_wall_score",
            "g_loose_up",
            "g_loose_down",
            "g_loose_wall",
            "g_mild_up",
            "g_mild_down",
            "g_mild_wall",
            "g_current_up",
            "g_current_down",
            "g_current_wall",
            "g_strict_up",
            "g_strict_down",
            "g_strict_wall",
        )
        for key in zero_keys:
            debug[key] = 0.0
        debug["g_probe_available"] = 0.0
        debug["g_probe_reason_code"] = float(reason_codes.get(reason, 99.0))
        self.env._stair_gate_debug = debug

    def _log_height_scan_probe_status(
        self,
        reason: str,
        available: float = 0.0,
        best_axis: float = 0.0,
        best_score: float = 0.0,
        loose_up: float = 0.0,
        mild_up: float = 0.0,
        current_up: float = 0.0,
        strict_up: float = 0.0,
        wall: float = 0.0,
        log_interval: int = 50,
    ):
        if not hasattr(self.env, "_height_scan_probe_log_count"):
            self.env._height_scan_probe_log_count = 0
        self.env._height_scan_probe_log_count += 1
        count = self.env._height_scan_probe_log_count
        interval = max(int(log_interval), 1)
        if count != 1 and count % interval != 0:
            return
        self._log_reward_warning(
            "[height_scan_probe] call=%d reason=%s available=%.3f best_axis=%.3f "
            "best_score=%.4f loose_up=%.3f mild_up=%.3f current_up=%.3f strict_up=%.3f wall=%.3f",
            count,
            reason,
            float(available),
            float(best_axis),
            float(best_score),
            float(loose_up),
            float(mild_up),
            float(current_up),
            float(strict_up),
            float(wall),
        )

    def _height_scan_probe_method(
        self,
        grid,
        *,
        mode: str,
        thresholds,
        body_y_start: int,
        body_y_end: int,
        near_x_start: int,
        near_x_end: int,
        front_x_start: int,
        front_x_end: int,
    ):
        min_h, max_h, wall_h, min_score, max_wall = thresholds
        y0 = max(0, min(int(body_y_start), 15))
        y1 = max(y0 + 1, min(int(body_y_end), 16))
        x0 = max(0, min(int(near_x_start), 15))
        x_near = max(x0 + 1, min(int(near_x_end), 16))
        x_front0 = max(0, min(int(front_x_start), 15))
        x_front1 = max(x_front0 + 1, min(int(front_x_end), 16))
        x1 = max(x0 + 2, min(int(front_x_end), 16))
        zeros = torch.zeros(self.env.num_envs, device=self.env.device)

        window = grid[:, y0:y1, x0:x1]
        if window.shape[1] < 1 or window.shape[2] < 2:
            return {
                "up": zeros.bool(),
                "down": zeros.bool(),
                "wall": zeros.bool(),
                "step_delta": zeros,
                "step_score": zeros,
                "wall_score": zeros,
                "up_evidence": zeros,
                "down_evidence": zeros,
            }

        deltas = window[:, :, 1:] - window[:, :, :-1]
        abs_delta = torch.abs(deltas)
        pos_like = (deltas >= min_h) & (deltas <= max_h)
        neg_like = (deltas <= -min_h) & (deltas >= -max_h)
        wall_like = abs_delta >= wall_h
        up_evidence = pos_like.float().mean(dim=(1, 2))
        down_evidence = neg_like.float().mean(dim=(1, 2))
        step_score = (pos_like | neg_like).float().mean(dim=(1, 2))
        wall_score = wall_like.float().mean(dim=(1, 2))

        if mode == "mean":
            near = grid[:, y0:y1, x0:x_near].mean(dim=(1, 2))
            front = grid[:, y0:y1, x_front0:x_front1].mean(dim=(1, 2))
            step_delta = front - near
            up = (step_score > min_score) & (wall_score < max_wall) & (step_delta >= min_h) & (step_delta <= max_h)
            down = (step_score > min_score) & (wall_score < max_wall) & (step_delta <= -min_h) & (step_delta >= -max_h)
        else:
            positive_edge = torch.where(pos_like, deltas, torch.zeros_like(deltas)).amax(dim=(1, 2))
            negative_edge = torch.where(neg_like, -deltas, torch.zeros_like(deltas)).amax(dim=(1, 2))
            step_delta = positive_edge - negative_edge
            up = (up_evidence > min_score) & (wall_score < max_wall) & (positive_edge >= min_h)
            down = (down_evidence > min_score) & (wall_score < max_wall) & (negative_edge >= min_h)
        wall = wall_score >= max_wall
        return {
            "up": up,
            "down": down,
            "wall": wall,
            "step_delta": step_delta,
            "step_score": step_score,
            "wall_score": wall_score,
            "up_evidence": up_evidence,
            "down_evidence": down_evidence,
        }

    def _tensor_mean(self, value):
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())

    def _tensor_ratio(self, value):
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        return float((value.detach().float() > 0.0).float().mean().item())

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

    def _foot_first_contact(
        self,
        contact_sensor,
        sensor_cfg,
        contact_now,
        min_air_time: float = 0.04,
        cache_name: str = "foot_contact",
    ):
        """Return one-frame foot touchdown events with a per-reward contact latch."""
        cache_attr = f"_{cache_name}_prev_contact"
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        contact_now = contact_now.bool()
        prev_contact = getattr(self.env, cache_attr, None)
        if prev_contact is None or prev_contact.shape != contact_now.shape:
            prev_contact = contact_now.detach().clone()
            setattr(self.env, cache_attr, prev_contact)
            return torch.zeros_like(contact_now, dtype=torch.bool)

        prev_contact = prev_contact.to(contact_now.device).bool()
        valid_air_time = last_air_time >= float(min_air_time)
        reset_mask = self._reset_mask().to(contact_now.device).unsqueeze(1)
        first_contact = contact_now & (~prev_contact) & valid_air_time & (~reset_mask)
        setattr(self.env, cache_attr, contact_now.detach().clone())
        return first_contact

    def _command_xy_speed_dir(self, command_name: str = "base_velocity"):
        command = self._get_velocity_command(command_name)
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
                command = self._get_velocity_command(command_name)
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
        cmd_vy = self._get_velocity_command(command_name)[:, 1]
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
        cmd_vx = self._get_velocity_command(command_name)[:, 0]
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
        cmd = self._get_velocity_command(command_name)
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
        cmd = self._get_velocity_command(command_name)
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
        cmd = self._get_velocity_command(command_name)
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

    def _reward_down_stair_speed_safety(
        self,
        command_name: str = "base_velocity",
        min_command_speed: float = 0.10,
        min_allowed_speed: float = 0.35,
        command_speed_ratio: float = 0.65,
        speed_std: float = 0.25,
        max_vertical_speed: float = 0.35,
        vertical_std: float = 0.25,
        max_ang_vel_xy: float = 0.75,
        ang_std: float = 0.45,
        vertical_weight: float = 1.0,
        angular_weight: float = 1.0,
        max_penalty: float = 2.0,
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
        """Penalize rushing and unstable body motion on detected down-steps."""
        asset = self._get_robot_asset()
        command = self._get_velocity_command(command_name)
        command_xy = command[:, :2]
        command_speed = torch.linalg.norm(command_xy, dim=1)
        command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1.0e-6)
        actual_xy = asset.data.root_lin_vel_b[:, :2]
        projected_speed = torch.sum(actual_xy * command_dir, dim=1)

        gate = self._height_scan_reward_stair_gate(
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

        allowed_speed = torch.maximum(
            torch.full_like(command_speed, float(min_allowed_speed)),
            float(command_speed_ratio) * command_speed,
        )
        speed_excess = torch.clamp(projected_speed - allowed_speed, min=0.0)
        vertical_excess = torch.clamp(torch.abs(asset.data.root_lin_vel_b[:, 2]) - float(max_vertical_speed), min=0.0)
        ang_xy = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
        angular_excess = torch.clamp(ang_xy - float(max_ang_vel_xy), min=0.0)
        value = (
            torch.square(speed_excess / max(float(speed_std), 1.0e-6))
            + float(vertical_weight) * torch.square(vertical_excess / max(float(vertical_std), 1.0e-6))
            + float(angular_weight) * torch.square(angular_excess / max(float(ang_std), 1.0e-6))
        )
        value = torch.clamp(value, min=0.0, max=float(max_penalty))
        active = (command_speed > float(min_command_speed)).float() * gate["down_step"].float()
        value = value * active

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["down_stair_speed_safety_penalty_mean"] = self._tensor_mean(value)
        debug["down_stair_speed_safety_active_ratio"] = self._tensor_ratio(active)
        debug["down_stair_speed_safety_projected_mean"] = self._tensor_mean(projected_speed)
        debug["down_stair_speed_safety_allowed_mean"] = self._tensor_mean(allowed_speed)
        self.env._stair_gate_debug = debug
        return value

    def _reward_down_stair_touchdown_safety(
        self,
        max_foot_down_vel: float = 0.45,
        foot_vel_std: float = 0.30,
        max_contact_force: float = 180.0,
        force_std: float = 120.0,
        max_penalty: float = 2.0,
        min_air_time: float = 0.04,
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
        """Penalize hard down-step touchdowns."""
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        gate = self._height_scan_reward_stair_gate(
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

        contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
        contact_now = contact_forces > 1.0
        first_contact = self._foot_first_contact(
            contact_sensor,
            sensor_cfg,
            contact_now,
            min_air_time=min_air_time,
            cache_name="down_stair_touchdown_safety",
        )
        foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        down_vel = torch.clamp(-foot_vel_z - float(max_foot_down_vel), min=0.0)
        force_excess = torch.clamp(contact_forces - float(max_contact_force), min=0.0)
        velocity_penalty = torch.square(down_vel / max(float(foot_vel_std), 1.0e-6)) * first_contact.float()
        force_penalty = torch.square(force_excess / max(float(force_std), 1.0e-6)) * contact_now.float()
        per_foot = velocity_penalty + force_penalty
        per_foot = torch.clamp(per_foot, min=0.0, max=float(max_penalty))
        active = gate["down_step"].float().unsqueeze(1) * ((first_contact | (force_excess > 0.0)).float())
        value = torch.sum(per_foot * active, dim=1) / max(len(asset_cfg.body_ids), 1)

        debug = getattr(self.env, "_stair_gate_debug", {})
        debug["down_stair_touchdown_safety_penalty_mean"] = self._tensor_mean(value)
        debug["down_stair_touchdown_safety_active_ratio"] = self._tensor_ratio(active)
        debug["down_stair_touchdown_safety_down_vel_mean"] = self._tensor_mean(torch.clamp(-foot_vel_z, min=0.0))
        self.env._stair_gate_debug = debug
        return value

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
        cmd = self._get_velocity_command(command_name)
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
