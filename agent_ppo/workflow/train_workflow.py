#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import Frame
import os
import time
from typing import List, Optional, Tuple
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from tools.utils import load_reward_keys_from_monitor_config
import torch
from collections import deque, defaultdict


class VelocityCurriculum:
    """Performance-based velocity curriculum, mirroring terrain curriculum logic.

    Promotes to next velocity stage when mean reward_track_lin_vel_xy >
    promote_threshold for promote_count consecutive episode-batch checks.
    Demotes to previous stage when it falls below demote_threshold for
    demote_count consecutive checks. Neither counter changes when the metric
    stays in the neutral zone [demote_threshold, promote_threshold).

    Completely independent of terrain.curriculum — terrain difficulty is managed
    separately (num_rows=10, difficulty_range=[0,1.0]). Stage changes call
    env.reset(usr_conf) to apply new velocity ranges.
    Policy weights are NOT affected — only env command-sampling changes.

    Configuration is loaded from usr_conf["velocity_curriculum"] (TOML section
    [velocity_curriculum]), so all thresholds and stage definitions live in the
    TOML file rather than being hard-coded here.

    性能驱动的速度课程，复用地形课程逻辑（表现好则升级，差则降级）。
    与 terrain.curriculum 完全独立——地形难度由 TOML [terrain] 节独立管理
    （10 档，覆盖完整 [0,1] 难度带）。
    所有阈值和阶段定义均从 TOML [velocity_curriculum] 节读取。
    """

    # Default stages used only when usr_conf has no [velocity_curriculum] section.
    # 仅当 TOML 缺少 [velocity_curriculum] 节时作为退路默认值。
    _DEFAULT_STAGES = [
        {"lin_vel_x": [0.0,  0.5], "lin_vel_y": [-0.3,  0.3], "ang_vel_yaw": [-1.0,  1.0]},
        {"lin_vel_x": [0.0,  1.0], "lin_vel_y": [-0.5,  0.5], "ang_vel_yaw": [-1.5,  1.5]},
        {"lin_vel_x": [0.0,  1.5], "lin_vel_y": [-0.8,  0.8], "ang_vel_yaw": [-1.5,  1.5]},
        {"lin_vel_x": [-0.5, 2.0], "lin_vel_y": [-1.0,  1.0], "ang_vel_yaw": [-1.5,  1.5]},
    ]

    # ep_info key used as performance signal.  Different framework versions may
    # expose reward terms with slightly different names, so lookup is tolerant.
    # 用于速度课程的 episode 指标。不同框架版本可能使用略有差异的 reward key，
    # 因此实际查找时会做兼容匹配。
    _TRACKING_KEY = "reward_track_lin_vel_xy"
    _TRACKING_KEY_CANDIDATES = (
        "reward_track_lin_vel_xy",
        "track_lin_vel_xy",
        "Episode_Reward/reward_track_lin_vel_xy",
        "Episode_Reward/track_lin_vel_xy",
    )

    def __init__(self, logger, usr_conf: dict):
        """Build curriculum from usr_conf["velocity_curriculum"] (TOML section).

        Falls back to _DEFAULT_STAGES / hard-coded thresholds if the section is absent.
        从 TOML 的 [velocity_curriculum] 节加载配置；节缺失时回退到默认值。
        """
        self.logger = logger
        fine_tune_conf = usr_conf.get("fine_tune_schedule", {})
        self._commands_owned_by_schedule = bool(
            fine_tune_conf.get("enabled", False)
            and (
                fine_tune_conf.get("command_initial", {})
                or fine_tune_conf.get("command_target", {})
            )
        )
        vc_conf = usr_conf.get("velocity_curriculum", {})
        tracking_reward_conf = usr_conf.get("rewards", {}).get("track_lin_vel_xy", {})

        self._tracking_reward_weight = abs(float(tracking_reward_conf.get("weight", 1.0)))
        if self._tracking_reward_weight <= 1e-6:
            logger.warning(
                "[VelocityCurriculum] track_lin_vel_xy weight is non-positive; "
                "falling back to 1.0 for curriculum normalization. "
                "Please check [rewards.track_lin_vel_xy].weight in TOML."
            )
            self._tracking_reward_weight = 1.0

        self.promote_threshold: float = self._normalize_threshold(
            float(vc_conf.get("promote_threshold", 0.64)), "promote_threshold"
        )
        self.demote_threshold:  float = self._normalize_threshold(
            float(vc_conf.get("demote_threshold", 0.32)), "demote_threshold"
        )
        self.promote_count:     int   = int(vc_conf.get("promote_count", 5))
        self.demote_count:      int   = int(vc_conf.get("demote_count",  3))
        self.min_checks_per_stage: int = max(0, int(vc_conf.get("min_checks_per_stage", 0)))

        raw_stages = vc_conf.get("stages", None)
        if raw_stages:
            self.STAGES = [
                {
                    "lin_vel_x":   list(s["lin_vel_x"]),
                    "lin_vel_y":   list(s["lin_vel_y"]),
                    "ang_vel_yaw": list(s["ang_vel_yaw"]),
                }
                for s in raw_stages
            ]
        else:
            self.STAGES = self._DEFAULT_STAGES
            logger.warning(
                "[VelocityCurriculum] No [velocity_curriculum.stages] found in usr_conf; "
                "falling back to hard-coded default stages."
            )

        command_ranges = usr_conf.get("commands", {}).get("ranges", {})
        stage0 = self.STAGES[0]
        if (
            list(command_ranges.get("lin_vel_x", [])) != stage0["lin_vel_x"]
            or list(command_ranges.get("lin_vel_y", [])) != stage0["lin_vel_y"]
            or list(command_ranges.get("ang_vel_yaw", [])) != stage0["ang_vel_yaw"]
        ):
            raise ValueError(
                "Velocity curriculum Stage 0 must exactly match [commands.ranges] in TOML. "
                f"Got commands.ranges={command_ranges}, stage0={stage0}."
            )

        self._stage_idx = 0
        self._promote_streak = 0  # consecutive checks above promote_threshold
        self._demote_streak = 0   # consecutive checks below demote_threshold
        self._last_mean_tracking_reward = 0.0
        self._last_mean_tracking_ratio = 0.0
        self._tracking_key_resolved = None
        self._tracking_key_warning_logged = False
        self._debug_check_count = 0
        self._stage_enter_check = 0
        logger.info(
            f"[VelocityCurriculum] Initialized: {len(self.STAGES)} stages, "
            f"tracking_weight={self._tracking_reward_weight}, "
            f"promote_threshold={self.promote_threshold}, demote_threshold={self.demote_threshold}, "
            f"promote_count={self.promote_count}, demote_count={self.demote_count}, "
            f"min_checks_per_stage={self.min_checks_per_stage}, "
            f"commands_owned_by_schedule={self._commands_owned_by_schedule}"
        )

    def _normalize_threshold(self, threshold_value: float, field_name: str) -> float:
        """Normalize legacy absolute thresholds into reward-ratio thresholds.

        Historical configs stored absolute weighted reward thresholds (e.g. 1.6).
        That makes curriculum behavior drift every time reward weight changes.
        New configs should store ratios in [0, 1], e.g. 0.55 means 55% of the
        current track_lin_vel_xy maximum reward.
        历史配置使用加权 reward 的绝对阈值（如 1.6），reward weight 一改就会漂。
        现在统一转为比例阈值：[0,1] 区间，0.55 表示达到当前最大 tracking reward 的 55%。
        """
        if threshold_value <= 1.0:
            return threshold_value

        normalized = threshold_value / self._tracking_reward_weight
        self.logger.warning(
            f"[VelocityCurriculum] {field_name}={threshold_value} detected as legacy absolute reward; "
            f"normalized to ratio {normalized:.3f} using tracking weight {self._tracking_reward_weight:.3f}."
        )
        return normalized

    @property
    def stage(self) -> int:
        return self._stage_idx

    @property
    def last_tracking_reward(self) -> float:
        return self._last_mean_tracking_reward

    @property
    def last_tracking_ratio(self) -> float:
        return self._last_mean_tracking_ratio

    def monitor_stats(self):
        cfg = self.STAGES[self._stage_idx]
        lin_y = cfg.get("lin_vel_y", [0.0, 0.0])
        yaw = cfg.get("ang_vel_yaw", [0.0, 0.0])
        return {
            "vel_cmd_vy_abs_max": max(abs(float(lin_y[0])), abs(float(lin_y[1]))) if len(lin_y) >= 2 else 0.0,
            "vel_cmd_wz_abs_max": max(abs(float(yaw[0])), abs(float(yaw[1]))) if len(yaw) >= 2 else 0.0,
        }

    def _resolve_tracking_metric(self, ep_info):
        """Return the tracking metric value and the key used to find it.

        返回 episode info 中的速度追踪指标值，以及命中的 key。
        """
        if self._tracking_key_resolved and self._tracking_key_resolved in ep_info:
            return ep_info[self._tracking_key_resolved], self._tracking_key_resolved

        for key in self._TRACKING_KEY_CANDIDATES:
            if key in ep_info:
                self._tracking_key_resolved = key
                return ep_info[key], key

        for key, value in ep_info.items():
            normalized = str(key).replace("/", "_")
            if normalized.endswith("track_lin_vel_xy"):
                self._tracking_key_resolved = key
                return value, key

        return None, None

    def _mean_tracking_reward(self, ep_infos) -> Tuple[Optional[float], Optional[float]]:
        """Average reward_track_lin_vel_xy across completed episodes.

        Returns both the raw weighted reward and the normalized ratio.
        返回加权后的原始 tracking reward，以及相对当前 reward weight 的归一化比例。
        """
        values = []
        for ep_info in ep_infos:
            v, key = self._resolve_tracking_metric(ep_info)
            if key is None:
                continue
            values.append(v.float().mean().item() if isinstance(v, torch.Tensor) else float(v))
        if not values:
            if ep_infos and not self._tracking_key_warning_logged:
                sample_keys = list(ep_infos[0].keys())
                self.logger.warning(
                    "[VelocityCurriculum] Cannot find tracking metric for velocity curriculum. "
                    f"Tried keys={self._TRACKING_KEY_CANDIDATES}; "
                    f"sample episode keys={sample_keys[:40]}"
                )
                self._tracking_key_warning_logged = True
            return None, None

        mean_reward = sum(values) / len(values)
        mean_ratio = mean_reward / self._tracking_reward_weight
        return mean_reward, mean_ratio

    def _apply_stage(self, usr_conf, env, obs, critic_obs):
        """Write current stage ranges into usr_conf and call env.reset.
        Returns (obs, critic_obs, reset_happened: bool).
        reset_happened=False on env.reset failure so caller skips stat-tensor zeroing.
        """
        cfg = self.STAGES[self._stage_idx]
        usr_conf["commands"]["ranges"]["lin_vel_x"]   = cfg["lin_vel_x"]
        usr_conf["commands"]["ranges"]["lin_vel_y"]   = cfg["lin_vel_y"]
        usr_conf["commands"]["ranges"]["ang_vel_yaw"] = cfg["ang_vel_yaw"]
        self.logger.info(
            f"[VelocityCurriculum] → Stage {self._stage_idx}: "
            f"lin_vel_x={cfg['lin_vel_x']}, lin_vel_y={cfg['lin_vel_y']}, "
            f"ang_vel_yaw={cfg['ang_vel_yaw']} — calling env.reset"
        )
        data = env.reset(usr_conf)
        if data is None:
            self.logger.error("[VelocityCurriculum] env.reset failed after stage change!")
            raise RuntimeError("VelocityCurriculum env.reset failed after stage change")
        new_obs, new_critic_obs = data
        if new_critic_obs is None:
            new_critic_obs = new_obs
        return torch.clone(new_obs), torch.clone(new_critic_obs), True

    def check_and_update(self, ep_infos, usr_conf, env, obs, critic_obs, rollout_stats=None):
        """Check episode-batch performance and promote / demote stage if warranted.

        Call this BEFORE ep_infos.clear() so the current batch's data is available.
        Returns (obs, critic_obs, reset_happened: bool).
        reset_happened=True means env.reset was called; caller should zero
        cur_reward_sum / cur_episode_length to avoid stale statistics.

        在 ep_infos.clear() 前调用，确保能读到当前批次的数据。
        reset_happened=True 时调用方需清零 cur_reward_sum / cur_episode_length，
        避免 env.reset 后旧累计值污染 rewbuffer 统计。
        """
        self._debug_check_count += 1
        rollout_stats = rollout_stats or {}
        mean_reward, mean_ratio = self._mean_tracking_reward(ep_infos)
        metric_source = "episode"
        if mean_reward is None or mean_ratio is None:
            rollout_reward = rollout_stats.get("rollout_track_lin_vel_xy_reward")
            rollout_ratio = rollout_stats.get("rollout_track_lin_vel_xy_ratio")
            if rollout_reward is not None and rollout_ratio is not None:
                mean_reward = float(rollout_reward)
                mean_ratio = float(rollout_ratio)
                metric_source = "rollout_critic_obs"

        if mean_reward is None or mean_ratio is None:
            if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0:
                sample_keys = list(ep_infos[0].keys())[:40] if ep_infos else []
                self.logger.warning(
                    "[VelocityCurriculumDebug] no tracking metric available; "
                    f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                    f"resolved_key={self._tracking_key_resolved}, sample_keys={sample_keys}, "
                    f"rollout_stats_keys={list(rollout_stats.keys())}"
                )
            return obs, critic_obs, False

        self._last_mean_tracking_reward = mean_reward
        self._last_mean_tracking_ratio = mean_ratio

        if self._commands_owned_by_schedule:
            self._promote_streak = 0
            self._demote_streak = 0
            if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0:
                current_cfg = self.STAGES[self._stage_idx]
                self.logger.warning(
                    "[VelocityCurriculumDebug] "
                    f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                    f"source={metric_source}, key={self._tracking_key_resolved}, "
                    f"reward={mean_reward:.4f}, "
                    f"ratio={mean_ratio:.4f}, stage={self._stage_idx}->{self._stage_idx}, "
                    "command_update=disabled_by_fine_tune_schedule, "
                    f"ranges={current_cfg}"
                )
            return obs, critic_obs, False

        stage_changed = False
        old_stage = self._stage_idx

        checks_in_stage = self._debug_check_count - self._stage_enter_check
        stage_age_ready = checks_in_stage >= self.min_checks_per_stage

        if mean_ratio >= self.promote_threshold:
            self._promote_streak += 1
            self._demote_streak = 0
            if (
                stage_age_ready
                and self._promote_streak >= self.promote_count
                and self._stage_idx < len(self.STAGES) - 1
            ):
                self._stage_idx += 1
                self._promote_streak = 0
                self._stage_enter_check = self._debug_check_count
                self.logger.warning(
                    f"[VelocityCurriculum] PROMOTE ↑ stage {self._stage_idx} "
                    f"(tracking_ratio={mean_ratio:.3f} >= {self.promote_threshold:.3f}, "
                    f"tracking_reward={mean_reward:.3f}, "
                    f"for {self.promote_count} consecutive checks, "
                    f"stage_age={checks_in_stage}/{self.min_checks_per_stage})"
                )
                stage_changed = True
        elif mean_ratio < self.demote_threshold:
            self._demote_streak += 1
            self._promote_streak = 0
            if self._demote_streak >= self.demote_count and self._stage_idx > 0:
                self._stage_idx -= 1
                self._demote_streak = 0
                self._stage_enter_check = self._debug_check_count
                self.logger.warning(
                    f"[VelocityCurriculum] DEMOTE ↓ stage {self._stage_idx} "
                    f"(tracking_ratio={mean_ratio:.3f} < {self.demote_threshold:.3f}, "
                    f"tracking_reward={mean_reward:.3f}, "
                    f"for {self.demote_count} consecutive checks)"
                )
                stage_changed = True
        else:
            # Neutral zone: slowly decay both streaks to avoid oscillation
            # 中性区间：缓慢衰减两个计数器，避免振荡。
            self._promote_streak = max(0, self._promote_streak - 1)
            self._demote_streak = max(0, self._demote_streak - 1)

        if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0 or stage_changed:
            current_cfg = self.STAGES[self._stage_idx]
            self.logger.warning(
                "[VelocityCurriculumDebug] "
                f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                f"source={metric_source}, key={self._tracking_key_resolved}, "
                f"reward={mean_reward:.4f}, "
                f"ratio={mean_ratio:.4f}, stage={old_stage}->{self._stage_idx}, "
                f"promote={self._promote_streak}/{self.promote_count} "
                f"@{self.promote_threshold:.3f}, "
                f"demote={self._demote_streak}/{self.demote_count} "
                f"@{self.demote_threshold:.3f}, "
                f"stage_age={checks_in_stage}/{self.min_checks_per_stage}, "
                f"ranges={current_cfg}"
            )

        if stage_changed:
            return self._apply_stage(usr_conf, env, obs, critic_obs)
        return obs, critic_obs, False


class FineTuneSchedule:
    """Wall-clock-first linear schedule for conservative HJC fine-tuning."""

    def __init__(self, logger, usr_conf: dict):
        self.logger = logger
        self.conf = usr_conf.get("fine_tune_schedule", {})
        self.enabled = bool(self.conf.get("enabled", False))
        self.transition_seconds = float(self.conf.get("transition_seconds", 900.0))
        self.transition_episodes = max(1, int(self.conf.get("fallback_transition_episodes", 200)))
        self.update_interval_episodes = max(1, int(self.conf.get("update_interval_episodes", 10)))
        self.log_interval_episodes = max(1, int(self.conf.get("log_interval_episodes", 10)))
        self.start_time = time.time()
        self._last_applied_episode = None
        self._last_alpha = 0.0
        self._last_reset_happened = False
        self._runtime_reward_weights = {}
        self._runtime_command_ranges = {}
        self._runtime_command_limits = {}

        self.reward_initial = dict(self.conf.get("reward_initial", {}))
        self.reward_target = dict(self.conf.get("reward_target", {}))
        self.command_initial = self._normalize_command_ranges(self.conf.get("command_initial", {}))
        self.command_target = self._normalize_command_ranges(self.conf.get("command_target", {}))
        self.terrain_phases = self._normalize_terrain_phases(self.conf.get("terrain_phases", []))
        self._current_terrain_phase_idx = None

        if not self.enabled:
            logger.warning("[FineTuneSchedule] disabled: no schedule will be applied.")
            self._log_reward_registration(usr_conf)
            return

        self._log_initial_alignment(usr_conf)
        self._apply_rewards(usr_conf, self.reward_initial)
        self._apply_commands(usr_conf, self.command_initial)
        self._apply_terrain_phase(usr_conf, self._terrain_phase_for_alpha(0.0))
        self._current_command_ranges = dict(usr_conf.get("commands", {}).get("ranges", {}))
        self._log_reward_registration(usr_conf)
        logger.warning(
            f"[FineTuneSchedule] initialized transition_seconds={self.transition_seconds:.1f} "
            f"fallback_episodes={self.transition_episodes} "
            f"update_interval={self.update_interval_episodes} "
            f"reward_initial={self.reward_initial} reward_target={self.reward_target} "
            f"command_initial={self.command_initial} command_target={self.command_target} "
            f"terrain_phases={self.terrain_phases}"
        )

    @property
    def last_alpha(self) -> float:
        return self._last_alpha

    def monitor_stats(self):
        ranges = self.command_target if self.command_target else {}
        current = getattr(self, "_current_command_ranges", {})
        lin_y = current.get("lin_vel_y", ranges.get("lin_vel_y", [0.0, 0.0]))
        yaw = current.get("ang_vel_yaw", ranges.get("ang_vel_yaw", [0.0, 0.0]))
        return {
            "scheduled_alpha": self._last_alpha,
            "scheduled_elapsed_seconds": max(0.0, time.time() - self.start_time),
            "scheduled_cmd_vy_abs_max": max(abs(float(lin_y[0])), abs(float(lin_y[1]))) if len(lin_y) >= 2 else 0.0,
            "scheduled_cmd_wz_abs_max": max(abs(float(yaw[0])), abs(float(yaw[1]))) if len(yaw) >= 2 else 0.0,
            "sched_track_lin_w": self._reward_weight("track_lin_vel_xy"),
            "sched_track_yaw_w": self._reward_weight("track_ang_vel_z"),
            "sched_hs_clear_w": self._reward_weight("height_scan_feet_clearance"),
            "sched_hs_wall_w": self._reward_weight("height_scan_wall_reject"),
            "sched_hs_probe_w": self._reward_weight("height_scan_gate_probe"),
            "sched_relax_ang_w": self._reward_weight("stair_relax_ang_vel_xy"),
            "sched_relax_height_w": self._reward_weight("stair_relax_base_height"),
            "sched_relax_joint_w": self._reward_weight("stair_relax_joint_position"),
            "sched_pivot_w": self._reward_weight("pivot_turning"),
            "sched_flat_orient_w": self._reward_weight("flat_orientation"),
            "sched_base_height_w": self._reward_weight("correct_base_height"),
            "scheduled_terrain_phase": float(self._current_terrain_phase_idx or 0),
        }

    def check_and_update(self, usr_conf, env, obs, critic_obs, episode: int):
        if not self.enabled:
            return obs, critic_obs, False
        if self._last_applied_episode is not None and episode - self._last_applied_episode < self.update_interval_episodes:
            return obs, critic_obs, False

        alpha = self._alpha(episode)
        reward_values = self._interpolate_dict(self.reward_initial, self.reward_target, alpha)
        command_values = self._interpolate_commands(self.command_initial, self.command_target, alpha)
        terrain_phase = self._terrain_phase_for_alpha(alpha)
        self._apply_rewards(usr_conf, reward_values)
        command_changed = self._apply_commands(usr_conf, command_values)
        terrain_changed = self._apply_terrain_phase(usr_conf, terrain_phase)
        reward_runtime = self._apply_rewards_runtime(env, reward_values)
        command_runtime = self._apply_commands_runtime(env, command_values)
        self._last_alpha = alpha
        self._last_applied_episode = episode

        should_log = episode <= self.log_interval_episodes or episode % self.log_interval_episodes == 0 or command_changed
        if should_log:
            self.logger.warning(
                f"[FineTuneSchedule] apply episode={episode} alpha={alpha:.3f} "
                f"elapsed={time.time() - self.start_time:.1f}s rewards={reward_values} "
                f"commands={command_values} command_changed={command_changed} "
                f"terrain_changed={terrain_changed} terrain_phase={terrain_phase} "
                f"runtime_rewards={reward_runtime} runtime_commands={command_runtime}"
            )

        needs_reset = self._needs_reset_for_schedule_apply(reward_runtime, command_runtime)
        if needs_reset or terrain_changed:
            data = env.reset(usr_conf)
            if data is None:
                self.logger.error("[FineTuneSchedule] env.reset failed after scheduled config update!")
                raise RuntimeError("FineTuneSchedule env.reset failed after scheduled config update")
            new_obs, new_critic_obs = data
            if new_critic_obs is None:
                new_critic_obs = new_obs
            if should_log:
                reset_reason = "terrain_phase_changed" if terrain_changed and not needs_reset else "runtime_manager_update_unavailable"
                self.logger.warning(
                    f"[FineTuneSchedule] applied_by=env_reset reason={reset_reason} "
                    f"episode={episode} alpha={alpha:.3f} terrain_changed={terrain_changed}"
                )
            return torch.clone(new_obs), torch.clone(new_critic_obs), True

        if should_log:
            self.logger.warning(
                f"[FineTuneSchedule] applied_by=runtime_manager episode={episode} alpha={alpha:.3f}"
            )
        return obs, critic_obs, False

    def _alpha(self, episode: int) -> float:
        if self.transition_seconds > 0.0:
            return max(0.0, min(1.0, (time.time() - self.start_time) / self.transition_seconds))
        return max(0.0, min(1.0, float(episode) / float(self.transition_episodes)))

    def _reward_weight(self, name: str) -> float:
        rewards = getattr(self, "_usr_conf_ref", {}).get("rewards", {})
        return float(rewards.get(name, {}).get("weight", 0.0))

    def _log_initial_alignment(self, usr_conf):
        rewards = usr_conf.get("rewards", {})
        mismatches = []
        for name, initial_value in self.reward_initial.items():
            reward_conf = rewards.get(name)
            if not isinstance(reward_conf, dict) or "weight" not in reward_conf:
                mismatches.append(f"{name}:missing_static->{float(initial_value):.6g}")
                continue
            static_value = float(reward_conf.get("weight", 0.0))
            initial_value = float(initial_value)
            if abs(static_value - initial_value) > 1.0e-9:
                mismatches.append(f"{name}:{static_value:.6g}->{initial_value:.6g}")
        if mismatches:
            self.logger.warning(
                "[FineTuneSchedule] static reward weights differ from reward_initial; "
                f"startup reset will apply reward_initial. mismatches={mismatches}"
            )
        else:
            self.logger.warning("[FineTuneSchedule] static reward weights match reward_initial.")

    def _apply_rewards(self, usr_conf, values):
        self._usr_conf_ref = usr_conf
        rewards = usr_conf.setdefault("rewards", {})
        for key, value in values.items():
            rewards.setdefault(key, {})["weight"] = float(value)

    def _apply_commands(self, usr_conf, values):
        if not values:
            return False
        ranges = usr_conf.setdefault("commands", {}).setdefault("ranges", {})
        limits = usr_conf.setdefault("commands", {}).setdefault("limit", {})
        changed = False
        for key, value in values.items():
            value = [float(value[0]), float(value[1])]
            if list(ranges.get(key, [])) != value:
                changed = True
            ranges[key] = value
            limit_key = "ang_vel_z" if key == "ang_vel_yaw" else key
            if limit_key in limits:
                limits[limit_key] = [float(value[0]), float(value[1])]
        self._current_command_ranges = dict(ranges)
        return changed

    def _apply_terrain_phase(self, usr_conf, phase):
        if not phase:
            return False
        terrain = usr_conf.setdefault("terrain", {})
        standard = terrain.setdefault("standard", {})
        changed = False

        if "difficulty_range" in phase:
            value = [float(phase["difficulty_range"][0]), float(phase["difficulty_range"][1])]
            if list(terrain.get("difficulty_range", [])) != value:
                changed = True
            terrain["difficulty_range"] = value

        if "max_init_terrain_level" in phase:
            value = int(phase["max_init_terrain_level"])
            if int(terrain.get("max_init_terrain_level", -1)) != value:
                changed = True
            terrain["max_init_terrain_level"] = value

        for key in (
            "pyramid_slope",
            "pyramid_slope_inv",
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "maze",
        ):
            if key not in phase:
                continue
            value = float(phase[key])
            current = float(standard.get(key, {}).get("proportion", -1.0))
            if abs(current - value) > 1.0e-9:
                changed = True
            standard.setdefault(key, {})["proportion"] = value

        return changed

    def _apply_rewards_runtime(self, env, values):
        isaac_env = _get_isaac_env(env)
        if isaac_env is None:
            return {
                "ok": False,
                "reason": "isaac_env_unavailable",
                "candidates": _describe_env_unwrap_candidates(env, max_items=8),
            }
        reward_manager = getattr(isaac_env, "reward_manager", None)
        if reward_manager is None:
            return {"ok": False, "reason": "reward_manager_unavailable"}

        result = {}
        for name, value in values.items():
            try:
                cfg = reward_manager.get_term_cfg(name)
                if cfg is None:
                    result[name] = {"ok": False, "reason": "term_missing"}
                    continue
                old_weight = float(getattr(cfg, "weight", 0.0))
                cfg.weight = float(value)
                set_term_cfg = getattr(reward_manager, "set_term_cfg", None)
                if callable(set_term_cfg):
                    set_term_cfg(name, cfg)
                readback_cfg = reward_manager.get_term_cfg(name)
                readback = float(getattr(readback_cfg, "weight", float(value)))
                self._runtime_reward_weights[name] = readback
                result[name] = {
                    "ok": abs(readback - float(value)) <= 1.0e-6,
                    "old": round(old_weight, 6),
                    "target": round(float(value), 6),
                    "readback": round(readback, 6),
                }
            except Exception as exc:
                result[name] = {"ok": False, "reason": type(exc).__name__, "detail": str(exc)[:160]}
        return result

    @staticmethod
    def _runtime_update_ok(result):
        if not isinstance(result, dict):
            return False
        if result.get("ok") is False:
            return False
        if result.get("ok") is True:
            return True
        term_results = [value for value in result.values() if isinstance(value, dict)]
        if not term_results:
            return False
        return all(bool(value.get("ok", False)) for value in term_results)

    def _needs_reset_for_schedule_apply(self, reward_runtime, command_runtime):
        return not (
            self._runtime_update_ok(reward_runtime)
            and self._runtime_update_ok(command_runtime)
        )

    def _apply_commands_runtime(self, env, values):
        if not values:
            return {"ok": True, "reason": "no_values"}
        isaac_env = _get_isaac_env(env)
        if isaac_env is None:
            return {
                "ok": False,
                "reason": "isaac_env_unavailable",
                "candidates": _describe_env_unwrap_candidates(env, max_items=8),
            }
        command_manager = getattr(isaac_env, "command_manager", None)
        if command_manager is None:
            return {"ok": False, "reason": "command_manager_unavailable"}
        try:
            command_term = command_manager.get_term("base_velocity")
        except Exception as exc:
            return {"ok": False, "reason": type(exc).__name__, "detail": str(exc)[:160]}
        cfg = getattr(command_term, "cfg", None)
        if cfg is None:
            return {"ok": False, "reason": "command_cfg_unavailable"}

        mapping = {"lin_vel_x": "lin_vel_x", "lin_vel_y": "lin_vel_y", "ang_vel_yaw": "ang_vel_z"}
        applied_ranges = {}
        applied_limits = {}
        try:
            ranges = getattr(cfg, "ranges", None)
            limit_ranges = getattr(cfg, "limit_ranges", None)
            for key, value in values.items():
                attr = mapping.get(key)
                if attr is None:
                    continue
                pair = (float(value[0]), float(value[1]))
                if ranges is not None and hasattr(ranges, attr):
                    setattr(ranges, attr, pair)
                    applied_ranges[key] = list(pair)
                if limit_ranges is not None and hasattr(limit_ranges, attr):
                    setattr(limit_ranges, attr, pair)
                    applied_limits[key] = list(pair)
            self._runtime_command_ranges = applied_ranges
            self._runtime_command_limits = applied_limits
            return {"ok": True, "ranges": applied_ranges, "limits": applied_limits}
        except Exception as exc:
            return {"ok": False, "reason": type(exc).__name__, "detail": str(exc)[:160]}

    def _normalize_command_ranges(self, values):
        out = {}
        for key, value in values.items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                out[key] = [float(value[0]), float(value[1])]
        return out

    def _normalize_terrain_phases(self, phases):
        out = []
        if not isinstance(phases, list):
            return out
        terrain_keys = (
            "pyramid_slope",
            "pyramid_slope_inv",
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "maze",
        )
        for idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                continue
            normalized = {"alpha": float(phase.get("alpha", 0.0 if idx == 0 else 1.0))}
            if "difficulty_range" in phase and len(phase["difficulty_range"]) >= 2:
                normalized["difficulty_range"] = [
                    float(phase["difficulty_range"][0]),
                    float(phase["difficulty_range"][1]),
                ]
            if "max_init_terrain_level" in phase:
                normalized["max_init_terrain_level"] = int(phase["max_init_terrain_level"])
            for key in terrain_keys:
                if key in phase:
                    normalized[key] = float(phase[key])
            total = sum(float(normalized.get(key, 0.0)) for key in terrain_keys)
            if any(key in normalized for key in terrain_keys) and abs(total - 1.0) > 1e-6:
                self.logger.warning(
                    f"[FineTuneSchedule] terrain phase idx={idx} proportion sum={total:.6f}, "
                    "expected 1.0; phase will still be applied for debugging."
                )
            out.append(normalized)
        out.sort(key=lambda item: item.get("alpha", 0.0))
        return out

    def _terrain_phase_for_alpha(self, alpha):
        if not self.terrain_phases:
            self._current_terrain_phase_idx = None
            return {}
        selected_idx = 0
        for idx, phase in enumerate(self.terrain_phases):
            if alpha + 1.0e-9 >= float(phase.get("alpha", 0.0)):
                selected_idx = idx
            else:
                break
        self._current_terrain_phase_idx = selected_idx
        return dict(self.terrain_phases[selected_idx])

    def _interpolate_dict(self, initial, target, alpha):
        keys = set(initial) | set(target)
        out = {}
        for key in keys:
            start = float(initial.get(key, target.get(key, 0.0)))
            end = float(target.get(key, start))
            out[key] = start + (end - start) * alpha
        return out

    def _interpolate_commands(self, initial, target, alpha):
        keys = set(initial) | set(target)
        out = {}
        for key in keys:
            start = initial.get(key, target.get(key, [0.0, 0.0]))
            end = target.get(key, start)
            out[key] = [
                float(start[0]) + (float(end[0]) - float(start[0])) * alpha,
                float(start[1]) + (float(end[1]) - float(start[1])) * alpha,
            ]
        return out

    def _log_reward_registration(self, usr_conf):
        rewards = usr_conf.get("rewards", {})
        for name in (
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "height_scan_feet_clearance",
            "height_scan_wall_reject",
            "height_scan_gate_probe",
            "stair_relax_ang_vel_xy",
            "stair_relax_base_height",
            "stair_relax_joint_position",
            "feet_air_time",
            "air_time_variance_penalty",
            "base_lateral_vel",
            "pivot_turning",
        ):
            if name in rewards:
                self.logger.warning(
                    f"[RewardDebug] {name} configured weight={rewards[name].get('weight')} "
                    f"params={rewards[name].get('params', {})}"
                )
            else:
                self.logger.warning(
                    f"[RewardDebug] {name} not configured in TOML; no reward will be computed."
                )


def _initialize_training_state(env, agent, logger):
    """
    Initialize training state including storage, buffers, and observations.
    初始化训练状态，包括存储、缓冲区和观测。

    Returns:
        tuple: (storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
        返回值：(storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
    """
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)

    terrain_conf = usr_conf.get("terrain", {}).get("standard", {})
    terrain_keys = (
        "pyramid_slope",
        "pyramid_slope_inv",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "maze",
    )
    terrain_total = sum(float(terrain_conf.get(key, {}).get("proportion", 0.0)) for key in terrain_keys)
    if abs(terrain_total - 1.0) > 1e-6:
        message = (
            f"Invalid standard terrain proportions: sum={terrain_total:.6f}, expected 1.0. "
            f"Please check {usr_conf_file}."
        )
        logger.error(message)
        raise ValueError(message)

    # Validate configuration before proceeding
    # 在继续之前校验配置
    from tools.train_env_conf_validate import check_usr_conf

    valid, message = check_usr_conf(usr_conf, is_eval=False, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    # Set model to training mode
    # 设置模型为训练模式
    agent.algorithm.actor_critic.train()

    # Initialize buffers and statistics
    # 初始化缓冲区和统计信息
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    # Use algorithm's internal storage (same object used by learn())
    # 使用算法内部的 storage（与 learn() 使用同一个对象）
    storage = agent.algorithm.storage

    # Reset environment and get initial observations
    # 重置环境并获取初始观测
    data = env.reset(usr_conf)
    if data is None:
        error_message = "reset failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    obs, critic_obs = data
    if critic_obs is None:
        critic_obs = obs
    obs = torch.clone(obs)
    critic_obs = torch.clone(critic_obs)
    logger.info(f"obs.shape:{obs.shape}, critic_obs.shape:{critic_obs.shape}")

    # Load reward keys from monitor config
    # 从 monitor 配置加载 reward_keys
    reward_keys = load_reward_keys_from_monitor_config()
    logger.info(f"reward_keys list is {reward_keys}")

    return (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Main training workflow.
    主训练工作流。
    """
    agent = agents[0]
    env = envs[0]

    # Initialize training state
    # 初始化训练状态
    (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    ) = _initialize_training_state(env, agent, logger)

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0
    episode = 0

    # Velocity curriculum: expands command ranges independently of terrain curriculum.
    # terrain difficulty spans the full [0, 1.0] band via difficulty_range in TOML,
    # with 10 curriculum rows and initial placement capped at level 0;
    # velocity stages expand independently via VelocityCurriculum.
    # 速度课程：独立于地形课程扩大速度指令范围。
    # 地形难度由 TOML difficulty_range=[0,1.0] + 10 个课程档位独立限制，
    # 初始放置等级上限为 0；速度范围由 VelocityCurriculum 逐阶扩大。
    vel_curriculum = VelocityCurriculum(logger, usr_conf)
    fine_tune_schedule = FineTuneSchedule(logger, usr_conf)
    if fine_tune_schedule.enabled:
        data = env.reset(usr_conf)
        if data is None:
            logger.error("[FineTuneSchedule] initial env.reset failed after applying initial schedule config!")
            raise RuntimeError("FineTuneSchedule initial env.reset failed")
        obs, critic_obs = data
        if critic_obs is None:
            critic_obs = obs
        last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
        cur_reward_sum.zero_()
        cur_episode_length.zero_()
        logger.warning("[FineTuneSchedule] initial reward/command config applied by env.reset before rollout.")

    # Main Training Loop
    # 主训练循环
    while True:
        logger.info(f"Episode {episode} start, usr_conf is {usr_conf}")
        start_time = time.time()

        # Phase 1: Data Collection
        # 阶段1：数据收集
        last_obs, last_critic_obs, storage_stats = run_episodes_(
            env,
            agent,
            storage,
            logger,
            last_obs,
            last_critic_obs,
            episode,
            ep_infos,
            cur_reward_sum,
            cur_episode_length,
            rewbuffer,
            lenbuffer,
            usr_conf,
        )

        episode += 1

        # Phase 1.5: Velocity Curriculum Check (performance-based, before ep_infos.clear)
        # 阶段1.5：速度课程检查（性能驱动，必须在 ep_infos.clear() 之前调用）
        last_obs, last_critic_obs, vel_reset = vel_curriculum.check_and_update(
            ep_infos, usr_conf, env, last_obs, last_critic_obs, rollout_stats=storage_stats
        )
        # If env.reset was triggered by a stage change, stale accumulated rewards
        # from interrupted episodes must be discarded to prevent corrupting rewbuffer.
        # 若阶段切换触发了 env.reset，必须清零未完成 episode 的累计统计，
        # 防止旧值在下次 dones 触发时污染 rewbuffer。
        if vel_reset:
            cur_reward_sum.zero_()
            cur_episode_length.zero_()

        last_obs, last_critic_obs, schedule_reset = fine_tune_schedule.check_and_update(
            usr_conf, env, last_obs, last_critic_obs, episode
        )
        if schedule_reset:
            cur_reward_sum.zero_()
            cur_episode_length.zero_()

        # Phase 2: Policy Update
        # 阶段2：策略更新
        # framework=True lets the framework directly call back to the business layer,
        # skipping the sample data guard.
        # framework=True 让框架层直接回调业务层，跳过 sample data guard
        agent.learn(list_sample_data=None)
        # Reset buffer pointer for next data collection
        # 重置 buffer 指针，为下一轮数据收集做准备
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        # 阶段3：监控指标处理
        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats,
                                vel_stage=vel_curriculum.stage,
                                vel_tracking_ratio=vel_curriculum.last_tracking_ratio,
                                vel_tracking_reward=vel_curriculum.last_tracking_reward,
                                schedule_stats={
                                    **fine_tune_schedule.monitor_stats(),
                                    **vel_curriculum.monitor_stats(),
                                },
                                lenbuffer=lenbuffer, rewbuffer=rewbuffer)
            last_report_monitor_time = now

        ep_infos.clear()

        # Phase 4: Model Saving
        # 阶段4：模型保存
        if episode % agent.save_interval == 0:
            agent.save_model()

    env.close()


def _extract_metric_value(ep_info, key, device):
    """Extract and convert metric value to tensor.

    提取指标值并转换为 tensor。
    """
    if key not in ep_info:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    metric = ep_info[key]
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric, device=device)
    return metric.float().mean()


def _aggregate_metrics(generic_metrics):
    """Aggregate metrics by computing mean values.

    通过计算均值汇总指标。
    """
    aggregated = {}
    for metric_key, values in generic_metrics.items():
        if values:
            aggregated[metric_key] = torch.stack(values).mean().item()
        else:
            aggregated[metric_key] = 0.0
    return aggregated


def _collect_episode_metrics(ep_infos, reward_keys, device):
    """Collect metrics from episode infos.

    从 episode info 中收集指标。
    """
    generic_metrics = defaultdict(list)
    for ep_info in ep_infos:
        for key in reward_keys:
            metric_value = _extract_metric_value(ep_info, key, device)
            generic_metrics[key].append(metric_value)
    return _aggregate_metrics(generic_metrics)


def report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats=None,
                        vel_stage: int = 0, vel_tracking_ratio: float = 0.0,
                        vel_tracking_reward: float = 0.0, schedule_stats=None,
                        lenbuffer=None, rewbuffer=None):
    """
    Report monitoring data to monitor system.
    上报监控数据到监控系统。
    """
    monitor_data = {
        "episode_cnt": episode,
        "vel_curriculum_stage": vel_stage,
        "vel_curriculum_tracking_ratio": vel_tracking_ratio,
        "vel_curriculum_tracking_reward": vel_tracking_reward,
    }

    # Merge all storage stats: reward_mean/reward_std AND physics obs_ keys.
    # 将所有 storage_stats 合并写入，包含 reward 统计和物理量观测 obs_ 键。
    if storage_stats:
        monitor_data.update(storage_stats)
    if schedule_stats:
        monitor_data.update(schedule_stats)

    # Episode health metrics: episode length and cumulative reward per episode.
    # 训练进展指标：episode 存活步数和每 episode 累计奖励（与权重无关）。
    if lenbuffer:
        monitor_data["mean_episode_length"] = float(sum(lenbuffer) / len(lenbuffer))
    if rewbuffer:
        monitor_data["mean_episode_reward"] = float(sum(rewbuffer) / len(rewbuffer))

    if ep_infos:
        metrics = _collect_episode_metrics(ep_infos, reward_keys, agent.device)
        # Do not let episode-level missing keys overwrite workflow-level metrics.
        # monitor_builder exposes non-episode metrics such as vel_curriculum_* and
        # obs_*; _collect_episode_metrics returns 0 for keys absent from ep_info.
        # If blindly updated, those valid workflow/storage values become flat 0
        # on the dashboard.
        # 不允许 episode 聚合结果覆盖 workflow/storage 级指标。对于 ep_info 中不存在的
        # vel_curriculum_* / obs_*，_collect_episode_metrics 会返回 0；盲目 update
        # 会把真实上报值覆盖成 0。
        for key, value in metrics.items():
            if key not in monitor_data:
                monitor_data[key] = value
        monitor_data["episode_reward"] = sum(monitor_data.get(key, 0) for key in reward_keys)

    logger = getattr(agent, "logger", None)
    if logger is not None:
        logger.warning(
            "[MonitorDebug] reporting curriculum metrics: "
            f"episode={episode}, stage={monitor_data.get('vel_curriculum_stage')}, "
            f"tracking_ratio={monitor_data.get('vel_curriculum_tracking_ratio')}, "
            f"tracking_reward={monitor_data.get('vel_curriculum_tracking_reward')}, "
            f"has_obs_lin_vel_x_error={'obs_lin_vel_x_error' in monitor_data}, "
            f"has_obs_base_height={'obs_base_height' in monitor_data}"
        )
        logger.warning(
            "[LongTrainGate] "
            f"episode={episode}, "
            f"up={monitor_data.get('hs_up_ratio', 0.0):.4f}, "
            f"down={monitor_data.get('hs_down_ratio', 0.0):.4f}, "
            f"wall={monitor_data.get('hs_wall_ratio', 0.0):.4f}, "
            f"flat={monitor_data.get('hs_flat_ratio', 0.0):.4f}, "
            f"step_score={monitor_data.get('hs_step_score', 0.0):.4f}, "
            f"wall_score={monitor_data.get('hs_wall_score', 0.0):.4f}, "
            f"probe_available={monitor_data.get('g_probe_available', 0.0):.4f}, "
            f"probe_reason={monitor_data.get('g_probe_reason_code', 0.0):.0f}, "
            f"clear_active={monitor_data.get('hs_clear_active', 0.0):.4f}, "
            f"wall_active={monitor_data.get('hs_wall_active', 0.0):.4f}, "
            f"relax_ang_active={monitor_data.get('relax_ang_active', 0.0):.4f}, "
            f"relax_height_active={monitor_data.get('relax_height_active', 0.0):.4f}, "
            f"relax_joint_active={monitor_data.get('relax_joint_active', 0.0):.4f}"
        )

    monitor.put_data({os.getpid(): monitor_data})


def _process_env_step_result(data, episode, logger):
    """
    Process environment step result.
    处理环境交互结果。
    """
    if data is None:
        error_message = "step failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data

    if privileged_obs is not None:
        critic_obs = torch.clone(privileged_obs)
    else:
        critic_obs = torch.clone(obs)
    obs = torch.clone(obs)

    if obs is None:
        logger.error(f"episode {episode}, obs is None after processing!")
        raise Exception(f"episode {episode}, obs is None after processing!")

    dones = torch.logical_or(terminated, truncated)
    return frame_no, obs, critic_obs, rewards, dones, infos


def _move_tensors_to_device(obs, critic_obs, rewards, dones, device):
    """Move tensors to specified device.

    将张量移动到指定设备。
    """
    return (
        obs.to(device),
        critic_obs.to(device),
        rewards.to(device),
        dones.to(device),
    )


def _update_transition_data(
    transition,
    actions,
    values,
    actions_log_prob,
    action_mean,
    action_sigma,
    obs,
    critic_obs,
    rewards,
    dones,
    infos,
    agent,
):
    """
    Update transition with step data.
    使用步骤数据更新 transition。
    """
    transition.actions = actions
    transition.values = values
    transition.actions_log_prob = actions_log_prob
    transition.action_mean = action_mean
    transition.action_sigma = action_sigma
    transition.observations = obs
    transition.critic_observations = critic_obs
    transition.rewards = rewards.clone()
    transition.dones = dones

    # Bootstrapping on time outs
    # 处理 timeouts
    if "time_outs" in infos:
        transition.rewards += agent.algorithm.gamma * torch.squeeze(
            transition.values * infos["time_outs"].unsqueeze(1).to(agent.device), 1
        )


def _update_episode_statistics(
    dones,
    rewards,
    infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
    ep_infos,
):
    """Update episode statistics and buffers.

    更新 episode 统计和缓冲区。
    """
    if "episode" in infos:
        ep_infos.append(infos["episode"])

    cur_reward_sum += rewards
    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0


def _compute_advantages_and_returns(storage, agent, critic_obs, logger):
    """
    Compute advantage function and returns.
    计算优势函数和回报。
    """
    last_critic_obs = torch.clone(critic_obs)
    last_values = agent.algorithm.actor_critic.evaluate(last_critic_obs.detach()).detach()
    storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)

    storage_stats = {
        "reward_mean": storage.rewards.mean().item(),
        "reward_std": storage.rewards.std().item(),
    }

    return storage_stats


def _sample_rollout_tracking_stats(storage, usr_conf, logger=None):
    """Estimate velocity-tracking curriculum metrics from rollout critic obs.

    This avoids waiting for completed episodes.  With long stable episodes,
    ``infos["episode"]`` may be absent for many PPO updates, so an episode-only
    curriculum can remain stuck at zero even though the policy is already
    tracking commands well.

    critic_obs layout:
      [0:3] base_lin_vel, [9:12] velocity command.
    """
    critic_obs = getattr(storage, "privileged_observations", None)
    if critic_obs is None or storage.step <= 0 or critic_obs.shape[-1] < 12:
        return {}

    reward_conf = usr_conf.get("rewards", {}).get("track_lin_vel_xy", {})
    params = reward_conf.get("params", {})
    weight = abs(float(reward_conf.get("weight", 1.0)))
    std = float(params.get("std", 0.25))
    if weight <= 1e-6:
        weight = 1.0
    if std <= 1e-6:
        if logger is not None:
            logger.warning(
                "[VelocityCurriculumDebug] invalid track_lin_vel_xy std in TOML; "
                f"std={std}, falling back to 0.25"
            )
        std = 0.25

    rollout_critic_obs = critic_obs[:storage.step]
    actual_xy = rollout_critic_obs[..., 0:2]
    command_xy = rollout_critic_obs[..., 9:11]
    squared_error = torch.sum(torch.square(actual_xy - command_xy), dim=-1)
    tracking_ratio = torch.exp(-squared_error / (std * std)).mean().item()
    tracking_reward = tracking_ratio * weight

    return {
        "rollout_track_lin_vel_xy_ratio": tracking_ratio,
        "rollout_track_lin_vel_xy_reward": tracking_reward,
    }


def _get_isaac_env(env):
    """Try to unwrap the KaiwuDRL env wrapper to the underlying Isaac Lab env.

    尝试解包 KaiwuDRL 包装层，获取底层的 Isaac Lab 环境对象。
    Returns the first object that exposes a `command_manager` attribute,
    or None if none is found.
    返回第一个带有 command_manager 属性的对象，找不到则返回 None。
    """
    direct = _get_robot_gym_unwrapped(env)
    if direct is not None:
        return direct

    # Walk common wrapper chains recursively instead of assuming one fixed depth.
    # 递归遍历常见 wrapper 链，避免假设包装层只有固定一层。
    seen_ids = set()
    pending = [env]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        if hasattr(candidate, "command_manager") and hasattr(candidate, "scene"):
            return candidate
        for attr_name in (
            "env_object",
            "_env_object",
            "env",
            "_env",
            "gym_env",
            "_gym_env",
            "envs",
            "_envs",
            "unwrapped",
            "wrapped_env",
            "_wrapped_env",
            "venv",
            "_venv",
            "isaac_env",
            "_isaac_env",
            "sim_env",
            "_sim_env",
            "base_env",
            "_base_env",
            "robot_env",
            "_robot_env",
            "rl_env",
            "_rl_env",
            "gym",
            "_gym",
            "task",
            "_task",
        ):
            try:
                child = getattr(candidate, attr_name, None)
            except Exception:
                child = None
            if child is not None:
                pending.append(child)
        if hasattr(candidate, "__dict__"):
            try:
                for key, child in vars(candidate).items():
                    if key.startswith("__"):
                        continue
                    if key in ("command_manager", "scene"):
                        continue
                    if child is not None and not isinstance(child, (str, bytes, int, float, bool, tuple, list, dict)):
                        pending.append(child)
            except Exception:
                pass
    return None


def _get_robot_gym_unwrapped(env):
    """Fast path for tools.base_env.base_env.Robot: env._gym_env.unwrapped."""
    try:
        gym_env = getattr(env, "_gym_env", None)
        unwrapped = getattr(gym_env, "unwrapped", None) if gym_env is not None else None
        if unwrapped is not None and hasattr(unwrapped, "command_manager") and hasattr(unwrapped, "scene"):
            return unwrapped
    except Exception:
        return None
    return None


def _describe_env_unwrap_candidates(env, max_items: int = 16) -> str:
    parts = []
    seen_ids = set()
    pending = [("root", env)]
    while pending and len(parts) < max_items:
        path, candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        try:
            keys = list(vars(candidate).keys())[:12] if hasattr(candidate, "__dict__") else []
        except Exception:
            keys = []
        parts.append(f"{path}:{type(candidate).__module__}.{type(candidate).__name__} keys={keys}")
        for attr_name in ("env_object", "_env_object", "_gym_env", "env", "_env", "unwrapped", "venv", "_venv"):
            try:
                child = getattr(candidate, attr_name, None)
            except Exception:
                child = None
            if child is not None:
                pending.append((f"{path}.{attr_name}", child))
    return " | ".join(parts)


def _has_robot_state(asset) -> bool:
    data = getattr(asset, "data", None)
    return (
        data is not None
        and hasattr(data, "root_lin_vel_b")
        and hasattr(data, "root_pos_w")
        and hasattr(data, "root_ang_vel_b")
    )


def _get_robot_asset_from_env(isaac_env):
    """Best-effort robot asset lookup across common Isaac Lab scene layouts.

    奖励模块通过平台基类间接取 robot asset，这里无法复用闭源 helper，
    因此改为遍历常见 scene 容器布局做稳健查找。
    """
    if isaac_env is None:
        return None

    scene = getattr(isaac_env, "scene", None)
    if scene is None:
        return None

    if hasattr(scene, "__getitem__"):
        scene_keys = []
        keys_fn = getattr(scene, "keys", None)
        if callable(keys_fn):
            try:
                scene_keys = list(keys_fn())
            except Exception:
                scene_keys = []
        for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2", *scene_keys):
            try:
                asset = scene[key]
            except Exception:
                asset = None
            if _has_robot_state(asset):
                return asset

    for container_name in (
        "articulations",
        "_articulations",
        "rigid_objects",
        "_rigid_objects",
        "entities",
        "_entities",
    ):
        container = getattr(scene, container_name, None)
        if container is None:
            continue

        if hasattr(container, "get"):
            for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2"):
                asset = container.get(key)
                if _has_robot_state(asset):
                    return asset

        values = getattr(container, "values", None)
        if callable(values):
            for asset in values():
                if _has_robot_state(asset):
                    return asset

    for attr_name in ("robot", "_robot"):
        asset = getattr(isaac_env, attr_name, None)
        if _has_robot_state(asset):
            return asset

    return None


def _sample_physics_stats_from_critic_obs(critic_obs):
    """Fallback physics metrics from critic observation layout.

    critic_obs layout is:
      [0:3] base_lin_vel, [3:6] base_ang_vel, [9:12] velocity command.

    This path does not provide base height because height is not part of the
    documented critic observation.  It still keeps velocity and attitude panels
    alive when the wrapped Isaac env cannot be reached from the workflow.
    """
    if critic_obs is None or not hasattr(critic_obs, "shape") or critic_obs.shape[-1] < 12:
        return {}

    actual_vx = critic_obs[:, 0]
    actual_vy = critic_obs[:, 1]
    cmd_vx = critic_obs[:, 9]
    cmd_vy = critic_obs[:, 10]

    return {
        "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
        "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
        "obs_actual_vel_x": actual_vx.mean().item(),
        "obs_ang_vel_xy": torch.norm(critic_obs[:, 3:5], dim=1).mean().item(),
    }


def _sample_physics_stats(env, logger=None, critic_obs=None):
    """Take a point-in-time snapshot of key physical quantities across all envs.

    在所有并行环境上对关键物理量做一次快照（均值）。
    这些指标与 reward 权重无关，是判断策略真实收敛情况的第一手依据：
      obs_lin_vel_x_error — 前向速度追踪误差 |cmd_vx - actual_vx| (m/s)
      obs_lin_vel_y_error — 侧向速度追踪误差 |cmd_vy - actual_vy| (m/s)
      obs_actual_vel_x    — 机体前向实际速度均值 (m/s)
      obs_base_height     — 机身高度均值 (m)，目标 0.38 m
      obs_ang_vel_xy      — pitch/roll 角速度幅值均值 (rad/s)

    Falls back to critic_obs for metrics that are available there if the
    underlying Isaac Lab env is not accessible.
    如果访问不到底层 Isaac Lab 环境，则从 critic_obs 中兜底计算可用指标。
    """
    try:
        isaac_env = _get_isaac_env(env)
        if isaac_env is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to unwrap Isaac Lab env; "
                    "falling back to critic_obs for partial physics metrics. "
                    f"candidates={_describe_env_unwrap_candidates(env, max_items=8)}"
                )
            return _sample_physics_stats_from_critic_obs(critic_obs)

        cmd = isaac_env.command_manager.get_command("base_velocity")  # (N, 3)
        asset = _get_robot_asset_from_env(isaac_env)
        if asset is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to locate robot asset in scene; "
                    "falling back to critic_obs for partial physics metrics."
                )
            return _sample_physics_stats_from_critic_obs(critic_obs)

        actual_vx = asset.data.root_lin_vel_b[:, 0]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        cmd_vx    = cmd[:, 0]
        cmd_vy    = cmd[:, 1]

        return {
            "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
            "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
            "obs_actual_vel_x":    actual_vx.mean().item(),
            "obs_base_height":     asset.data.root_pos_w[:, 2].mean().item(),
            "obs_ang_vel_xy":      torch.norm(
                asset.data.root_ang_vel_b[:, :2], dim=1).mean().item(),
        }
    except Exception as exc:
        if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
            env._physics_stats_error_logged = True
            logger.warning(
                f"[PhysicsStats] Failed to sample physics metrics from Isaac env: {exc}; "
                "falling back to critic_obs for partial physics metrics."
            )
        return _sample_physics_stats_from_critic_obs(critic_obs)


def run_episodes_(
    env,
    agent,
    storage,
    logger,
    last_obs,
    last_critic_obs,
    episode,
    ep_infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
    usr_conf,
):
    """
    Run episodes to collect trajectory data.
    运行 episodes 收集轨迹数据。

    Returns:
        tuple: (last_obs, last_critic_obs, storage_stats)
        返回值：(last_obs, last_critic_obs, storage_stats)
    """
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs

    # TODO: for hierarchical training, handle the mismatch between env action and
    # PPO storage action on your own.
    # TODO：如需分层训练，自行处理 env action 与 PPO storage action 不一致的问题。

    # Policy execution loop
    # 策略执行循环
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            # Predict actions
            # 预测动作
            predict_data = (obs, critic_obs)
            predict_result = agent.predict(predict_data)

            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
            ) = predict_result
            joint_actions = actions

            # Clip joint actions for env
            # 裁剪关节动作
            command_actions = torch.clip(joint_actions, -6.0, 6.0).to(agent.device)
            if i == 0:
                logger.info(f"clipped_action:{command_actions}")

            # Environment interaction
            # 环境交互
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)

            # Move tensors to device
            # 将张量移动到设备
            obs, critic_obs, rewards, dones = _move_tensors_to_device(obs, critic_obs, rewards, dones, agent.device)

            # Update episode statistics (always, regardless of decimation)
            # 更新 episode 统计（始终执行，不受降频影响）
            _update_episode_statistics(
                dones,
                rewards,
                infos,
                cur_reward_sum,
                cur_episode_length,
                rewbuffer,
                lenbuffer,
                ep_infos,
            )

            # Write transition to storage every step (flat PPO)
            # 每步写入 storage（扁平 PPO）
            _update_transition_data(
                transition,
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
                rewards,
                dones,
                infos,
                agent,
            )
            storage.add_transitions(transition)
            transition.clear()

        # Compute advantages and returns
        # 计算优势函数和回报
        storage_stats = _compute_advantages_and_returns(storage, agent, critic_obs, logger)
        storage_stats.update(_sample_rollout_tracking_stats(storage, usr_conf, logger))
        last_obs = torch.clone(obs)

    # Note: batch generation now handled by AlgorithmPPO.learn()
    # Storage will be cleared after learning
    # 注：batch 生成已由 AlgorithmPPO.learn() 处理，
    # storage 将在训练完成后被清空。

    # Append a physics snapshot (averaged across all envs).
    # Wrapped in try/except inside _sample_physics_stats, so always safe.
    # 追加物理量快照（跨所有并行环境取均值）。
    # _sample_physics_stats 内部已有 try/except，调用总是安全的。
    storage_stats.update(_sample_physics_stats(env, logger, critic_obs=critic_obs))
    storage_stats.update(_sample_stair_gate_debug_stats(env))
    storage_stats.update(_sample_rollout_height_scan_probe_stats(critic_obs, logger))

    return last_obs, critic_obs, storage_stats


def _sample_rollout_height_scan_probe_stats(critic_obs, logger=None):
    """Compute stair-gate probe metrics directly from rollout critic observations.

    Reward-side debug attributes can be hidden behind Kaiwu/Isaac wrappers or
    process boundaries.  The critic observation is already available in the PPO
    workflow, so these metrics are used as the reliable monitor source for
    height-scan gate validation.  Critic layout is:

        [critic_proprio(60) | height_scan(256)]

    The height scan is a scaled distance-to-ground signal.  For local geometry
    deltas we convert it to a ground-height proxy by negating and unscaling it,
    so positive front-minus-near deltas mean higher terrain ahead.
    """
    zero_stats = _empty_height_scan_probe_stats()
    try:
        if critic_obs is None or not isinstance(critic_obs, torch.Tensor):
            zero_stats["g_probe_reason_code"] = 91.0
            return zero_stats
        if critic_obs.dim() != 2 or critic_obs.shape[1] < 316:
            zero_stats["g_probe_reason_code"] = 92.0
            return zero_stats

        height_scan = critic_obs[:, 60:316].detach().float()
        if height_scan.shape[1] < 256:
            zero_stats["g_probe_reason_code"] = 92.0
            return zero_stats

        # Observation height scan uses distance-to-ground.  Local ground-height
        # proxy is enough for step detection because only deltas are used.
        grid = (-height_scan[:, :256] / 2.5).view(height_scan.shape[0], 16, 16)
        grid = torch.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

        threshold_sets = {
            "loose": (0.010, 0.280, 0.320, 0.002, 0.250),
            "mild": (0.015, 0.260, 0.300, 0.005, 0.200),
            "current": (0.025, 0.200, 0.240, 0.020, 0.080),
            "strict": (0.030, 0.180, 0.220, 0.030, 0.060),
        }
        methods = {
            "mean_x": (0.0, grid, "mean"),
            "mean_y": (1.0, grid.transpose(1, 2), "mean"),
            "edge_x": (2.0, grid, "edge"),
            "edge_y": (3.0, grid.transpose(1, 2), "edge"),
        }

        method_outputs = {
            name: _height_scan_probe_method_from_grid(
                method_grid,
                mode=mode,
                thresholds=threshold_sets["mild"],
            )
            for name, (_method_id, method_grid, mode) in methods.items()
        }
        stacked_scores = torch.stack(
            [method_outputs[name]["step_score"] for name in methods],
            dim=1,
        )
        best_indices = torch.argmax(stacked_scores, dim=1).float()
        best_score = torch.amax(stacked_scores, dim=1)

        stats = dict(zero_stats)
        for method_name, output in method_outputs.items():
            stats[f"gx_{method_name}"] = _tensor_mean(output["step_score"])
        stats["g_best_axis"] = _tensor_mean(best_indices)
        stats["g_best_score"] = _tensor_mean(best_score)

        threshold_outputs = {}
        for name, thresholds in threshold_sets.items():
            outputs = [
                _height_scan_probe_method_from_grid(
                    method_grid,
                    mode=mode,
                    thresholds=thresholds,
                )
                for _method_id, method_grid, mode in methods.values()
            ]
            threshold_outputs[name] = {
                "up": torch.stack([out["up"].float() for out in outputs], dim=1).amax(dim=1),
                "down": torch.stack([out["down"].float() for out in outputs], dim=1).amax(dim=1),
                "wall": torch.stack([out["wall"].float() for out in outputs], dim=1).amax(dim=1),
                "step_delta": torch.stack([out["step_delta"] for out in outputs], dim=1).mean(dim=1),
                "step_score": torch.stack([out["step_score"] for out in outputs], dim=1).amax(dim=1),
                "wall_score": torch.stack([out["wall_score"] for out in outputs], dim=1).amax(dim=1),
                "up_evidence": torch.stack([out["up_evidence"] for out in outputs], dim=1).amax(dim=1),
                "down_evidence": torch.stack([out["down_evidence"] for out in outputs], dim=1).amax(dim=1),
            }

        current_probe = threshold_outputs["current"]
        mild_probe = threshold_outputs["mild"]
        stats.update(
            {
                "g_probe_available": 1.0,
                "g_probe_reason_code": 10.0,
                "hs_up_ratio": _tensor_ratio(current_probe["up"]),
                "hs_down_ratio": _tensor_ratio(current_probe["down"]),
                "hs_wall_ratio": _tensor_ratio(current_probe["wall"]),
                "hs_flat_ratio": _tensor_ratio(
                    ~(current_probe["up"].bool() | current_probe["down"].bool() | current_probe["wall"].bool())
                ),
                "hs_step_delta": _tensor_mean(current_probe["step_delta"]),
                "hs_step_score": _tensor_mean(current_probe["step_score"]),
                "hs_wall_score": _tensor_mean(current_probe["wall_score"]),
                "g_step_delta": _tensor_mean(mild_probe["step_delta"]),
                "g_edge_pos": _tensor_mean(mild_probe["up_evidence"]),
                "g_edge_neg": _tensor_mean(mild_probe["down_evidence"]),
                "g_wall_score": _tensor_mean(mild_probe["wall_score"]),
            }
        )
        for name, output in threshold_outputs.items():
            stats[f"g_{name}_up"] = _tensor_ratio(output["up"])
            stats[f"g_{name}_down"] = _tensor_ratio(output["down"])
            stats[f"g_{name}_wall"] = _tensor_ratio(output["wall"])
        return stats
    except Exception as exc:
        zero_stats["g_probe_reason_code"] = 93.0
        if logger is not None:
            logger.warning(f"[HeightScanRolloutProbe] failed: {exc}")
        return zero_stats


def _empty_height_scan_probe_stats():
    keys = (
        "g_probe_available",
        "g_probe_reason_code",
        "hs_up_ratio",
        "hs_down_ratio",
        "hs_wall_ratio",
        "hs_flat_ratio",
        "hs_step_delta",
        "hs_step_score",
        "hs_wall_score",
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
    stats = {key: 0.0 for key in keys}
    stats["g_probe_reason_code"] = 90.0
    return stats


def _height_scan_probe_method_from_grid(
    grid,
    *,
    mode: str,
    thresholds,
    body_y_start: int = 5,
    body_y_end: int = 11,
    near_x_start: int = 0,
    near_x_end: int = 4,
    front_x_start: int = 4,
    front_x_end: int = 10,
):
    min_h, max_h, wall_h, min_score, max_wall = thresholds
    num_envs = grid.shape[0]
    zeros = torch.zeros(num_envs, device=grid.device, dtype=grid.dtype)
    y0 = max(0, min(int(body_y_start), 15))
    y1 = max(y0 + 1, min(int(body_y_end), 16))
    x0 = max(0, min(int(near_x_start), 15))
    x_near = max(x0 + 1, min(int(near_x_end), 16))
    x_front0 = max(0, min(int(front_x_start), 15))
    x_front1 = max(x_front0 + 1, min(int(front_x_end), 16))
    x1 = max(x0 + 2, min(int(front_x_end), 16))

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


def _tensor_mean(value):
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean().item())


def _tensor_ratio(value):
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean().item())


def _sample_stair_gate_debug_stats(env):
    stats = {}
    try:
        for source in _iter_stair_gate_debug_sources(env):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                try:
                    stats[key] = float(value)
                except (TypeError, ValueError):
                    pass
    except Exception:
        return stats

    has_probe_debug = "g_probe_available" in stats

    # Stable short aliases for monitor panels. Keep the raw keys too.
    aliases = {
        "hs_up_ratio": "height_scan_up_step_mean",
        "hs_down_ratio": "height_scan_down_step_mean",
        "hs_wall_ratio": "height_scan_wall_mean",
        "hs_flat_ratio": "height_scan_flat_mean",
        "hs_step_delta": "height_scan_step_delta_mean",
        "hs_step_score": "height_scan_step_score_mean",
        "hs_wall_score": "height_scan_wall_score_mean",
        "hs_clear_active": "height_scan_feet_clearance_active_ratio",
        "hs_wall_active": "height_scan_wall_reject_active_ratio",
        "relax_ang_active": "stair_relax_ang_vel_xy_active_ratio",
        "relax_height_active": "stair_relax_base_height_active_ratio",
        "relax_joint_active": "stair_relax_joint_position_active_ratio",
        "relax_ang_value": "stair_relax_ang_vel_xy_reward_mean",
        "relax_height_value": "stair_relax_base_height_reward_mean",
        "relax_joint_value": "stair_relax_joint_position_reward_mean",
        "g_probe_available": "g_probe_available",
        "g_probe_reason_code": "g_probe_reason_code",
    }
    for alias, source_key in aliases.items():
        stats[alias] = float(stats.get(source_key, 0.0))
    if not has_probe_debug:
        stats["g_probe_available"] = 0.0
        stats["g_probe_reason_code"] = 98.0
    step_active = max(float(stats.get("hs_up_ratio", 0.0)), float(stats.get("hs_down_ratio", 0.0)))
    for alias in ("relax_ang_active", "relax_height_active", "relax_joint_active"):
        if alias not in stats or stats[alias] == 0.0:
            stats[alias] = step_active
    return stats


def _iter_stair_gate_debug_sources(env, max_items: int = 64):
    """Yield every discovered _stair_gate_debug dict in wrapper/object chains."""
    seen_ids = set()
    pending = [env]
    attr_names = (
        "env_object",
        "_env_object",
        "env",
        "_env",
        "gym_env",
        "_gym_env",
        "envs",
        "_envs",
        "unwrapped",
        "wrapped_env",
        "_wrapped_env",
        "venv",
        "_venv",
        "isaac_env",
        "_isaac_env",
        "sim_env",
        "_sim_env",
        "base_env",
        "_base_env",
        "robot_env",
        "_robot_env",
        "rl_env",
        "_rl_env",
        "gym",
        "_gym",
        "task",
        "_task",
    )
    while pending and len(seen_ids) < max_items:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        source = getattr(candidate, "_stair_gate_debug", None)
        if isinstance(source, dict):
            yield source
        for attr_name in attr_names:
            try:
                child = getattr(candidate, attr_name, None)
            except Exception:
                child = None
            if child is not None:
                pending.append(child)
        if hasattr(candidate, "__dict__"):
            try:
                for key, child in vars(candidate).items():
                    if key.startswith("__"):
                        continue
                    if child is not None and not isinstance(child, (str, bytes, int, float, bool, tuple, list, dict)):
                        pending.append(child)
            except Exception:
                pass
