#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Custom monitor panels for the active PPO stair-bridge run."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _panel(builder, name, name_en, metric=None):
    metric = metric or name_en
    return (
        builder.add_panel(name=name, name_en=name_en, type="line")
        .add_metric(metrics_name=name_en, expr=f"avg({metric}{{}})")
        .end_panel()
    )


def build_monitor():
    """Build a compact monitor.

    The earlier height-scan threshold experiments emitted many ``g_*`` probe
    panels.  They are intentionally removed here; this run only keeps actual
    training signals plus the new command_mix verification metrics.
    """
    monitor = MonitorConfigBuilder()

    config = monitor.title("四足机器人导航")

    config = config.add_group(group_name="训练进展", group_name_en="training_progress")
    config = _panel(config, "平均 episode 步数", "mean_episode_length")
    config = _panel(config, "每 episode 累计奖励", "mean_episode_reward")
    config = _panel(config, "综合训练奖励均值", "reward_mean")
    config = config.end_group()

    config = config.add_group(group_name="算法指标", group_name_en="algorithm")
    for name, key in (
        ("总损失", "total_loss"),
        ("价值损失", "value_loss"),
        ("策略损失", "policy_loss"),
        ("熵损失", "entropy_loss"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="速度跟踪", group_name_en="velocity_tracking")
    for name, key in (
        ("线速度跟踪", "reward_track_lin_vel_xy"),
        ("偏航角速度跟踪", "reward_track_ang_vel_z"),
        ("课程追踪比例", "vel_curriculum_tracking_ratio"),
        ("速度课程等级", "vel_curriculum_stage"),
        ("线性调度进度", "scheduled_alpha"),
        ("调度已运行秒数", "scheduled_elapsed_seconds"),
        ("侧向指令上限", "scheduled_cmd_vy_abs_max"),
        ("偏航指令上限", "scheduled_cmd_wz_abs_max"),
        ("实际侧向指令上限", "vel_cmd_vy_abs_max"),
        ("实际偏航指令上限", "vel_cmd_wz_abs_max"),
        ("线速度跟踪权重", "sched_track_lin_w"),
        ("偏航跟踪权重", "sched_track_yaw_w"),
        ("原地转向惩罚权重", "sched_pivot_w"),
        ("地形调度阶段", "scheduled_terrain_phase"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="命令混合验证", group_name_en="command_mix")
    for name, key in (
        ("命令混合启用", "command_mix_enabled"),
        ("运行调试可见", "command_mix_runtime_seen"),
        ("运行原因码", "command_mix_reason_code"),
        ("旋转模式比例", "command_mix_spin_ratio"),
        ("仅前进模式比例", "command_mix_vx_only_ratio"),
        ("平移无偏航比例", "command_mix_vx_vy_ratio"),
        ("全命令模式比例", "command_mix_full_ratio"),
        ("目标旋转比例", "command_mix_target_spin"),
        ("目标仅前进比例", "command_mix_target_vx_only"),
        ("目标平移无偏航", "command_mix_target_vx_vy"),
        ("目标全命令比例", "command_mix_target_full"),
        ("仅前进命令实测", "command_mix_only_vx_like"),
        ("旋转命令实测", "command_mix_spin_like"),
        ("侧向指令均值", "command_mix_cmd_vy_abs_mean"),
        ("偏航指令均值", "command_mix_cmd_wz_abs_mean"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="姿态质量", group_name_en="posture_quality")
    for name, key in (
        ("机身水平姿态惩罚", "reward_flat_orientation"),
        ("机身高度惩罚", "reward_correct_base_height"),
        ("全身关节偏离惩罚", "reward_joint_position_penalty"),
        ("侧向漂移惩罚", "reward_base_lateral_vel"),
        ("pitch roll 角速度惩罚", "reward_ang_vel_xy"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="步态质量", group_name_en="gait_quality")
    for name, key in (
        ("脚部滞空时间奖励", "reward_feet_air_time"),
        ("关节速度惩罚", "reward_dof_vel"),
        ("步态对称性惩罚", "reward_air_time_variance_penalty"),
        ("原地旋转惩罚", "reward_pivot_turning"),
        ("脚部打滑惩罚", "reward_feet_slide"),
        ("脚撞台阶边缘惩罚", "reward_feet_stumble"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="稳定接触", group_name_en="stability_contact")
    for name, key in (
        ("垂直速度惩罚", "reward_lin_vel_z"),
        ("非预期接触惩罚", "reward_undesired_contacts"),
        ("终止惩罚", "reward_termination"),
        ("关节位置极限惩罚", "reward_dof_pos_limits"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="关节动作平滑", group_name_en="joint_action_smoothness")
    for name, key in (
        ("关节加速度惩罚", "reward_joint_acc"),
        ("动作变化率惩罚 一阶", "reward_action_rate"),
        ("动作平滑惩罚 二阶", "reward_action_smoothness"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="能耗扭矩", group_name_en="energy_torque")
    config = _panel(config, "能耗惩罚", "reward_energy")
    config = _panel(config, "关节扭矩惩罚", "reward_joint_torques")
    config = config.end_group()

    config = config.add_group(group_name="物理观测量", group_name_en="physics_obs")
    for name, key in (
        ("前向速度追踪误差", "obs_lin_vel_x_error"),
        ("侧向速度追踪误差", "obs_lin_vel_y_error"),
        ("实际前向速度", "obs_actual_vel_x"),
        ("机身高度", "obs_base_height"),
        ("pitch roll 角速度", "obs_ang_vel_xy"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    return config.build()
