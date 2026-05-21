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


def _expr_panel(builder, name, name_en, expr):
    return (
        builder.add_panel(name=name, name_en=name_en, type="line")
        .add_metric(metrics_name=name_en, expr=expr)
        .end_panel()
    )


def _multi_expr_panel(builder, name, name_en, metrics):
    panel = builder.add_panel(name=name, name_en=name_en, type="line")
    for metric_name, expr in metrics:
        panel = panel.add_metric(metrics_name=metric_name, expr=expr)
    return panel.end_panel()


def _level_non_timeout_metrics(terrain_name):
    return [
        (
            f"l{level}",
            f"avg(abnormal_count_{terrain_name}_l{level}{{}}) - "
            f"avg(timeout_count_{terrain_name}_l{level}{{}})",
        )
        for level in range(10)
    ]


def build_monitor():
    """Build a compact monitor.

    Keep panels that produced useful data in the latest run.  Fragile debug
    aliases that stayed flat zero are replaced by reward-term panels or removed.
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
        ("线速度跟踪权重", "w_track_lin"),
        ("指令方向进展权重", "w_cmd_prog"),
        ("指令方向偏差权重", "w_cmd_dir_dev"),
        ("指令路程进展权重", "w_cmd_path"),
        ("长时停滞惩罚权重", "w_cmd_stall"),
        ("偏航跟踪权重", "w_track_yaw"),
        ("密集抬脚固定权重", "w_feet_clear"),
        ("台阶抬脚固定权重", "w_hs_clear"),
        ("落台阶固定权重", "w_stair_place"),
        ("抬脚过高惩罚权重", "w_over_clear"),
        ("台阶机身离地权重", "w_base_clear"),
        ("台阶边沿对齐权重", "w_edge_align"),
        ("下台阶速度安全权重", "w_down_speed"),
        ("下台阶落脚安全权重", "w_down_touch"),
        ("无指令转向固定权重", "w_no_yaw"),
        ("朝向漂移固定权重", "w_heading_drift"),
        ("pitch roll 角速度权重", "w_ang_xy"),
        ("原地转向惩罚权重", "w_pivot"),
        ("地形调度阶段", "scheduled_terrain_phase"),
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
        ("指令方向进展奖励", "reward_command_direction_progress"),
        ("指令方向偏差惩罚", "reward_command_direction_deviation"),
        ("指令路程进展奖励", "reward_command_path_progress"),
        ("长时停滞惩罚", "reward_commanded_stall_penalty"),
        ("密集抬脚奖励", "reward_feet_clearance"),
        ("密集抬脚激活率", "feet_clear_active"),
        ("密集抬脚高度", "feet_clear_height"),
        ("台阶抬脚高度奖励", "reward_height_scan_feet_clearance"),
        ("前伸落台阶奖励", "reward_stair_forward_foot_placement"),
        ("抬脚过高惩罚", "reward_stair_over_clearance_penalty"),
        ("台阶机身离地惩罚", "reward_stair_base_clearance_penalty"),
        ("台阶边沿法线惩罚", "reward_stair_edge_normal_alignment"),
        ("下台阶速度安全惩罚", "reward_down_stair_speed_safety"),
        ("下台阶落脚安全惩罚", "reward_down_stair_touchdown_safety"),
        ("无指令转向惩罚", "reward_uncommanded_yaw_rate"),
        ("无指令朝向漂移惩罚", "reward_uncommanded_heading_drift"),
        ("无指令转向激活率", "uncommanded_yaw_active"),
        ("台阶转向惩罚激活率", "uncommanded_yaw_stair"),
        ("抬脚门控激活率", "hs_clear_active"),
        ("落台阶门控激活率", "hs_place_active"),
        ("路径奖励 cap 因子", "cmd_path_cap"),
        ("停滞惩罚激活率", "cmd_stall_active"),
        ("过高抬脚激活率", "over_clear_active"),
        ("机身离地惩罚激活率", "base_clear_active"),
        ("机身离地高度", "base_clearance"),
        ("机身离地缺口", "base_clear_deficit"),
        ("边沿对齐惩罚激活率", "edge_align_active"),
        ("边沿法线余弦", "edge_align_cos"),
        ("边沿侧向比例", "edge_lateral_ratio"),
        ("下台阶速度惩罚激活率", "down_speed_active"),
        ("下台阶落脚惩罚激活率", "down_touch_active"),
        ("关节速度惩罚", "reward_dof_vel"),
        ("步态对称性惩罚", "reward_air_time_variance_penalty"),
        ("原地旋转惩罚", "reward_pivot_turning"),
        ("脚部打滑惩罚", "reward_feet_slide"),
        ("脚撞台阶边缘惩罚", "reward_feet_stumble"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    config = config.add_group(group_name="台阶门控验证", group_name_en="stair_gate")
    for name, key in (
        ("离散台阶非缓坡门控", "g_stair_noslope_m"),
        ("上台阶方向门控", "g_dom_up_02"),
        ("下台阶方向门控", "g_dom_down_02"),
        ("奖励台阶门控", "hs_reward_stair_gate"),
        ("奖励上台阶门控", "hs_reward_up_step"),
        ("奖励下台阶门控", "hs_reward_down_step"),
        ("上下方向歧义率", "g_dom_both_02"),
        ("台阶边缘高度", "g_step_edge_sharpness"),
        ("台阶边缘局部性", "g_step_edge_locality"),
        ("缓坡平滑度", "g_slope_smoothness"),
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

    config = config.add_group(group_name="非超时失败", group_name_en="non_timeout_fail")
    config = _expr_panel(
        config,
        "总非超时失败",
        "ntfail_total",
        "avg(abnormal_count{}) - avg(timeout_count{})",
    )
    for panel_name, panel_en, terrain_name in (
        ("缓坡非超时失败", "ntfail_slope", "pyramid_slope"),
        ("反缓坡非超时失败", "ntfail_slope_inv", "pyramid_slope_inv"),
        ("台阶非超时失败", "ntfail_stairs", "pyramid_stairs"),
        ("反台阶非超时失败", "ntfail_stairs_inv", "pyramid_stairs_inv"),
    ):
        config = _multi_expr_panel(
            config,
            panel_name,
            panel_en,
            _level_non_timeout_metrics(terrain_name),
        )
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
        ("pitch roll 角速度", "obs_ang_vel_xy"),
    ):
        config = _panel(config, name, key)
    config = config.end_group()

    return config.build()
