#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    #
    # Panel organization (Phase-0: flat-ground gait shaping):
    # 面板组织（Phase-0 平地步态塑形）：
    #
    # Group 1: 算法指标   — PPO loss curves
    # Group 2: 速度跟踪   — velocity tracking rewards (primary positive signal)
    # Group 3: 姿态质量   — posture rewards (Phase-0 primary training objective)
    # Group 4: 步态质量   — gait quality rewards (Phase-0 secondary objective)
    # Group 5: 稳定/接触  — stability & contact penalties
    # Group 6: 关节/动作  — joint & action smoothness penalties
    # Group 7: 能耗       — energy / torque penalties (competition scoring)

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 1: PPO algorithm loss curves
        # Group 1: PPO 算法损失曲线
        # ==============================================================
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="总损失", name_en="total_loss", type="line")
            .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
            .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
            .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
            .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
            .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
            .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
            .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 2: Velocity tracking (primary positive reward signal)
        # Group 2: 速度跟踪（主正向奖励信号）
        # ==============================================================
        .add_group(group_name="速度跟踪", group_name_en="velocity_tracking")
        .add_panel(name="线速度跟踪", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy",
                        expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .add_panel(name="偏航角速度跟踪", name_en="reward_track_ang_vel_z", type="line")
            .add_metric(metrics_name="reward_track_ang_vel_z",
                        expr="avg(reward_track_ang_vel_z{})")
            .end_panel()
        .add_panel(name="速度课程等级", name_en="vel_curriculum_stage", type="line")
            .add_metric(metrics_name="vel_curriculum_stage",
                        expr="avg(vel_curriculum_stage{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 3: Posture quality (Phase-0 PRIMARY objective)
        # Group 3: 姿态质量（Phase-0 核心训练目标）
        # ==============================================================
        .add_group(group_name="姿态质量", group_name_en="posture_quality")
        .add_panel(name="机身水平姿态惩罚", name_en="reward_flat_orientation", type="line")
            .add_metric(metrics_name="reward_flat_orientation",
                        expr="avg(reward_flat_orientation{})")
            .end_panel()
        .add_panel(name="机身高度惩罚", name_en="reward_correct_base_height", type="line")
            .add_metric(metrics_name="reward_correct_base_height",
                        expr="avg(reward_correct_base_height{})")
            .end_panel()
        .add_panel(name="全身关节偏离惩罚", name_en="reward_joint_position_penalty", type="line")
            .add_metric(metrics_name="reward_joint_position_penalty",
                        expr="avg(reward_joint_position_penalty{})")
            .end_panel()
        .add_panel(name="侧向漂移惩罚", name_en="reward_base_lateral_vel", type="line")
            .add_metric(metrics_name="reward_base_lateral_vel",
                        expr="avg(reward_base_lateral_vel{})")
            .end_panel()
        .add_panel(name="pitch roll 角速度惩罚", name_en="reward_ang_vel_xy", type="line")
            .add_metric(metrics_name="reward_ang_vel_xy",
                        expr="avg(reward_ang_vel_xy{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 4: Gait quality (Phase-0 secondary objective)
        # Group 4: 步态质量（Phase-0 次要训练目标）
        # ==============================================================
        .add_group(group_name="步态质量", group_name_en="gait_quality")
        .add_panel(name="脚部滞空时间奖励", name_en="reward_feet_air_time", type="line")
            .add_metric(metrics_name="reward_feet_air_time",
                        expr="avg(reward_feet_air_time{})")
            .end_panel()
        .add_panel(name="关节速度惩罚", name_en="reward_dof_vel", type="line")
            .add_metric(metrics_name="reward_dof_vel",
                        expr="avg(reward_dof_vel{})")
            .end_panel()
        .add_panel(name="步态对称性惩罚", name_en="reward_air_time_variance_penalty", type="line")
            .add_metric(metrics_name="reward_air_time_variance_penalty",
                        expr="avg(reward_air_time_variance_penalty{})")
            .end_panel()
        .add_panel(name="原地旋转惩罚", name_en="reward_pivot_turning", type="line")
            .add_metric(metrics_name="reward_pivot_turning",
                        expr="avg(reward_pivot_turning{})")
            .end_panel()
        .add_panel(name="脚部打滑惩罚", name_en="reward_feet_slide", type="line")
            .add_metric(metrics_name="reward_feet_slide",
                        expr="avg(reward_feet_slide{})")
            .end_panel()
        .add_panel(name="脚撞台阶边缘惩罚", name_en="reward_feet_stumble", type="line")
            .add_metric(metrics_name="reward_feet_stumble",
                        expr="avg(reward_feet_stumble{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 5: Stability & contact penalties
        # Group 5: 稳定性与接触惩罚
        # ==============================================================
        .add_group(group_name="稳定接触", group_name_en="stability_contact")
        .add_panel(name="垂直速度惩罚", name_en="reward_lin_vel_z", type="line")
            .add_metric(metrics_name="reward_lin_vel_z",
                        expr="avg(reward_lin_vel_z{})")
            .end_panel()
        .add_panel(name="非预期接触惩罚", name_en="reward_undesired_contacts", type="line")
            .add_metric(metrics_name="reward_undesired_contacts",
                        expr="avg(reward_undesired_contacts{})")
            .end_panel()
        .add_panel(name="终止惩罚", name_en="reward_termination", type="line")
            .add_metric(metrics_name="reward_termination",
                        expr="avg(reward_termination{})")
            .end_panel()
        .add_panel(name="关节位置极限惩罚", name_en="reward_dof_pos_limits", type="line")
            .add_metric(metrics_name="reward_dof_pos_limits",
                        expr="avg(reward_dof_pos_limits{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 6: Joint & action smoothness penalties
        # Group 6: 关节与动作平滑惩罚
        # ==============================================================
        .add_group(group_name="关节动作平滑", group_name_en="joint_action_smoothness")
        .add_panel(name="关节加速度惩罚", name_en="reward_joint_acc", type="line")
            .add_metric(metrics_name="reward_joint_acc",
                        expr="avg(reward_joint_acc{})")
            .end_panel()
        .add_panel(name="动作变化率惩罚 一阶", name_en="reward_action_rate", type="line")
            .add_metric(metrics_name="reward_action_rate",
                        expr="avg(reward_action_rate{})")
            .end_panel()
        .add_panel(name="动作平滑惩罚 二阶", name_en="reward_action_smoothness", type="line")
            .add_metric(metrics_name="reward_action_smoothness",
                        expr="avg(reward_action_smoothness{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 7: Energy / torque (competition scoring items)
        # Group 7: 能耗 / 扭矩（赛题评分项）
        # ==============================================================
        .add_group(group_name="能耗扭矩", group_name_en="energy_torque")
        .add_panel(name="能耗惩罚", name_en="reward_energy", type="line")
            .add_metric(metrics_name="reward_energy",
                        expr="avg(reward_energy{})")
            .end_panel()
        .add_panel(name="关节扭矩惩罚", name_en="reward_joint_torques", type="line")
            .add_metric(metrics_name="reward_joint_torques",
                        expr="avg(reward_joint_torques{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
