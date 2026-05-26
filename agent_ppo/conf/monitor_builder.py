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
        # Group 0: Training progress — reward-weight-agnostic convergence signals
        # Group 0: 训练进展 — 与 reward 权重无关的收敛信号
        #
        # mean_episode_length: average steps per episode.
        #   Monotonically rising → robot survives longer → genuine convergence.
        #   Plateau at low value → robot keeps falling → weight or algo problem.
        #
        # mean_episode_reward: cumulative reward summed over one episode.
        #   Combines all weighted terms; the one true "fitness" curve.
        #   Use this to compare runs with different weight configurations.
        #
        # mean_episode_length: 每 episode 的平均存活步数。
        #   单调上升 → 机器人越来越耐摔 → 真正收敛。
        #   早期很低并长期平台 → 机器人一直倒 → 权重或算法有问题。
        #
        # mean_episode_reward: 每 episode 全部加权奖励之和。
        #   是唯一综合所有项的真实适应度曲线，用来跨配置对比训练。
        # ==============================================================
        .add_group(group_name="训练进展", group_name_en="training_progress")
        .add_panel(name="平均 episode 步数", name_en="mean_episode_length", type="line")
            .add_metric(metrics_name="mean_episode_length",
                        expr="avg(mean_episode_length{})")
            .end_panel()
        .add_panel(name="每 episode 累计奖励", name_en="mean_episode_reward", type="line")
            .add_metric(metrics_name="mean_episode_reward",
                        expr="avg(mean_episode_reward{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Track result diagnostics from platform scorer counters only.
        # These panels do not depend on custom rewards or rollout-side probes.
        # ==============================================================
        .add_group(group_name="Track结果诊断", group_name_en="track_result_diagnostics")
        .add_panel(name="完成率", name_en="track_completion_ratio", type="line")
            .add_metric(metrics_name="completion_pct",
                        expr="100 * avg(completed_count{}) / clamp_min(avg(completed_count{}) + avg(abnormal_count{}) + avg(timeout_count{}), 1)")
            .add_metric(metrics_name="abnormal_pct",
                        expr="100 * avg(abnormal_count{}) / clamp_min(avg(completed_count{}) + avg(abnormal_count{}) + avg(timeout_count{}), 1)")
            .add_metric(metrics_name="timeout_pct",
                        expr="100 * avg(timeout_count{}) / clamp_min(avg(completed_count{}) + avg(abnormal_count{}) + avg(timeout_count{}), 1)")
            .end_panel()
        .add_panel(name="L样本基数", name_en="track_level_episode_count", type="line")
            .add_metric(metrics_name="l0_count", expr="avg(completed_count_track_l0{}) + avg(abnormal_count_track_l0{}) + avg(timeout_count_track_l0{})")
            .add_metric(metrics_name="l1_count", expr="avg(completed_count_track_l1{}) + avg(abnormal_count_track_l1{}) + avg(timeout_count_track_l1{})")
            .add_metric(metrics_name="l2_count", expr="avg(completed_count_track_l2{}) + avg(abnormal_count_track_l2{}) + avg(timeout_count_track_l2{})")
            .add_metric(metrics_name="l3_count", expr="avg(completed_count_track_l3{}) + avg(abnormal_count_track_l3{}) + avg(timeout_count_track_l3{})")
            .add_metric(metrics_name="l4_count", expr="avg(completed_count_track_l4{}) + avg(abnormal_count_track_l4{}) + avg(timeout_count_track_l4{})")
            .add_metric(metrics_name="l5_count", expr="avg(completed_count_track_l5{}) + avg(abnormal_count_track_l5{}) + avg(timeout_count_track_l5{})")
            .add_metric(metrics_name="l6_count", expr="avg(completed_count_track_l6{}) + avg(abnormal_count_track_l6{}) + avg(timeout_count_track_l6{})")
            .add_metric(metrics_name="l7_count", expr="avg(completed_count_track_l7{}) + avg(abnormal_count_track_l7{}) + avg(timeout_count_track_l7{})")
            .add_metric(metrics_name="l8_count", expr="avg(completed_count_track_l8{}) + avg(abnormal_count_track_l8{}) + avg(timeout_count_track_l8{})")
            .add_metric(metrics_name="l9_count", expr="avg(completed_count_track_l9{}) + avg(abnormal_count_track_l9{}) + avg(timeout_count_track_l9{})")
            .end_panel()
        .add_panel(name="L异常率", name_en="track_level_abnormal_pct", type="line")
            .add_metric(metrics_name="l0_abn_pct", expr="100 * avg(abnormal_count_track_l0{}) / clamp_min(avg(completed_count_track_l0{}) + avg(abnormal_count_track_l0{}) + avg(timeout_count_track_l0{}), 1)")
            .add_metric(metrics_name="l1_abn_pct", expr="100 * avg(abnormal_count_track_l1{}) / clamp_min(avg(completed_count_track_l1{}) + avg(abnormal_count_track_l1{}) + avg(timeout_count_track_l1{}), 1)")
            .add_metric(metrics_name="l2_abn_pct", expr="100 * avg(abnormal_count_track_l2{}) / clamp_min(avg(completed_count_track_l2{}) + avg(abnormal_count_track_l2{}) + avg(timeout_count_track_l2{}), 1)")
            .add_metric(metrics_name="l3_abn_pct", expr="100 * avg(abnormal_count_track_l3{}) / clamp_min(avg(completed_count_track_l3{}) + avg(abnormal_count_track_l3{}) + avg(timeout_count_track_l3{}), 1)")
            .add_metric(metrics_name="l4_abn_pct", expr="100 * avg(abnormal_count_track_l4{}) / clamp_min(avg(completed_count_track_l4{}) + avg(abnormal_count_track_l4{}) + avg(timeout_count_track_l4{}), 1)")
            .add_metric(metrics_name="l5_abn_pct", expr="100 * avg(abnormal_count_track_l5{}) / clamp_min(avg(completed_count_track_l5{}) + avg(abnormal_count_track_l5{}) + avg(timeout_count_track_l5{}), 1)")
            .add_metric(metrics_name="l6_abn_pct", expr="100 * avg(abnormal_count_track_l6{}) / clamp_min(avg(completed_count_track_l6{}) + avg(abnormal_count_track_l6{}) + avg(timeout_count_track_l6{}), 1)")
            .add_metric(metrics_name="l7_abn_pct", expr="100 * avg(abnormal_count_track_l7{}) / clamp_min(avg(completed_count_track_l7{}) + avg(abnormal_count_track_l7{}) + avg(timeout_count_track_l7{}), 1)")
            .add_metric(metrics_name="l8_abn_pct", expr="100 * avg(abnormal_count_track_l8{}) / clamp_min(avg(completed_count_track_l8{}) + avg(abnormal_count_track_l8{}) + avg(timeout_count_track_l8{}), 1)")
            .add_metric(metrics_name="l9_abn_pct", expr="100 * avg(abnormal_count_track_l9{}) / clamp_min(avg(completed_count_track_l9{}) + avg(abnormal_count_track_l9{}) + avg(timeout_count_track_l9{}), 1)")
            .end_panel()
        .add_panel(name="L超时率", name_en="track_level_timeout_pct", type="line")
            .add_metric(metrics_name="l0_timeout_pct", expr="100 * avg(timeout_count_track_l0{}) / clamp_min(avg(completed_count_track_l0{}) + avg(abnormal_count_track_l0{}) + avg(timeout_count_track_l0{}), 1)")
            .add_metric(metrics_name="l1_timeout_pct", expr="100 * avg(timeout_count_track_l1{}) / clamp_min(avg(completed_count_track_l1{}) + avg(abnormal_count_track_l1{}) + avg(timeout_count_track_l1{}), 1)")
            .add_metric(metrics_name="l2_timeout_pct", expr="100 * avg(timeout_count_track_l2{}) / clamp_min(avg(completed_count_track_l2{}) + avg(abnormal_count_track_l2{}) + avg(timeout_count_track_l2{}), 1)")
            .add_metric(metrics_name="l3_timeout_pct", expr="100 * avg(timeout_count_track_l3{}) / clamp_min(avg(completed_count_track_l3{}) + avg(abnormal_count_track_l3{}) + avg(timeout_count_track_l3{}), 1)")
            .add_metric(metrics_name="l4_timeout_pct", expr="100 * avg(timeout_count_track_l4{}) / clamp_min(avg(completed_count_track_l4{}) + avg(abnormal_count_track_l4{}) + avg(timeout_count_track_l4{}), 1)")
            .add_metric(metrics_name="l5_timeout_pct", expr="100 * avg(timeout_count_track_l5{}) / clamp_min(avg(completed_count_track_l5{}) + avg(abnormal_count_track_l5{}) + avg(timeout_count_track_l5{}), 1)")
            .add_metric(metrics_name="l6_timeout_pct", expr="100 * avg(timeout_count_track_l6{}) / clamp_min(avg(completed_count_track_l6{}) + avg(abnormal_count_track_l6{}) + avg(timeout_count_track_l6{}), 1)")
            .add_metric(metrics_name="l7_timeout_pct", expr="100 * avg(timeout_count_track_l7{}) / clamp_min(avg(completed_count_track_l7{}) + avg(abnormal_count_track_l7{}) + avg(timeout_count_track_l7{}), 1)")
            .add_metric(metrics_name="l8_timeout_pct", expr="100 * avg(timeout_count_track_l8{}) / clamp_min(avg(completed_count_track_l8{}) + avg(abnormal_count_track_l8{}) + avg(timeout_count_track_l8{}), 1)")
            .add_metric(metrics_name="l9_timeout_pct", expr="100 * avg(timeout_count_track_l9{}) / clamp_min(avg(completed_count_track_l9{}) + avg(abnormal_count_track_l9{}) + avg(timeout_count_track_l9{}), 1)")
            .end_panel()
        .end_group()

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
        .add_panel(name="课程追踪比例", name_en="vel_curriculum_tracking_ratio", type="line")
            .add_metric(metrics_name="vel_curriculum_tracking_ratio",
                        expr="avg(vel_curriculum_tracking_ratio{})")
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
        .add_panel(name="高难度压力", name_en="high_difficulty_pressure", type="line")
            .add_metric(metrics_name="reward_high_difficulty_pose_pressure",
                        expr="avg(reward_high_difficulty_pose_pressure{})")
            .add_metric(metrics_name="reward_high_difficulty_ang_vel_xy",
                        expr="avg(reward_high_difficulty_ang_vel_xy{})")
            .add_metric(metrics_name="reward_high_difficulty_energy",
                        expr="avg(reward_high_difficulty_energy{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # ==============================================================
        # Group 8: RL navigation rewards
        # ==============================================================
        .add_group(group_name="RL导航", group_name_en="rl_navigation")
        .add_panel(name="前进头向", name_en="reward_forward_heading_velocity", type="line")
            .add_metric(metrics_name="reward_forward_heading_velocity",
                        expr="avg(reward_forward_heading_velocity{})")
            .end_panel()
        .add_panel(name="后退惩罚", name_en="reward_backward_penalty", type="line")
            .add_metric(metrics_name="reward_backward_penalty",
                        expr="avg(reward_backward_penalty{})")
            .end_panel()
        .add_panel(name="目标朝向", name_en="reward_goal_heading_alignment", type="line")
            .add_metric(metrics_name="reward_goal_heading_alignment",
                        expr="avg(reward_goal_heading_alignment{})")
            .end_panel()
        .add_panel(name="目标推进", name_en="reward_goal_velocity_projection", type="line")
            .add_metric(metrics_name="reward_goal_velocity_projection",
                        expr="avg(reward_goal_velocity_projection{})")
            .end_panel()
        .add_panel(name="逆向惩罚", name_en="reward_goal_backtrack_penalty", type="line")
            .add_metric(metrics_name="reward_goal_backtrack_penalty",
                        expr="avg(reward_goal_backtrack_penalty{})")
            .end_panel()
        .add_panel(name="近终点方向错误", name_en="reward_goal_near_wrong_direction_penalty", type="line")
            .add_metric(metrics_name="reward_goal_near_wrong_direction_penalty",
                        expr="avg(reward_goal_near_wrong_direction_penalty{})")
            .end_panel()
        .add_panel(name="接近目标", name_en="reward_approach_goal", type="line")
            .add_metric(metrics_name="reward_approach_goal",
                        expr="avg(reward_approach_goal{})")
            .end_panel()
        .add_panel(name="目标距离", name_en="reward_goal_distance", type="line")
            .add_metric(metrics_name="reward_goal_distance",
                        expr="avg(reward_goal_distance{})")
            .end_panel()
        .add_panel(name="到达目标", name_en="reward_reach_goal", type="line")
            .add_metric(metrics_name="reward_reach_goal",
                        expr="avg(reward_reach_goal{})")
            .end_panel()
        .add_panel(name="任务完成", name_en="reward_task_complete", type="line")
            .add_metric(metrics_name="reward_task_complete",
                        expr="avg(reward_task_complete{})")
            .end_panel()
        .add_panel(name="错过终点惩罚", name_en="reward_goal_miss_penalty", type="line")
            .add_metric(metrics_name="reward_goal_miss_penalty",
                        expr="avg(reward_goal_miss_penalty{})")
            .end_panel()
        .add_panel(name="导航时间", name_en="reward_navigation_time", type="line")
            .add_metric(metrics_name="reward_navigation_time",
                        expr="avg(reward_navigation_time{})")
            .end_panel()
        .add_panel(name="地形时间压力", name_en="terrain_time_pressure", type="line")
            .add_metric(metrics_name="reward_slope_time_pressure",
                        expr="avg(reward_slope_time_pressure{})")
            .add_metric(metrics_name="reward_low_mid_stair_time_pressure",
                        expr="avg(reward_low_mid_stair_time_pressure{})")
            .end_panel()
        .add_panel(name="迷宫门控", name_en="reward_maze_context_gate", type="line")
            .add_metric(metrics_name="reward_maze_context_gate",
                        expr="avg(reward_maze_context_gate{})")
            .end_panel()
        .add_panel(name="撞墙惩罚", name_en="reward_wall_collision", type="line")
            .add_metric(metrics_name="reward_wall_collision",
                        expr="avg(reward_wall_collision{})")
            .end_panel()
        .add_panel(name="撞墙冲击惩罚", name_en="reward_nav_wall_impact_penalty", type="line")
            .add_metric(metrics_name="reward_nav_wall_impact_penalty",
                        expr="avg(reward_nav_wall_impact_penalty{})")
            .end_panel()
        .add_panel(name="卡墙前推惩罚", name_en="reward_nav_wall_stuck_push_penalty", type="line")
            .add_metric(metrics_name="reward_nav_wall_stuck_push_penalty",
                        expr="avg(reward_nav_wall_stuck_push_penalty{})")
            .end_panel()
        .add_panel(name="贴墙停滞", name_en="reward_wall_stall_penalty", type="line")
            .add_metric(metrics_name="reward_wall_stall_penalty",
                        expr="avg(reward_wall_stall_penalty{})")
            .end_panel()
        .add_panel(name="离墙惩罚", name_en="reward_wall_proximity", type="line")
            .add_metric(metrics_name="reward_wall_proximity",
                        expr="avg(reward_wall_proximity{})")
            .end_panel()
        .add_panel(name="空旷奖励", name_en="reward_open_space", type="line")
            .add_metric(metrics_name="reward_open_space",
                        expr="avg(reward_open_space{})")
            .end_panel()
        .add_panel(name="中心偏离", name_en="reward_corridor_centering", type="line")
            .add_metric(metrics_name="reward_corridor_centering",
                        expr="avg(reward_corridor_centering{})")
            .end_panel()
        .add_panel(name="探索奖励", name_en="reward_directed_exploration", type="line")
            .add_metric(metrics_name="reward_directed_exploration",
                        expr="avg(reward_directed_exploration{})")
            .end_panel()
        .add_panel(name="停滞惩罚", name_en="reward_stuck_penalty", type="line")
            .add_metric(metrics_name="reward_stuck_penalty",
                        expr="avg(reward_stuck_penalty{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 9: Final terrain gate.  All ratio panels are percentages.
        # Reward probe terms use a tiny weight, so ratios divide probes with
        # the same weight; raw value panels multiply by 1e6 to recover units.
        # ==============================================================
        .add_group(group_name="最终地形门控", group_name_en="terrain_gate_final")
        .add_panel(name="地形占比", name_en="gate_final_terrain_pct", type="line")
            .add_metric(metrics_name="all_flat_pct",
                        expr="100 * avg(reward_gate_final_flat{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="all_slope_pct",
                        expr="100 * avg(reward_gate_final_slope{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="all_stairs_pct",
                        expr="100 * avg(reward_gate_final_stairs{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="all_maze_pct",
                        expr="100 * avg(reward_gate_final_maze{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="all_wall_or_invalid_pct",
                        expr="100 * avg(reward_gate_final_invalid{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="flat_non_maze_pct",
                        expr="100 * avg(reward_gate_final_flat{}) / clamp_min(avg(reward_gate_final_non_maze_sum{}), 1e-12)")
            .add_metric(metrics_name="slope_non_maze_pct",
                        expr="100 * avg(reward_gate_final_slope{}) / clamp_min(avg(reward_gate_final_non_maze_sum{}), 1e-12)")
            .add_metric(metrics_name="stairs_non_maze_pct",
                        expr="100 * avg(reward_gate_final_stairs{}) / clamp_min(avg(reward_gate_final_non_maze_sum{}), 1e-12)")
            .add_metric(metrics_name="wall_or_invalid_non_maze_pct",
                        expr="100 * avg(reward_gate_final_invalid{}) / clamp_min(avg(reward_gate_final_non_maze_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="有效难度占比", name_en="gate_final_valid_difficulty_pct", type="line")
            .add_metric(metrics_name="final_low_valid_pct",
                        expr="100 * avg(reward_gate_final_difficulty_low{}) / clamp_min(avg(reward_gate_final_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="final_mid_valid_pct",
                        expr="100 * avg(reward_gate_final_difficulty_mid{}) / clamp_min(avg(reward_gate_final_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="final_high_valid_pct",
                        expr="100 * avg(reward_gate_final_difficulty_high{}) / clamp_min(avg(reward_gate_final_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="nan_low_valid_pct",
                        expr="100 * avg(reward_gate_nan_active_difficulty_low{}) / clamp_min(avg(reward_gate_nan_active_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="nan_mid_valid_pct",
                        expr="100 * avg(reward_gate_nan_active_difficulty_mid{}) / clamp_min(avg(reward_gate_nan_active_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="nan_high_valid_pct",
                        expr="100 * avg(reward_gate_nan_active_difficulty_high{}) / clamp_min(avg(reward_gate_nan_active_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="raw_low_valid_pct",
                        expr="100 * avg(reward_gate_raw_active_difficulty_low{}) / clamp_min(avg(reward_gate_raw_active_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="raw_mid_valid_pct",
                        expr="100 * avg(reward_gate_raw_active_difficulty_mid{}) / clamp_min(avg(reward_gate_raw_active_difficulty_sum{}), 1e-12)")
            .add_metric(metrics_name="raw_high_valid_pct",
                        expr="100 * avg(reward_gate_raw_active_difficulty_high{}) / clamp_min(avg(reward_gate_raw_active_difficulty_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="难度覆盖", name_en="gate_final_difficulty_coverage", type="line")
            .add_metric(metrics_name="difficulty_valid_pct",
                        expr="100 * avg(reward_gate_final_difficulty_sum{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="difficulty_not_applicable_pct",
                        expr="100 * avg(reward_gate_final_difficulty_unknown{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="valid_slope_stairs_raw",
                        expr="avg(reward_gate_final_difficulty_sum{}) * 1000000")
            .end_panel()
        .add_panel(name="训练压力覆盖", name_en="gate_training_pressure_pct", type="line")
            .add_metric(metrics_name="slope_time_pressure_pct",
                        expr="100 * avg(reward_gate_slope_time_pressure_active{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="low_mid_stair_time_pressure_pct",
                        expr="100 * avg(reward_gate_low_mid_stair_time_pressure_active{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="high_diff_pressure_pct",
                        expr="100 * avg(reward_gate_high_difficulty_pressure_active{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="运行状态与来源", name_en="gate_final_status_source_pct", type="line")
            .add_metric(metrics_name="final_valid_pct",
                        expr="100 * avg(reward_gate_final_gate_valid{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="raw_height_available_pct",
                        expr="100 * avg(reward_gate_raw_available{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="nav_scanner_available_pct",
                        expr="100 * avg(reward_gate_raw_nav_available{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="shadow_mode_pct",
                        expr="100 * avg(reward_gate_mode_shadow{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="control_mode_pct",
                        expr="100 * avg(reward_gate_mode_control{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="command_written_pct",
                        expr="100 * avg(reward_gate_command_written{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="source_current_pct",
                        expr="100 * avg(reward_gate_final_source_current{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="source_nan_pct",
                        expr="100 * avg(reward_gate_final_source_nan{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="source_raw_fused_pct",
                        expr="100 * avg(reward_gate_final_source_raw{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="source_sticky_pct",
                        expr="100 * avg(reward_gate_final_source_sticky{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="Sticky稳定性", name_en="gate_final_sticky_stability", type="line")
            .add_metric(metrics_name="hold_seconds",
                        expr="avg(reward_gate_sticky_hold_steps{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12) * 0.02")
            .add_metric(metrics_name="pending_confirm_steps",
                        expr="avg(reward_gate_sticky_pending_count{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="flat_release_confirm_steps",
                        expr="avg(reward_gate_sticky_flat_release_count{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="terrain_switch_env_pct",
                        expr="100 * avg(reward_gate_terrain_switch{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="difficulty_switch_env_pct",
                        expr="100 * avg(reward_gate_difficulty_switch{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="early_flat_release_pct",
                        expr="100 * avg(reward_gate_sticky_early_flat_release{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="active_preempt_switch_pct",
                        expr="100 * avg(reward_gate_sticky_active_preempt_switch{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="迷宫先验", name_en="gate_final_maze_prior", type="line")
            .add_metric(metrics_name="maze_instant_pct",
                        expr="100 * avg(reward_gate_maze_instant{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="maze_sticky_pct",
                        expr="100 * avg(reward_gate_maze_phase{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="maze_confirm_steps",
                        expr="avg(reward_gate_maze_confirm_count{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="难度缓存", name_en="gate_final_difficulty_cache", type="line")
            .add_metric(metrics_name="cache_valid_pct",
                        expr="100 * avg(reward_gate_track_cache_valid{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_unknown_pct",
                        expr="100 * avg(reward_gate_track_cache_unknown{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_low_pct",
                        expr="100 * avg(reward_gate_track_cache_low{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_mid_pct",
                        expr="100 * avg(reward_gate_track_cache_mid{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_high_pct",
                        expr="100 * avg(reward_gate_track_cache_high{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_source_slope_pct",
                        expr="100 * avg(reward_gate_track_cache_source_slope{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="cache_source_stairs_pct",
                        expr="100 * avg(reward_gate_track_cache_source_stairs{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="maze_using_cache_pct",
                        expr="100 * avg(reward_gate_maze_using_cached_difficulty{}) / clamp_min(avg(reward_gate_final_maze{}), 1e-12)")
            .add_metric(metrics_name="cache_factor",
                        expr="avg(reward_gate_track_cache_factor{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="上下台阶后验", name_en="gate_final_stair_direction", type="line")
            .add_metric(metrics_name="stair_up_pct",
                        expr="100 * avg(reward_gate_stair_up{}) / clamp_min(avg(reward_gate_final_stairs{}), 1e-12)")
            .add_metric(metrics_name="stair_down_pct",
                        expr="100 * avg(reward_gate_stair_down{}) / clamp_min(avg(reward_gate_final_stairs{}), 1e-12)")
            .add_metric(metrics_name="stair_unknown_pct",
                        expr="100 * avg(reward_gate_stair_unknown_dir{}) / clamp_min(avg(reward_gate_final_stairs{}), 1e-12)")
            .add_metric(metrics_name="pitch_abs_rad",
                        expr="avg(reward_gate_pitch_abs{}) * 1000000")
            .add_metric(metrics_name="updown_confidence_pct",
                        expr="avg(reward_gate_updown_confidence{}) * 100000000")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 10: Gate source comparison and command verification.
        # ==============================================================
        .add_group(group_name="地形门控校验", group_name_en="terrain_gate_validation")
        .add_panel(name="有效样本量", name_en="gate_validation_active_samples", type="line")
            .add_metric(metrics_name="final_active_sum_raw",
                        expr="avg(reward_gate_final_difficulty_sum{}) * 1000000")
            .add_metric(metrics_name="nan_active_sum_raw",
                        expr="avg(reward_gate_nan_active_difficulty_sum{}) * 1000000")
            .add_metric(metrics_name="raw_active_sum_raw",
                        expr="avg(reward_gate_raw_active_difficulty_sum{}) * 1000000")
            .add_metric(metrics_name="final_non_maze_sum_raw",
                        expr="avg(reward_gate_final_non_maze_sum{}) * 1000000")
            .add_metric(metrics_name="final_terrain_sum_raw",
                        expr="avg(reward_gate_final_terrain_sum{}) * 1000000")
            .end_panel()
        .add_panel(name="门控分歧", name_en="gate_validation_disagreement_pct", type="line")
            .add_metric(metrics_name="nan_raw_stair_disagree_pct",
                        expr="100 * avg(reward_gate_nan_raw_stair_disagree{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="nan_raw_slope_disagree_pct",
                        expr="100 * avg(reward_gate_nan_raw_slope_disagree{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="nan_raw_wall_disagree_pct",
                        expr="100 * avg(reward_gate_nan_raw_wall_disagree{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="raw_wall_non_maze_pct",
                        expr="100 * avg(reward_gate_raw_wall_non_maze{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .add_panel(name="raw几何信号", name_en="gate_validation_raw_geometry", type="line")
            .add_metric(metrics_name="edge_sharpness_m",
                        expr="avg(reward_gate_raw_edge_sharpness{}) * 1000000")
            .add_metric(metrics_name="edge_locality",
                        expr="avg(reward_gate_raw_edge_locality{}) * 1000000")
            .add_metric(metrics_name="slope_smoothness",
                        expr="avg(reward_gate_raw_slope_smoothness{}) * 1000000")
            .add_metric(metrics_name="step_delta_m",
                        expr="avg(reward_gate_raw_step_delta{}) * 1000000")
            .add_metric(metrics_name="difficulty_signal",
                        expr="avg(reward_gate_raw_difficulty_signal{}) * 1000000")
            .end_panel()
        .add_panel(name="raw nav墙体", name_en="gate_validation_raw_nav_wall", type="line")
            .add_metric(metrics_name="front_wall_score",
                        expr="avg(reward_gate_raw_nav_front_wall_score{}) * 1000000")
            .add_metric(metrics_name="left_wall_score",
                        expr="avg(reward_gate_raw_nav_left_wall_score{}) * 1000000")
            .add_metric(metrics_name="right_wall_score",
                        expr="avg(reward_gate_raw_nav_right_wall_score{}) * 1000000")
            .add_metric(metrics_name="front_blocked_pct",
                        expr="avg(reward_gate_raw_nav_front_blocked{}) * 100000000")
            .add_metric(metrics_name="open_side",
                        expr="avg(reward_gate_raw_nav_open_side{}) * 1000000")
            .end_panel()
        .add_panel(name="命令同步", name_en="gate_validation_command_sync", type="line")
            .add_metric(metrics_name="worker_cmd_vx_mps",
                        expr="avg(reward_gate_worker_cmd_vx{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="target_cmd_vx_mps",
                        expr="avg(reward_gate_target_cmd_vx{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="policy_cmd_error_mps",
                        expr="avg(reward_gate_policy_cmd_error{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="critic_cmd_error_mps",
                        expr="avg(reward_gate_critic_cmd_error{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .add_metric(metrics_name="actual_minus_cmd_vx_mps",
                        expr="avg(reward_gate_actual_minus_cmd_vx{}) / clamp_min(avg(reward_gate_final_terrain_sum{}), 1e-12)")
            .end_panel()
        .end_group()

        # Group 9: Physical observations (SI units, weight-independent)
        # Group 10: 物理观测量（SI 单位，与 reward 权重无关）
        #
        # These panels show the *physical* quantities that the reward functions
        # operate on, NOT the weighted reward values.  They let you answer
        # "is the robot actually converging in terms of real physics?"
        # independently of whether you've tuned the reward weights well.
        #
        # 这些面板显示奖励函数实际计算所用的物理量（SI 单位），
        # 而非加权后的奖励值。用于独立于权重设置判断"机器人物理收敛了吗"。
        #
        # obs_lin_vel_x_error: mean |cmd_vx - actual_vx| (m/s)
        #   → 速度追踪误差（前向）；收敛后应趋近 0 m/s。
        #     值持续 >0.3 m/s 且 reward 已平台 → 权重可能有矛盾梯度。
        #
        # obs_lin_vel_y_error: mean |cmd_vy - actual_vy| (m/s)
        #   → 速度追踪误差（侧向）；正常约 0.1 m/s 以内。
        #
        # obs_actual_vel_x: mean actual forward speed (m/s)
        #   → 机器人实际平均前向速度；Stage-0 目标 0.25 m/s，随阶段升高。
        #
        # obs_base_height: mean base height (m)
        #   → 机身高度均值；目标 0.38 m。
        #     偏低说明腿弯曲不足（或摔倒率高）；偏高说明过度伸展。
        #
        # obs_ang_vel_xy: mean |ω_pitch, ω_roll| magnitude (rad/s)
        #   → pitch/roll 角速度幅值；反映机身倾斜程度。
        #     正常行走约 0.3–0.8 rad/s；持续 >1.5 说明姿态不稳定。
        # ==============================================================
        .add_group(group_name="物理观测量", group_name_en="physics_obs")
        .add_panel(name="前向速度对照", name_en="velocity_x_stat", type="stat")
            .add_metric(metrics_name="cmd_vx",
                        expr="avg(obs_cmd_vel_x{})")
            .add_metric(metrics_name="actual_vx",
                        expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="侧向速度对照", name_en="velocity_y_stat", type="stat")
            .add_metric(metrics_name="cmd_vy",
                        expr="avg(obs_cmd_vel_y{})")
            .add_metric(metrics_name="actual_vy",
                        expr="avg(obs_actual_vel_y{})")
            .end_panel()
        .add_panel(name="偏航速度对照", name_en="velocity_yaw_stat", type="stat")
            .add_metric(metrics_name="cmd_yaw",
                        expr="avg(obs_cmd_yaw{})")
            .add_metric(metrics_name="actual_yaw",
                        expr="avg(obs_actual_yaw{})")
            .end_panel()
        .add_panel(name="速度误差对照", name_en="velocity_error_stat", type="stat")
            .add_metric(metrics_name="vx_error",
                        expr="avg(obs_lin_vel_x_error{})")
            .add_metric(metrics_name="yaw_error",
                        expr="avg(obs_yaw_error{})")
            .end_panel()
        .add_panel(name="前向速度追踪误差", name_en="obs_lin_vel_x_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_x_error",
                        expr="avg(obs_lin_vel_x_error{})")
            .end_panel()
        .add_panel(name="侧向速度追踪误差", name_en="obs_lin_vel_y_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_y_error",
                        expr="avg(obs_lin_vel_y_error{})")
            .end_panel()
        .add_panel(name="实际前向速度", name_en="obs_actual_vel_x", type="line")
            .add_metric(metrics_name="obs_actual_vel_x",
                        expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="机身高度", name_en="obs_base_height", type="line")
            .add_metric(metrics_name="obs_base_height",
                        expr="avg(obs_base_height{})")
            .end_panel()
        .add_panel(name="pitch roll 角速度", name_en="obs_ang_vel_xy", type="line")
            .add_metric(metrics_name="obs_ang_vel_xy",
                        expr="avg(obs_ang_vel_xy{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
