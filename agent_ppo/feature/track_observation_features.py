# -*- coding: UTF-8 -*-
"""Compact Track-mode observation features.

The default environment observation already contains proprioception and the
16x16 height scan.  Track mode additionally needs a small amount of task
context: where the final goal is relative to the robot, and whether the local
scan looks like a wall, stair, or open corridor.
"""

from __future__ import annotations

import math

import torch

from agent_ppo.feature.height_scan_features import compact_maze_scan_features


def compact_track_goal_nav_features(env, scan_flat: torch.Tensor) -> torch.Tensor:
    """Return 10 Track features: goal(4) + local wall/stair navigation(6)."""
    goal_features = compact_goal_features(env, device=scan_flat.device, dtype=scan_flat.dtype)
    nav_features = compact_scan_nav_features(scan_flat)
    return torch.cat((goal_features, nav_features), dim=-1)


def compact_goal_features(env, device=None, dtype=None) -> torch.Tensor:
    """Return normalized goal features in robot frame.

    Layout:
      0. body-frame goal dx, clipped by 10 m
      1. body-frame goal dy, clipped by 10 m
      2. goal distance, clipped by 10 m
      3. wrapped heading error to goal, normalized by pi
    """
    num_envs = int(getattr(env, "num_envs", 1))
    device = device or getattr(env, "device", "cpu")
    dtype = dtype or torch.float32

    if not hasattr(env, "goal_positions") or env.goal_positions is None:
        return torch.zeros(num_envs, 4, device=device, dtype=dtype)

    robot = _get_robot_asset(env)
    if robot is None or not hasattr(robot, "data") or not hasattr(robot.data, "root_pos_w"):
        return torch.zeros(num_envs, 4, device=device, dtype=dtype)

    root_xy = robot.data.root_pos_w[:, :2].to(device=device, dtype=dtype)
    goal_xy = env.goal_positions[:, :2].to(device=device, dtype=dtype)
    goal_delta_w = goal_xy - root_xy

    yaw = _robot_yaw_w(robot, device=device, dtype=dtype)
    if yaw is None:
        return torch.zeros(root_xy.shape[0], 4, device=device, dtype=dtype)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx_w = goal_delta_w[:, 0]
    dy_w = goal_delta_w[:, 1]
    dx_b = cos_yaw * dx_w + sin_yaw * dy_w
    dy_b = -sin_yaw * dx_w + cos_yaw * dy_w

    dist = torch.linalg.norm(goal_delta_w, dim=1)
    goal_yaw = torch.atan2(dy_w, dx_w)
    yaw_error = _wrap_to_pi(goal_yaw - yaw)

    distance_scale = 10.0
    return torch.stack(
        (
            torch.clamp(dx_b / distance_scale, -1.0, 1.0),
            torch.clamp(dy_b / distance_scale, -1.0, 1.0),
            torch.clamp(dist / distance_scale, 0.0, 1.0),
            torch.clamp(yaw_error / math.pi, -1.0, 1.0),
        ),
        dim=1,
    )


def compact_scan_nav_features(scan_flat: torch.Tensor) -> torch.Tensor:
    """Return 6 local navigation features from the scaled 16x16 height scan."""
    grid = scan_flat.view(scan_flat.shape[0], 16, 16)
    return compact_maze_scan_features(
        grid,
        wall_height_threshold=0.18 * 2.5,
        body_clearance_threshold=0.30 * 2.5,
        wall_jump_threshold=0.16 * 2.5,
        stair_min_delta=0.03 * 2.5,
        stair_max_delta=0.24 * 2.5,
    )


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _robot_yaw_w(robot, device, dtype):
    heading = getattr(robot.data, "heading_w", None)
    if heading is not None:
        return heading.to(device=device, dtype=dtype)

    quat = getattr(robot.data, "root_quat_w", None)
    if quat is None:
        return None

    quat = quat.to(device=device, dtype=dtype)
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _get_robot_asset(env):
    scene = getattr(env, "scene", None)
    if scene is None:
        return None

    if hasattr(scene, "__getitem__"):
        keys = []
        keys_fn = getattr(scene, "keys", None)
        if callable(keys_fn):
            try:
                keys = list(keys_fn())
            except Exception:
                keys = []
        for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2", *keys):
            try:
                asset = scene[key]
            except Exception:
                asset = None
            if _has_robot_state(asset):
                return asset

    for attr_name in ("robot", "_robot"):
        asset = getattr(env, attr_name, None)
        if _has_robot_state(asset):
            return asset
    return None


def _has_robot_state(asset) -> bool:
    data = getattr(asset, "data", None)
    return data is not None and hasattr(data, "root_pos_w")
