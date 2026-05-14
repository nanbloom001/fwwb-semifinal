# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256)] → 301 dim
观测布局：[proprio(45) | height_scan(256)] → 301 维

Extending to track terrain (optional):
    In track terrain the environment additionally provides the following
    read-only attributes (not available in standard terrain):
      - env.goal_positions  (num_envs, 3)  — exit position in world frame
      - env.goal_yaw        (num_envs,)    — exit heading in world frame
    The environment always exposes these scene sensors (available in both
    standard and track terrains, accessed via env.scene.sensors["<name>"]):
      - "height_scanner"  — default forward ground-clearance scan
      - "nav_scanner"     — forward-looking occlusion scan (wider range,
                             suited for obstacle avoidance / turning)
    Players can construct their own obs from these inputs. After appending,
    update the Stage config (observation dim) and model input dim accordingly.

扩展到 track 地形时（可选）：
    track 地形下，环境会额外提供以下只读属性（standard 地形没有）：
      - env.goal_positions  (num_envs, 3)  — 出口在世界坐标系下的 3D 位置
      - env.goal_yaw        (num_envs,)    — 出口在世界坐标系下的朝向
    环境在两种地形下都会通过 env.scene.sensors["<name>"] 提供以下传感器：
      - "height_scanner"  — 默认前方地面高度扫描
      - "nav_scanner"     — 前瞻遮挡扫描（范围更大，适合避障 / 转向判断）
    选手可从这些属性和传感器自行构造 obs。
    拼接后需同步修改 Stage 的观测维度和 model 输入维度。
"""

import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.track_observation_features import (
    compact_scan_nav_features,
    compact_track_goal_nav_features,
)
from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"
    _EXPECTED_OBS_DIM = 301

    def process(self):
        obs = self.default_observation()
        stage = Config.current_stage()
        extra_obs = self._build_extra_obs(obs, stage)
        if extra_obs is not None:
            if hasattr(self, "concatenate_terms"):
                obs = self.concatenate_terms(obs, extra_obs)
            else:
                obs = torch.cat((obs, extra_obs), dim=-1)

        expected_dim = self._EXPECTED_OBS_DIM + int(getattr(stage, "num_extra_obs", 0))
        if obs.shape[-1] != expected_dim:
            raise ValueError(
                f"Policy observation dim mismatch: expected {expected_dim}, got {obs.shape[-1]}. "
                "This usually means height_scan is missing or the observation layout changed."
            )
        return obs

    def _build_extra_obs(self, obs, stage):
        mode = getattr(stage, "extra_obs_mode", "none")
        if mode == "none":
            return None
        if mode == "track_goal_nav":
            return compact_track_goal_nav_features(self.env, obs[:, 45:301])
        if mode == "maze_scan":
            return self._maze_scan_features(obs[:, 45:301])
        raise ValueError(f"Unsupported policy extra_obs_mode: {mode}")

    def _maze_scan_features(self, scan_flat):
        return compact_scan_nav_features(scan_flat)
