# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.track_observation_features import (
    compact_scan_nav_features,
    compact_track_goal_nav_features,
)
from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"
    _EXPECTED_OBS_DIM = 316

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
                f"Critic observation dim mismatch: expected {expected_dim}, got {obs.shape[-1]}. "
                "This usually means height_scan or privileged observation layout changed unexpectedly."
            )
        return obs

    def _build_extra_obs(self, obs, stage):
        mode = getattr(stage, "extra_obs_mode", "none")
        if mode == "none":
            return None
        if mode == "track_goal_nav":
            return compact_track_goal_nav_features(self.env, obs[:, -256:])
        if mode == "maze_scan":
            return self._maze_scan_features(obs[:, -256:])
        raise ValueError(f"Unsupported critic extra_obs_mode: {mode}")

    def _maze_scan_features(self, scan_flat):
        return compact_scan_nav_features(scan_flat)
