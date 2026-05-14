# Repository Notes

## Project

This repository is for Tencent Kaiwu `legged_robot_competition_26` on IDE 22.0.3.
The task trains a Unitree Go2 quadruped in Isaac Sim / Isaac Lab through KaiwuDRL.

Current main implementation is `agent_ppo`, not `agent_diy`.

Key entry points:

- `agent_ppo/conf/conf.py`: active stage, model dimensions, PPO hyperparameters.
- `agent_ppo/conf/train_env_conf_standard_locomotion.toml`: current standard-mode training environment, terrain curriculum, reward weights, velocity curriculum.
- `agent_ppo/workflow/train_workflow.py`: PPO training loop, velocity curriculum, monitoring.
- `agent_ppo/feature/policy_observation_process.py`: policy observation processing and 301-dim assertion.
- `agent_ppo/feature/critic_observation_process.py`: critic observation processing and 316-dim assertion.
- `agent_ppo/feature/reward_process.py`: custom locomotion and goal rewards.
- `agent_ppo/model/actor_critic.py`: asymmetric actor-critic MLP.
- `train_test.py`: local quick test runner, currently uses `algorithm_name = "ppo"`.

Documentation to read first:

- `legged-robot/开发指南/01-项目简介.md`: competition task and scoring rules.
- `legged-robot/开发指南/02-环境详述.md`: environment config, observations, actions, monitoring metrics.
- `legged-robot/开发指南/04-数据协议.md`: exact reset/step protocol and tensor layouts.
- `legged-robot/赛题综合分析报告.md`: consolidated analysis and strategy notes.
- `README.md`: current repository-specific summary and training state.

## Competition Rules

There are two modes:

- Standard: traverse mixed terrain as far as possible.
- Track: navigate from start to goal through a chained course ending in a maze.

Standard score:

```text
total = 0.4 * Score_forward
      + 0.2 * Score_time
      + 0.2 * Score_energy
      + 0.2 * Score_posture
```

Standard implications:

- Forward distance is the largest term at 40%.
- Time score is only available after the robot traverses the terrain.
- Traversal is defined from the terrain block center by 2D Euclidean distance:
  `||pos_current - pos_spawn||_2 >= L_terrain / 2 - 0.1`.
- Default `L_terrain = 8m`, so traversal threshold is about `3.9m`.
- The traversal criterion is direction-independent 2D displacement, not strictly x-axis progress.
- Energy is based on joint mechanical power and rewards efficient gait.
- Posture is based on roll/pitch stability.

Track score:

```text
total = completion_ratio * (
    0.4 * Score_time
  + 0.4 * Score_posture
  + 0.2 * Score_energy
)
```

Track implications:

- `completion_ratio = completed robots / total robots`.
- If completion ratio is 0, total score is 0.
- Track optimization must first maximize completion, then improve speed, posture, and energy.
- Track terrain must end with `open_entry_maze`; otherwise environment setup errors.

Failure and timeout:

- Failure: robot body or non-foot joints contact the ground / abnormal posture / fall.
- Timeout: episode reaches maximum step count without completion.
- Algorithm code should use `dones = terminated | truncated`.

Important distinction:

- Environment `reward` is the RL training signal from TOML reward terms.
- Platform `total_score`, `distance_score`, `time_score`, `energy_score`, and `pose_score` are competition scoring/monitoring metrics and are not the same as the training reward.

## Environment And Tensor Protocol

`env.reset(usr_conf)` returns:

```python
obs, critic_obs = env.reset(usr_conf)
```

`env.step(actions)` returns:

```python
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = env.step(actions)
dones = terminated | truncated
```

Policy observation is 301 dim:

```text
obs = [proprio(45) | height_scan(256)]
```

Policy layout:

- `[0:3]`: base angular velocity, scaled by 0.25.
- `[3:6]`: projected gravity.
- `[6:9]`: velocity command `(vx, vy, wz)`.
- `[9:21]`: relative joint positions.
- `[21:33]`: relative joint velocities, scaled by 0.05.
- `[33:45]`: previous action.
- `[45:301]`: 16x16 `height_scan`, clipped to `[-5, 5]`, scaled by 2.5.

Critic observation is 316 dim:

```text
critic_obs = [critic_proprio(60) | height_scan(256)]
```

Critic includes privileged training-only data:

- `base_lin_vel` in addition to policy proprioception.
- `joint_effort` in addition to policy proprioception.

Actions:

- Shape: `(num_envs, 12)`.
- Range: `[-1.0, 1.0]`.
- Meaning: normalized joint target offsets.
- Mapped by `action_scale = 0.25` and added to default joint angles for PD targets.
- Joint order: FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf.

Track extras:

- `env.goal_positions`: `(num_envs, 3)`, target/exit world position.
- `env.goal_yaw`: `(num_envs,)`, target heading.
- `env.scene.sensors["nav_scanner"]`: wider forward occlusion scan for navigation.
- These are not included in default 301-dim observation. If appending them, update policy obs processing, critic obs processing, config dimensions, and model inputs together.

Track observation implementation notes:

- `goal_position_in_robot_frame()` is referenced by several branches and by the
  current `agent_ppo` `track_goal` path, but container/source searches found no
  actual method definition. Do not rely on this helper unless you first implement
  it locally. Expected semantics for a 4-dim goal feature are robot-frame
  relative goal XY, goal distance, and wrapped yaw error to the goal/exit.
- `height_scanner` is already included in the default policy observation as
  `obs[:, 45:301]`. It is a 16x16 scan from
  `env.scene.sensors["height_scanner"]`, scaled by 2.5 and clipped to `[-5, 5]`
  by the default observation term. For raw geometry in custom code, use:
  `scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]`.
- `nav_scanner` is not included in the default 301-dim observation. It is a
  forward occlusion scan from `env.scene.sensors["nav_scanner"]`, configured as
  about 13x11 = 143 rays over 2.5m forward by 2.0m lateral. Access raw hits with
  `nav.data.ray_hits_w` and `nav.data.pos_w`; prefer compact left/center/right
  clearance or wall features before increasing actor input by the full 143 dims.

## Current Training State

Active stage is `Config.CURRENT = LocomotionConfig` in `agent_ppo/conf/conf.py`.
It uses Standard mode locomotion training.

Model:

- Actor input: 301.
- Critic input: 316.
- Actions: 12.
- Actor MLP: `[512, 256, 128] -> 12`, ELU.
- Critic MLP: `[512, 256, 128] -> 1`, LayerNorm + ELU.
- PPO + GAE parameters include `gamma = 0.99` and `lambda = 0.95` in the PPO implementation.

Current Standard terrain config:

- `terrain.mode = "standard"`.
- `num_rows = 10`.
- `num_cols = 20`.
- `difficulty_range = [0.0, 0.6]`.
- `terrain.curriculum = true`.
- `max_init_terrain_level = 1`.
- Terrain proportions currently favor stair acquisition while preserving some slope gait:
  - `pyramid_slope = 0.20`
  - `pyramid_slope_inv = 0.20`
  - `pyramid_stairs = 0.30`
  - `pyramid_stairs_inv = 0.30`
  - `maze = 0.0`
- Domain randomization is enabled.
- Friction randomization is enabled with `[0.3, 1.5]`.
- External pushes are currently disabled.
- Observation noise is enabled.
- `num_envs = 4096`, which is inside the documented valid range `[1, 4096]`.
- Model checkpoints are saved every `200` training episodes via `model_save_interval`.

Velocity curriculum is managed in Python by `VelocityCurriculum` in `train_workflow.py`.
It is separate from terrain curriculum.
It first tries episode-level `reward_track_lin_vel_xy` metrics.  If no episode
ends during the current PPO rollout, it falls back to rollout-level tracking
computed from critic observations:

- `critic_obs[..., 0:2]`: actual body-frame XY velocity.
- `critic_obs[..., 9:11]`: commanded XY velocity.
- `rollout_track_lin_vel_xy_ratio = mean(exp(-||actual_xy - command_xy||^2 / std^2))`.

This fallback is intentional because long stable episodes can leave
`infos["episode"]` empty for many PPO updates.

Current velocity stages:

- Stage 0: `lin_vel_x [0.0, 0.5]`, `lin_vel_y [-0.2, 0.2]`, `ang_vel_yaw [-0.8, 0.8]`.
- Stage 1: `lin_vel_x [0.0, 0.8]`, `lin_vel_y [-0.3, 0.3]`, `ang_vel_yaw [-1.0, 1.0]`.
- Stage 2: `lin_vel_x [0.0, 1.2]`, `lin_vel_y [-0.5, 0.5]`, `ang_vel_yaw [-1.2, 1.2]`.

Stage 0 must exactly match `[commands.ranges]`, or startup raises a `ValueError`.
The current promotion rules are:

- `promote_threshold = 0.62`
- `promote_count = 3`
- `min_checks_per_stage = 40`
- `demote_threshold = 0.38`
- `demote_count = 2`

The maximum commanded forward speed is intentionally capped at `1.2 m/s`
because higher velocity ranges were observed to hurt posture stability and
energy efficiency more than they helped Standard-mode scoring.

Recent stability-related reward settings are intentionally kept close to the
pre-stair-specialization values:

- `ang_vel_xy = -0.35`
- `action_rate = -0.02`
- `action_smoothness = -0.01`
- `dof_vel = -5e-4`

Do not assume stair climbing requires globally relaxed posture constraints.
The preferred direction is to preserve dynamic stability and add better
foot-clearance signals when needed, rather than letting the base pitch/roll
freely to solve stairs.

## Development Guidance

### Local-To-Container Workflow

Make source changes in the local repository first. Treat the Tencent Kaiwu
development container as a runtime target, not as the primary edit source.

Container persistence rule:

- Only these top-level directories are expected to persist across development
  container restarts: `agent_diy`, `agent_ppo`, `conf`, and `log`.
- Store durable project code and operating docs under those directories first.
- Other top-level directories may disappear after a container restart; use them
  only for temporary experiments, scratch uploads, or disposable test outputs.

Recommended flow:

1. Modify and review files locally in this repository.
2. Run local syntax/static checks where possible.
3. Commit or otherwise record the local change set.
4. Synchronize the final intended files to the online development container in
   one batch through the RPC bridge.
5. Verify the container copy by reading back hashes or running a targeted check.

Default sync command:

```bash
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dry-run
CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --apply --py-compile
```

AI agent operating rule:

- When the user asks to sync, deploy, copy local changes to the container, or
  verify local/container parity, prefer the safe sync script above instead of
  editing files directly in the container.
- Always run `--dry-run` first and inspect the file count/sample list before
  `--apply`.
- `--dry-run` is local-only and does not require `agent-browser`, RPC tokens, or
  an active container.
- `--apply` requires a logged-in Tencent Arena browser context plus
  `CODEX_RPC_TOKEN` and `CODEX_RPC_ADMIN_TOKEN` supplied by the user or local
  environment. Never write those token values into files, commits, logs, or
  final answers.
- Strictly distinguish first-time RPC setup from daily use. Only open
  `${CODEX_RPC_BASE}/api/health` in a new browser tab when `IDE_ID`, Tencent
  Arena login state, or the RPC proxy address has not yet been confirmed. If an
  IDE tab such as `/p/common/competition/ide/...` is already open and RPC has
  been verified by script or fetch, do not open more health tabs; use the
  existing IDE/session RPC/fetch path for health, read, write, and exec calls.
- If RPC is not reachable, ask the user to start it inside the container with
  `bash agent_diy/codex_rpc_bridge/start_rpc.sh`.
- If the IDE needs to stay open, use the local keepalive manager:
  `bash agent_diy/codex_rpc_bridge/start_keepalive.sh --status` and
  `bash agent_diy/codex_rpc_bridge/start_keepalive.sh`.
- Full RPC bridge instructions are in
  `agent_diy/codex_rpc_bridge/README.md`,
  `agent_diy/codex_rpc_bridge/SETUP.md`, and
  `agent_diy/codex_rpc_bridge/DAILY_USAGE.md`.

Do not make exploratory edits directly in the container unless the user
explicitly asks for a container-only experiment. If container testing requires a
patch, apply the same patch locally first, then sync it to the container.

Prioritize changes according to scoring:

- Standard: first increase traversal reliability and distance, then improve speed, energy, and posture.
- Track: first increase completion ratio, then optimize time/posture/energy.

When changing observation dimensions:

- Update policy observation processing.
- Update critic observation processing.
- Update `StageConfig` dimensions in `agent_ppo/conf/conf.py`.
- Update model construction assumptions.
- Keep runtime assertions meaningful.

When changing rewards:

- Implement custom `_reward_<name>` in `agent_ppo/feature/reward_process.py`.
- Activate it in TOML as `[rewards.<name>]`.
- Remember training reward is not identical to competition score.
- Energy and posture shaping should support, not block, traversal.

When changing terrain:

- Standard subterrain proportions must sum to 1.0.
- Valid Standard terrains are `pyramid_slope`, `pyramid_slope_inv`, `pyramid_stairs`, `pyramid_stairs_inv`, and `maze`.
- Track subterrain list must end with `open_entry_maze`.
- There is no explicit flat terrain; difficulty 0 slope behaves approximately flat.

When running local checks:

- Use `python train_test.py` for the provided quick training test if the Kaiwu/Isaac environment is available.
- This repository depends on Tencent KaiwuDRL and Isaac environment packages, so ordinary Python unit tests may not run outside the competition environment.

## Cautions

- Do not remove the 301/316 observation dimension checks unless replacing them with updated checks.
- Do not treat `reward_mean` as the competition score.
- Do not optimize time before the robot can reliably complete the relevant task.
- Do not enable Track mode without adding navigation features or a suitable Track training stage.
- Be careful with strong flat-orientation and base-height penalties on stairs; they can suppress necessary body motion for climbing.
