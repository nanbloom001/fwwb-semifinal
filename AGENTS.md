<!-- AGENTS.md for fwwb-RL-dog -->
# Repository Notes

## Project Overview

This repository trains a Unitree Go2 quadruped in Isaac Sim / Isaac Lab through KaiwuDRL
(Tencent Kaiwu `legged_robot_competition_26`). Two task modes are supported:
standard (mixed-terrain traversal) and track (waypoint/maze navigation).

Current main implementation is `agent_ppo`, not `agent_diy`.
The active training stage is defined in `agent_ppo/conf/conf.py` via `Config.CURRENT`.
As of the latest codebase, `Config.CURRENT = TrackNavConfig` (track navigation fine-tuning),
but it can be switched to `LocomotionConfig` for standard locomotion training.

Key entry points:

- `agent_ppo/conf/conf.py`: active stage, model dimensions, PPO hyperparameters.
- `agent_ppo/conf/train_env_conf_track_nav.toml`: track-mode navigation training config.
- `agent_diy/conf/train_env_conf_standard_locomotion.toml`: standard-mode config template
  (copy it to `agent_ppo/conf/` when switching `Config.CURRENT` to `LocomotionConfig`).
- `agent_ppo/workflow/train_workflow.py`: PPO training loop, velocity curriculum, monitoring.
- `agent_ppo/feature/policy_observation_process.py`: policy observation processing and dimension assertion.
- `agent_ppo/feature/critic_observation_process.py`: critic observation processing and dimension assertion.
- `agent_ppo/feature/reward_process.py`: custom locomotion and goal rewards.
- `agent_ppo/model/actor_critic.py`: asymmetric actor-critic MLP.
- `train_test.py`: local quick test runner, currently uses `algorithm_name = "ppo"`.

Documentation to read first:

- `legged-robot/开发指南/01-项目简介.md`: competition task and scoring rules.
- `legged-robot/开发指南/02-环境详述.md`: environment config, observations, actions, monitoring metrics.
- `legged-robot/开发指南/04-数据协议.md`: exact reset/step protocol and tensor layouts.
- `legged-robot/赛题综合分析报告.md`: consolidated analysis and strategy notes.
- `README.md`: repository summary and current training state.
- `docs/TRAINING_HANDOFF_GUIDE.md`: detailed training logic, reward explanations, and handoff checklist.
- `CHANGES.md`: recent code and configuration change summary.

## Technology Stack

- **Simulation**: Isaac Sim / Isaac Lab (Isaac Lab Unitree-Go2-Velocity task).
- **RL Framework**: KaiwuDRL (Tencent AI Arena internal framework).
- **Deep Learning**: PyTorch.
- **Configuration**: TOML (no `pyproject.toml`, `package.json`, or `Cargo.toml`).
- **Language**: Python 3, with bilingual (Chinese/English) comments and docstrings.
- **Platform**: Training and evaluation run inside a Tencent Kaiwu development container.

## Code Organization

```
agent_ppo/               # Main training agent (current主线)
  algorithm/             # PPO + GAE implementation
    algorithm_ppo.py
  conf/                  # Stage configs, TOML env configs, monitor builder
    conf.py              # StageConfig, Config.CURRENT switch
    monitor_builder.py   # Custom monitoring panel definitions
    train_env_conf_track_nav.toml
  feature/               # Observation processing, rewards, env bridge
    policy_observation_process.py
    critic_observation_process.py
    reward_process.py    # Custom _reward_<name> implementations
    definition.py        # RolloutStorage, Transition, ObsData, ActData
    terrain_gate.py      # Height-scan terrain classification and gating
    velocity_curriculum.py
    phase_command_adapter.py
    goal_features.py
    isaac_env_bridge.py
  model/                 # Actor-Critic network
    actor_critic.py
  workflow/              # Training loop
    train_workflow.py
  agent.py               # Agent class (platform entry point)

agent_diy/               # Alternative agent template (非当前主线)
  algorithm/ conf/ feature/ model/ workflow/ agent.py

conf/                    # Platform-level app/algo config
  app_conf_legged_robot_competition_26.toml
  algo_conf_legged_robot_competition_26.toml
  configure_app.toml

isaac_env/               # Isaac environment interface stub

legged-robot/            # Competition docs and guides (开发指南, 框架文档)

docs/                    # Project docs (TRAINING_HANDOFF_GUIDE, investigations, change records)

arena_frontend_monitor/  # Platform monitoring data capture and post-processing

train_test.py            # Local quick test entry
kaiwu.json               # Platform metadata (version, project_code)
```

## Build and Test Commands

### Local Quick Test

```bash
python train_test.py
```

This runs a minimal training loop via `kaiwudrl.common.utils.train_test_utils.run_train_test`.
It requires the KaiwuDRL and Isaac environment packages, so it generally only works inside
the competition container or a properly configured local environment.

`train_test.py` settings:
- `algorithm_name = "ppo"` (can be switched to `"diy"`).
- Small env vars for fast verification: `train_batch_size=2`, `max_frame_no=1000`, etc.

### Local Syntax / Static Checks

No formal unit test suite exists; use py_compile for quick syntax checks:

```bash
python3 -m py_compile agent_ppo/agent.py
python3 -m py_compile agent_ppo/workflow/train_workflow.py
python3 -m py_compile agent_ppo/feature/reward_process.py
```

### Short-Run Validation Protocol

Before any long training run (e.g., 5h), always do a 15–30 minute short run to verify:

1. Training starts without `NameError`, `ValueError`, or reward-manager registration errors.
2. Monitor panels show non-zero reward curves.
3. `mean_episode_length` is rising or stable.
4. If using command mix / fine-tune schedule / terrain gate, check the corresponding logs appear.
5. `agent_ppo/conf/conf.py` `Config.CURRENT` matches the intended TOML file.

## Code Style Guidelines

1. **Bilingual Comments**: All significant functions and classes have both English and Chinese
   docstrings/comments. New code should follow this convention.
2. **Dimension Assertions**: Policy and critic observation processors enforce runtime dimension
   checks (301-dim and 316-dim base). Do not remove these unless replacing them with updated checks.
3. **Reward Naming**: Custom rewards are implemented as `_reward_<name>` in `reward_process.py`
   and activated in TOML as `[rewards.<name>]`. Parameters go under `[rewards.<name>.params]`.
4. **TOML Naming Convention**: Training env config filenames must follow:
   `train_env_conf_<task_type>_<stage.name>.toml`
5. **Stage Switching**: Change `Config.CURRENT` in `agent_ppo/conf/conf.py`.
   Ensure the corresponding TOML file exists.
6. **NaN/Inf Guards**: The PPO algorithm and rollout storage have explicit NaN/Inf handling.
   Maintain these guards when modifying training logic.
7. **Type Hints**: New Python code should use type hints where practical.

## Testing Instructions

- **No unit tests**: The project depends on KaiwuDRL and Isaac Lab, which are not available
  in a standard Python environment.
- **Validation via training**: The only reliable test is to run `python train_test.py`
  or a short platform training job.
- **Config validation**: `tools.train_env_conf_validate.check_usr_conf` is called at startup.
  If it fails, the agent raises an exception immediately.
- **Dimension validation**: If observation dimensions change, update:
  - `StageConfig` in `conf.py`
  - `policy_observation_process.py`
  - `critic_observation_process.py`
  - Model construction in `actor_critic.py`
  - Any hardcoded slice indices elsewhere.

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
- These are not included in default 301-dim observation. If appending them, update policy obs
  processing, critic obs processing, config dimensions, and model inputs together.

Track observation implementation notes:

- `goal_position_in_robot_frame()` is referenced by several branches, but no actual method
  definition was found in this repository. Do not rely on this helper unless you first implement
  it. Expected semantics for a 4-dim goal feature are robot-frame relative goal XY, goal distance,
  and wrapped yaw error to the goal/exit.
- `height_scanner` is already included in the default policy observation as `obs[:, 45:301]`
  (16x16 scan, scaled by 2.5, clipped to `[-5, 5]`). For raw geometry in custom code use:
  `scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]`.
- `nav_scanner` is not included in the default 301-dim observation (~13x11 = 143 rays over
  2.5m forward by 2.0m lateral). Prefer compact left/center/right clearance or wall features
  before increasing actor input by the full 143 dims.

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
- Default `L_terrain = 8m`, so traversal threshold is about 3.9m.
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
- Platform `total_score`, `distance_score`, `time_score`, `energy_score`, and `pose_score`
  are competition scoring/monitoring metrics and are not the same as the training reward.

## Current Training State

Active stage is controlled by `Config.CURRENT` in `agent_ppo/conf/conf.py`.
The codebase currently defaults to `TrackNavConfig` (track navigation), but retains
`LocomotionConfig` for standard locomotion. Switch stages by changing `Config.CURRENT`.

Model:

- Actor input: 301 (or 304 if `num_goal_obs=3` for track).
- Critic input: 316 (or 319 for track).
- Actions: 12.
- Actor MLP: `[512, 256, 128] -> 12`, ELU.
- Critic MLP: `[512, 256, 128] -> 1`, LayerNorm + ELU.
- PPO + GAE parameters include `gamma = 0.99` and `lambda = 0.95`.

Track terrain config (current `train_env_conf_track_nav.toml`):

- Track subterrain list must end with `open_entry_maze`.
- `num_envs = 3072` in the track TOML.
- Model checkpoints are saved every `model_save_interval` episodes.

Standard terrain config template (`agent_diy/conf/train_env_conf_standard_locomotion.toml`,
used with `LocomotionConfig`):

- `terrain.mode = "standard"`.
- `num_rows = 10`, `num_cols = 20`, `difficulty_range = [0.0, 1.0]`.
- `terrain.curriculum = true`, `max_init_terrain_level = 0` (from lowest difficulty start).
- Terrain proportions favor stair acquisition while preserving some slope gait.
- Domain randomization, friction randomization, external pushes, and observation noise
  are configured in the TOML.

Velocity curriculum is managed in Python by `VelocityCurriculum` in `train_workflow.py`.
It is separate from terrain curriculum.
It first tries episode-level `reward_track_lin_vel_xy` metrics. If no episode ends during the
current PPO rollout, it falls back to rollout-level tracking computed from critic observations:

- `critic_obs[..., 0:2]`: actual body-frame XY velocity.
- `critic_obs[..., 9:11]`: commanded XY velocity.
- `rollout_track_lin_vel_xy_ratio = mean(exp(-||actual_xy - command_xy||^2 / std^2))`.

This fallback is intentional because long stable episodes can leave `infos["episode"]` empty
for many PPO updates.

Velocity stages (as configured):

- Stage 0: `lin_vel_x [0.0, 0.5]`, `lin_vel_y [-0.3, 0.3]`, `ang_vel_yaw [-1.0, 1.0]`.
- Stage 1: `lin_vel_x [0.0, 1.0]`, `lin_vel_y [-0.5, 0.5]`, `ang_vel_yaw [-1.5, 1.5]`.
- Stage 2: `lin_vel_x [0.0, 1.5]`, `lin_vel_y [-0.8, 0.8]`, `ang_vel_yaw [-1.5, 1.5]`.
- Stage 3: `lin_vel_x [-0.5, 2.0]`, `lin_vel_y [-1.0, 1.0]`, `ang_vel_yaw [-1.5, 1.5]`.

Stage 0 must exactly match `[commands.ranges]`, or startup raises a `ValueError`.
Promotion/demotion rules are configurable in TOML under `[velocity_curriculum]`.

## Development Guidance

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
- Valid Standard terrains are `pyramid_slope`, `pyramid_slope_inv`, `pyramid_stairs`,
  `pyramid_stairs_inv`, and `maze`.
- Track subterrain list must end with `open_entry_maze`.
- There is no explicit flat terrain; difficulty 0 slope behaves approximately flat.

## Cautions

- Do not remove the 301/316 observation dimension checks unless replacing them with updated checks.
- Do not treat `reward_mean` as the competition score.
- Do not optimize time before the robot can reliably complete the relevant task.
- Do not enable Track mode without adding navigation features or a suitable Track training stage.
- Be careful with strong flat-orientation and base-height penalties on stairs; they can suppress
  necessary body motion for climbing.
- Do not schedule a reward term from `weight = 0` to non-zero at runtime; platform/Isaac reward
  manager may not register zero-weight terms at environment build time.
