# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tencent Kaiwu legged_robot_competition_26 (IDE 22.0.3). Trains a Unitree Go2 quadruped robot via PPO+GAE in Isaac Sim / KaiwuDRL. The active agent is `agent_ppo`; `agent_diy` is an alternative template not currently in use.

## Running

This project depends on Tencent KaiwuDRL and Isaac Lab — most code cannot run outside the competition platform environment.

```bash
# Local quick sanity test (requires KaiwuDRL environment)
python train_test.py

# Switch algorithm: edit algorithm_name in train_test.py ("ppo" or "diy")
```

Training is launched on the Tencent Arena platform, not locally. There is no standalone build/lint/test pipeline.

## Architecture

```
agent_ppo/                    # Active training agent (PPO + GAE, asymmetric actor-critic)
├── conf/
│   ├── conf.py               # StageConfig classes, model dims, hyperparams; Config.CURRENT selects stage
│   ├── train_env_conf_track_nav.toml            # Track-nav stage config (terrain/rewards/curriculum)
│   └── monitor_builder.py    # 9-group monitoring panel definitions
├── feature/
│   ├── policy_observation_process.py   # 301-dim policy obs (45 proprio + 256 height scan)
│   ├── critic_observation_process.py   # 316-dim critic obs (60 privileged + 256 height scan)
│   ├── reward_process.py               # Custom reward functions (_reward_<name> pattern)
│   └── definition.py                   # RolloutStorage and data structures
├── model/actor_critic.py     # Asymmetric MLP actor-critic
└── workflow/train_workflow.py # Training loop, VelocityCurriculum, monitoring, env interaction

conf/
├── configure_app.toml        # Platform-level replay buffer, batch size, model sync settings
├── app_conf_legged_robot_competition_26.toml
└── algo_conf_legged_robot_competition_26.toml
```

## Key Conventions

**Observation dimensions are enforced at runtime:**
- Policy: 301 dims = proprio(45) + height_scan(256)
- Critic: 316 dims = critic_proprio(60) + height_scan(256)
- Changing obs layout requires updating: observation processors, `StageConfig` in conf.py, model inputs, and runtime assertions — all together.

**Reward functions** follow the pattern `_reward_<name>` in `reward_process.py`, activated via TOML `[rewards.<name>]` sections. Training reward is NOT the competition score.

**Velocity curriculum** is Python-side (`VelocityCurriculum` in train_workflow.py), independent of terrain curriculum. Uses tracking ratio (not absolute reward) for stage promotion/demotion. Stage 0 must exactly match `[commands.ranges]` in TOML or startup raises ValueError.

**Terrain config:** Standard subterrain proportions (`pyramid_slope`, `pyramid_slope_inv`, `pyramid_stairs`, `pyramid_stairs_inv`, `maze`) must sum to 1.0.

**Config loading:** `conf.py` loads a base TOML then deep-merges the user TOML on top. Config file naming: `train_env_conf_<task_type>_<stage.name>.toml`.

## Competition Scoring (Standard Mode)

```
total = 0.4 * distance + 0.2 * time + 0.2 * energy + 0.2 * posture
```

Prioritize: traversal reliability/distance first, then speed, energy, posture.

## Environment Protocol

```python
obs, critic_obs = env.reset(usr_conf)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = env.step(actions)
dones = terminated | truncated
```

Actions: shape `(num_envs, 12)`, range `[-1, 1]`, scaled by `action_scale=0.25` to PD joint targets.

## Important Constraints

- Do not remove 301/316 observation dimension assertions without replacing them.
- Do not treat `reward_mean` as competition score.
- Do not relax posture constraints globally for stairs — prefer targeted foot-clearance signals.
- Maximum commanded forward speed is intentionally capped (currently 1.2 m/s) to protect posture/energy scores.
- `num_envs` must be in `[1, 4096]`.
