# 2026-05-11 wkn 分支迷宫训练同步记录

## 背景

本轮目标是在 `wkn` 分支同步其他分支中对楼梯、迷宫、Track 导航有价值的内容，并重点实现迷宫训练优化方案：

- 保留当前 locomotion/stair 能力，不一次性切换到高比例迷宫。
- 支持新观测项，同时允许旧 301/316 维模型继续以部分加载方式续训。
- 针对迷宫和上楼梯增加基于 height scan 几何特征的 reward，避免依赖当前容器暂时拿不到的 per-env terrain label。
- 地形比例和 reward 权重按 episode 线性软切换，降低训练崩溃风险。

当前分支：`wkn`。

## 本轮新增训练阶段

新增阶段通过环境变量 `FWWB_STAGE` 切换。当前代码默认阶段为 `StandardMazeConfig`；如需回到原始 locomotion 训练，可显式设置 `FWWB_STAGE=locomotion`。

新增阶段：

- `FWWB_STAGE=maze`
  - 文件：`agent_ppo/conf/train_env_conf_standard_maze.toml`
  - 模式：Standard
  - 用途：Standard 迷宫适配预训练。
  - policy obs：`301 + 6 = 307`
  - critic obs：`316 + 6 = 322`
  - 6 维新增观测来自 16x16 height scan 的几何判定：左/中/右墙体分数、台阶分数、转向偏置、中心墙体距离。

- `FWWB_STAGE=nav` 或 `FWWB_STAGE=track_nav`
  - 文件：`agent_ppo/conf/train_env_conf_track_nav.toml`
  - 模式：Track
  - 用途：后续 Track 导航训练预留。
  - policy obs：`301 + 4 = 305`
  - critic obs：`316 + 4 = 320`
  - Track 子地形最后一项保持为 `open_entry_maze`。

相关代码：

- `agent_ppo/conf/conf.py`
- `agent_ppo/agent.py`
- `agent_ppo/feature/policy_observation_process.py`
- `agent_ppo/feature/critic_observation_process.py`

## 迷宫软切换配置

文件：`agent_ppo/conf/train_env_conf_standard_maze.toml`

新增 `[training_schedule]`，由 `agent_ppo/workflow/train_workflow.py` 中的 `TrainingScheduleController` 执行。

当前配置：

```toml
[training_schedule]
enabled = true
start_episode = 0
end_episode = 800
apply_interval = 20
```

地形比例从原始比例线性切换到目标比例：

```text
initial:
  pyramid_slope       = 0.20
  pyramid_slope_inv   = 0.20
  pyramid_stairs      = 0.30
  pyramid_stairs_inv  = 0.30
  maze                = 0.00

target:
  pyramid_slope       = 0.15
  pyramid_slope_inv   = 0.15
  pyramid_stairs      = 0.10
  pyramid_stairs_inv  = 0.30
  maze                = 0.30
```

注意：用户要求的最终比例已落实：

- `maze = 0.30`
- `pyramid_stairs_inv = 0.30`

reward 权重也按 episode 线性切换：

```text
track_lin_vel_xy:    2.5  -> 1.5
track_ang_vel_z:     1.5  -> 0.8
forward_velocity:    0.0  -> 0.15
obstacle_evasion:    0.0  -> -1.5
undesired_contacts: -0.3  -> -0.5
feet_stumble:       -0.05 -> -0.08
```

每次应用 schedule 时会 warning 打印：

```text
[TrainingSchedule] apply episode=... progress=... terrain=... rewards=...; calling env.reset
```

## 迷宫和台阶 Reward 机制

新增或优化 reward：

- `_reward_obstacle_evasion`
  - 使用 height scan 判断“高于机身、跳变明显”的迷宫墙体。
  - 对正前方近距离墙体给更强惩罚，并按左右墙体分数给转向偏置。
  - 通过墙体高度、局部高度跳变和前方扇区共同判断，尽量避免把普通台阶误判为迷宫墙。

- `_reward_feet_clearance`
  - 用 height scan 的台阶分数做门控，主要在低矮、连续、可攀爬的台阶结构上生效。
  - 与迷宫墙体判定分离，避免在墙前鼓励抬脚硬撞。

当前版本没有继续使用 `terrain_scoped` 包装 reward。此前探测显示 aisrv 侧拿不到 worker 内部的 per-env terrain label；因此本轮实际采用“观测几何门控”的方式实现迷宫/台阶差异化 reward。

相关代码：`agent_ppo/feature/reward_process.py`。
共享特征代码：`agent_ppo/feature/height_scan_features.py`。

## Track 导航 Reward 预留

新增 Track/导航相关 reward：

- `_reward_goal_progress`
- `_reward_heading_to_goal`
- `_reward_navigation_time`
- `_reward_navigation_termination`

同时更新 `_reward_termination`，如果环境提供 `goal_reached`，成功到达目标不会被当作失败终止惩罚。

## 兼容旧模型续训

本轮保留并利用 `agent_ppo/agent.py` 中已有的 partial checkpoint loading 机制：

- 旧模型 301/316 输入可以部分加载到新增输入维度模型。
- 形状不一致的参数会复制重叠切片。
- 新增输入权重保持初始化值。

因此 `FWWB_STAGE=maze` 可以从旧 locomotion/stair checkpoint 续训，但第一次训练需要关注日志中是否有 partial load 提示。

## 推荐运行方式

Standard 迷宫软切换训练：

```bash
FWWB_STAGE=maze python train_test.py
```

Track 导航预留阶段：

```bash
FWWB_STAGE=nav python train_test.py
```

回到 locomotion：

```bash
FWWB_STAGE=locomotion python train_test.py
```

## 已做静态验证

已通过：

```bash
python3 -m py_compile \
  agent_ppo/conf/conf.py \
  agent_ppo/agent.py \
  agent_ppo/feature/policy_observation_process.py \
  agent_ppo/feature/critic_observation_process.py \
  agent_ppo/feature/reward_process.py \
  agent_ppo/workflow/train_workflow.py
```

已检查：

- `train_env_conf_standard_locomotion.toml` Standard terrain sum = `1.0`
- `train_env_conf_standard_maze.toml` Standard terrain sum = `1.0`
- `training_schedule.terrain.initial` sum = `1.0`
- `training_schedule.terrain.target` sum = `1.0`
- `training_schedule.terrain.target.maze = 0.30`
- `training_schedule.terrain.target.pyramid_stairs_inv = 0.30`
- `train_env_conf_track_nav.toml` 的 Track `sub_terrains` 最后一项为 `open_entry_maze`

## 尚未做的验证

未运行 `python train_test.py`，因为本地普通环境通常缺少 KaiwuDRL / Isaac Sim 运行依赖。

进入比赛环境后优先观察：

- `[TrainingSchedule] apply ...` 是否按 `apply_interval = 20` 打印。
- `Policy observation dim mismatch` / `Critic observation dim mismatch` 是否出现。
- 旧 checkpoint partial load 是否正常。
- 迷宫比例逐渐升至 `0.30` 后，机器人是否在墙前出现持续卡住或原地转圈。
- 台阶场景下 `_reward_feet_clearance` 是否提升过阶稳定性，且迷宫墙前没有被误当作台阶强行抬脚。

## 本轮涉及文件

修改：

- `agent_ppo/agent.py`
- `agent_ppo/conf/conf.py`
- `agent_ppo/feature/policy_observation_process.py`
- `agent_ppo/feature/critic_observation_process.py`
- `agent_ppo/feature/reward_process.py`
- `agent_ppo/workflow/train_workflow.py`

新增：

- `agent_ppo/conf/train_env_conf_standard_maze.toml`
- `agent_ppo/conf/train_env_conf_track_nav.toml`
- `docs/change_records/2026-05-11_wkn_maze_training_sync.md`

之前从其他分支同步并保留的配置文件：

- `agent_ppo/conf/train_env_conf_standard_stairs.toml`
- `agent_ppo/conf/train_env_conf_standard_stair_inv_finetune.toml`
- `agent_ppo/conf/train_env_conf_standard_standard_replay.toml`
