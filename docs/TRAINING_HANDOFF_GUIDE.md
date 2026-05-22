# 四足机器狗训练与代码交接指南

本文档面向下一位接手者：假设你不了解此前的实验历史，甚至不熟悉四足机器人强化学习。读完后，你应该能知道当前仓库在训练什么、为什么这样设计、代码入口在哪里、奖励函数各自负责什么、监控曲线应该怎么看，以及接手后第一天该做哪些检查。

更偏资产清单的交接说明见 `docs/HANDOFF.md`。本文更偏训练逻辑和代码讲解。

## 1. 当前训练在做什么

### 1.1 任务背景

这是腾讯开悟 `legged_robot_competition_26` 的四足机器人任务。机器人是 Unitree Go2，仿真环境是 Isaac Sim / Isaac Lab，训练框架走 KaiwuDRL + PPO。

当前主线不是 `agent_diy`，而是：

```text
agent_ppo
```

当前训练模式是 Standard 模式。Standard 模式不是导航到某个目标点，而是在混合地形中尽可能稳定穿越地形。平台分数大致看：

```text
总分 = 前进/通过能力 + 时间 + 能耗 + 姿态
```

训练 reward 和平台总分不是同一个东西。训练 reward 是我们写在 TOML 里的学习信号；平台分数是比赛系统用于评估的监控指标。

### 1.2 当前训练目标

当前阶段是一个“nan10 楼梯桥接训练”：

```text
从 nan10 系列 checkpoint 继续训练，
补强高难度台阶 / 反台阶能力，
同时避免 HJC 版本出现的沿对角线/边界上楼梯路径依赖。
```

更具体地说，目标是：

- 能稳定上下高难度台阶，尤其是 L7-L9。
- 不在台阶前明显转弯，不依赖两条边交界处的捷径。
- 维持自然步态，不出现明显跛脚、单腿画圈、前脚跨太远、后脚卡住。
- 下台阶时不要因为速度过快而倾覆。
- 姿态、能耗、平滑性要逐步收敛，但不能压制上楼梯动作。

### 1.3 当前阶段不是从零训练

当前代码是为“继续训练”设计的，不是从随机初始化开始训练。模型二进制 checkpoint 不在本仓库里，需要在腾讯平台模型列表中手动选择继续训练的起点。

近期讨论中常见的候选包括 nan10/nan36 系列中间 checkpoint。接手者需要根据平台模型列表确认具体使用哪个模型。

## 2. 当前活跃配置快照

### 2.1 入口配置

当前活跃阶段在：

```text
agent_ppo/conf/conf.py
```

关键配置：

```python
Config.CURRENT = Nan10StairBridgeConfig
```

该阶段会加载：

```text
agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml
```

当前 PPO 关键参数：

```text
lr = 1.5e-5
min_lr = 1e-5
max_lr = 2e-4
num_learning_epochs = 2
num_mini_batches = 4
num_steps_per_env = 48
model_save_interval = 50
```

`model_save_interval = 50` 表示平台每 50 个训练 episode 保存一次模型。分析模型潜力时，应重点看这些 50 轮保存点附近的曲线。

### 2.2 地形配置

当前 Standard 地形配置：

```text
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 9
```

当前地形比例：

```text
pyramid_slope      = 0.10
pyramid_slope_inv  = 0.10
pyramid_stairs     = 0.40
pyramid_stairs_inv = 0.40
maze               = 0.00
```

含义：

- 坡和反坡各保留 10%，用于保持基础步态和地形泛化。
- 台阶和反台阶各 40%，当前训练重心明确放在上下楼梯。
- 迷宫关闭，因为当前阶段不是 Track 导航训练。

### 2.3 episode 和速度命令

当前 episode 时长：

```text
episode_length_s = 30
```

速度命令重采样：

```text
resampling_time = [12.0, 18.0]
```

当前命令范围：

```text
lin_vel_x   = [0.0, 0.8]
lin_vel_y   = [-0.35, 0.35]
ang_vel_yaw = [-0.40, 0.40]
```

注意：这里的命令是速度跟踪命令。策略观测中会看到 `(vx, vy, wz)`，策略需要让机器人在自身坐标系中跟随这些速度。

### 2.4 command mix

当前启用了 PPO 侧 command mix：

```text
enabled = true
spin_only_ratio = 0.00
vx_only_ratio = 0.50
vx_vy_only_ratio = 0.00
full_ratio = 0.50
```

含义：

- 50% 环境只训练 `vx` 前进，不给 `vy/wz`。
- 50% 环境训练完整 `vx/vy/wz`。
- 不再使用 spin-only 和 vx-vy-only。

这么做的原因：

- 只给 `vx` 的评估场景里，机器人不应该自己乱转。
- 但未来迁移到更复杂任务时，机器人仍需要一定全方向命令能力，所以保留 50% full。

代码位置：

```text
agent_ppo/feature/command_mix.py
agent_ppo/workflow/train_workflow.py
```

训练日志中应能看到：

```text
[CommandMix]
[CommandMixMonitor]
```

如果 `runtime_seen=1` 且比例接近配置值，说明 command mix 实际生效。

### 2.5 fine tune schedule

当前启用线性调度：

```text
[fine_tune_schedule]
enabled = true
transition_seconds = 5400.0
fallback_transition_episodes = 540
update_interval_episodes = 10
```

`5400s` 是 90 分钟。它会把部分 reward 权重从 `reward_initial` 线性过渡到 `reward_target`。

注意一个历史经验：

```text
不要把一个需要启用的奖励从 weight = 0 调度到非 0。
```

原因是平台/Isaac reward manager 可能只在环境构建时注册非零 reward term。历史实验中，从 0 过渡到非 0 的项存在“看起来配置了，但运行时没有真正生效”的风险。因此现在需要用的奖励项都保持固定非零初值。

### 2.6 velocity curriculum

当前仍保留 `VelocityCurriculum` 代码，但 TOML 里的 4 个速度 stage 都是同一组范围：

```text
lin_vel_x   = [0.0, 0.8]
lin_vel_y   = [-0.35, 0.35]
ang_vel_yaw = [-0.40, 0.40]
```

也就是说：当前阶段实际不是逐步加速训练，而是固定速度范围训练。保留课程代码主要是为了监控和以后恢复分阶段速度训练。

## 3. 强化学习和观测动作的基本概念

### 3.1 策略网络做什么

每一步仿真中，策略网络接收观测 `obs`，输出 12 维动作：

```text
actions.shape = (num_envs, 12)
```

这 12 维对应 Go2 的 12 个关节目标偏移。动作范围是 `[-1, 1]`，环境会用 `action_scale = 0.25` 映射到 PD 控制目标。

### 3.2 观测是什么

当前 policy observation 是 301 维：

```text
obs = proprio(45) + height_scan(256)
```

其中：

- `0:3`：机身角速度。
- `3:6`：重力方向投影。
- `6:9`：速度命令 `(vx, vy, wz)`。
- `9:21`：关节位置。
- `21:33`：关节速度。
- `33:45`：上一帧动作。
- `45:301`：16x16 高度计扫描。

critic observation 是 316 维，比 actor 多一些 privileged 信息，例如机身线速度和关节力矩。

相关代码：

```text
agent_ppo/feature/policy_observation_process.py
agent_ppo/feature/critic_observation_process.py
```

不要随意改观测维度。如果改，必须同步修改 actor/critic 输入维度和断言。

## 4. 代码结构导览

### 4.1 平台加载入口

```text
agent_ppo/agent.py
```

平台会从这里加载 PPO agent。这个文件连接算法、模型、观测处理和训练 workflow。

### 4.2 阶段和超参

```text
agent_ppo/conf/conf.py
```

这里定义：

- 当前训练阶段。
- actor/critic 输入输出维度。
- PPO 学习率、epoch、mini-batch、保存间隔。
- 根据 `task_type` 选择 TOML 配置。

### 4.3 当前训练 TOML

```text
agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml
```

这是当前最重要的配置文件。它定义：

- 地形比例。
- episode 时长。
- 速度命令范围。
- command mix。
- 所有 reward 权重和参数。
- fine tune schedule。
- velocity curriculum。

### 4.4 奖励函数

```text
agent_ppo/feature/reward_process.py
```

TOML 中 `[rewards.xxx]` 有两类来源：

- 自定义奖励：通常对应这里的 `_reward_xxx`。
- Isaac Lab / 基类内置奖励：例如 `flat_orientation`、`ang_vel_xy` 等，可能不在本文件里实现。

例如：

```text
[rewards.feet_clearance]
```

对应：

```python
def _reward_feet_clearance(...)
```

如果接手者新增自定义奖励，应先在这里实现 `_reward_<name>`，再在 TOML 中启用 `[rewards.<name>]`。

### 4.5 command mix

```text
agent_ppo/feature/command_mix.py
```

这里把环境原始采样的速度命令改写成当前训练使用的混合命令。注意它作用于 PPO 侧观测和自定义 reward 读取到的命令，而不是改 Isaac 原生命令生成器本身。

### 4.6 训练循环和监控

```text
agent_ppo/workflow/train_workflow.py
```

这里负责：

- env reset / step。
- PPO rollout。
- fine tune schedule 应用。
- velocity curriculum。
- command mix 调试统计。
- height scan 门控统计。
- 监控指标上报。

### 4.7 监控面板

```text
agent_ppo/conf/monitor_builder.py
```

这里定义平台上能看到的自定义面板。注意有些 runtime debug 指标在历史上不稳定，当前策略是尽量保留能稳定上报的 reward-term 面板和官方指标。

### 4.8 平台曲线读取和后处理

```text
arena_frontend_monitor/
```

常用入口：

```bash
arena_frontend_monitor/manual_metric_recorder.sh <monitor_url>
```

该工具用于手动滚动平台监控页时记录曲线数据，结束后做 AI 可读后处理。

## 5. 奖励函数分组解释

下面只解释当前 TOML 中启用的主要奖励。权重有正有负：正值是奖励，负值是惩罚。

### 5.1 速度和方向跟踪

#### `track_lin_vel_xy`

鼓励机器人实际 XY 速度跟随命令 XY 速度。它是主要的移动能力奖励。

当前权重：

```text
initial 2.40 -> target 2.70
```

风险：

- 太强会让机器人为了速度硬冲，尤其下台阶容易失稳。
- 太弱会导致怯战、timeout、不愿上楼梯。

#### `track_ang_vel_z`

鼓励机器人跟随 yaw 角速度命令。

当前权重：

```text
initial 1.33 -> target 1.40
```

它不是 heading 命令，而是持续角速度跟踪。当前没有改 Isaac 命令层为 heading 模式。

#### `command_direction_progress`

奖励实际速度在命令方向上的投影。通俗说：不要只是动，要沿命令方向动。

当前权重：

```text
initial 0.70 -> target 0.55
```

它比单纯 `track_lin_vel_xy` 更强调方向一致性。

#### `command_direction_deviation`

惩罚实际运动方向和命令方向之间的角度偏差。采用二次型惩罚，偏差越大罚得越重。

当前权重：

```text
initial -0.52 -> target -0.62
```

这是用来抑制“只给 vx 时自己慢慢转斜”的关键项之一。

#### `command_path_progress`

奖励沿命令路径的世界坐标进展。`vx_only` 时使用命令段/reset anchor 方向；full command 时使用当前命令方向。

当前权重：

```text
initial 0.40 -> target 0.28
```

历史经验：这个奖励不能太强，否则下台阶会鼓励硬冲；但完全没有，又容易 timeout 或不愿前进。

#### `commanded_stall_penalty`

惩罚长时间不沿命令方向推进。它不是每一步都鼓励前进，而是在持续停滞后才惩罚。

当前权重：

```text
固定 -0.10
```

这是为了减少“宁愿 timeout 也不上楼梯”的行为。

### 5.2 抬脚、落脚和小步上楼梯

#### `feet_clearance`

密集抬脚奖励。它是当前最基础的抬腿信号。

当前权重：

```text
initial 0.40 -> target 0.28
```

为什么 target 降低：历史上纯抬腿过强会导致抬脚过高、卡住、步态变形。

#### `height_scan_feet_clearance`

基于 16x16 高度计的台阶区域抬脚奖励。它只在高度计判断前方像台阶时增强抬脚。

当前权重：

```text
initial 0.28 -> target 0.22
```

它比 `feet_clearance` 更有针对性，但仍然不能过强。

#### `stair_forward_foot_placement`

鼓励脚在上台阶时向前落到合理位置。它偏 sparse，因为依赖触地瞬间。

当前权重：

```text
initial 0.33 -> target 0.38
```

目的：从“只会抬腿”推进到“抬腿后落准”。

#### `stair_swing_step_targeting`

摆腿中期目标奖励。它比落脚奖励更 dense，鼓励脚在摆动过程中就朝下一阶合理区域移动。

当前权重：

```text
initial 0.065 -> target 0.16
```

目的：让机器人学会小步、逐阶上楼，而不是前脚跨很远。

### 5.3 防止错误步态

#### `stair_over_clearance_penalty`

惩罚台阶上抬脚过高。

当前权重：

```text
initial -0.13 -> target -0.20
```

目的：防止“上台阶就一直高抬腿”，减少卡住和能耗浪费。

#### `stair_stride_length_penalty`

惩罚台阶上步幅过大、前脚伸太远、后脚落后太多。

当前权重：

```text
initial -0.030 -> target -0.10
```

这是针对“前脚跨两个台阶，后脚还在地面”的问题。

#### `stair_support_continuity_penalty`

惩罚台阶上支撑断档，例如接触脚太少、只有前脚或只有后脚支撑。

当前权重：

```text
initial -0.020 -> target -0.075
```

它不强制固定步态相位，只要求基本支撑连续性。

#### `feet_air_time`

脚部滞空奖励。它有助于迈步，但过强会导致单腿长期悬空或画圈。

当前权重：

```text
initial 0.45 -> target 0.22
```

历史经验：这个奖励不适合长期维持过高。

#### `air_time_variance_penalty`

惩罚四条腿滞空时间差异过大。

当前权重：

```text
initial -0.78 -> target -0.88
```

用于抑制单腿画圈、跛脚、快慢步。

### 5.4 台阶安全

#### `stair_base_clearance_penalty`

惩罚机身/腹部在台阶边缘附近离地过低。

当前权重：

```text
initial -0.08 -> target -0.14
```

这个奖励是为了处理“明明正着上楼梯，但身体趴太低导致卡台阶/失败”的问题。

#### `down_stair_speed_safety`

下台阶时惩罚冲太快、垂直速度过大、pitch/roll 角速度过大。

当前权重：

```text
initial -0.10 -> target -0.16
```

目的：防止下楼梯时失控前冲。

#### `down_stair_touchdown_safety`

惩罚下台阶落脚冲击过大。

当前权重：

```text
initial -0.16 -> target -0.22
```

目的：让下台阶落脚更柔和。

### 5.5 防转向和防路径依赖

#### `uncommanded_yaw_rate`

当命令要求平移但没有 yaw 时，惩罚机器人自己产生 yaw 角速度。

当前权重：

```text
initial -0.095 -> target -0.12
```

#### `uncommanded_heading_drift`

当命令段中没有 yaw 时，惩罚朝向相对命令段 anchor 漂移。

当前权重：

```text
initial -0.095 -> target -0.12
```

#### `stair_edge_normal_alignment`

用高度计估计台阶边沿方向，惩罚沿台阶边沿走。它不是世界轴奖励，不绑定固定 x 轴。

当前权重：

```text
固定 -0.035
```

目的：减少沿两条边交接线/对角线走捷径。

#### `pivot_turning`

惩罚低线速度、高 yaw、脚在地上蹭着原地旋转。

当前权重：

```text
initial -0.22 -> target -0.25
```

### 5.6 姿态、能耗和平滑

#### `flat_orientation`

惩罚机身偏离水平。

当前权重：

```text
initial -1.25 -> target -1.35
```

注意：上楼梯需要一定 pitch/roll，自身不能无限强。

#### `correct_base_height`

惩罚机身高度偏离目标。

当前权重：

```text
initial -0.57 -> target -0.42
```

当前 target 变轻，是因为楼梯上机身高度本来会变化，过强会抑制上台阶。

#### `ang_vel_xy`

惩罚 pitch/roll 角速度。

当前权重：

```text
initial -0.32 -> target -0.38
```

它用于抑制姿态抖动，但不能太强，否则会压制楼梯动作。

#### `energy`

能耗惩罚。

当前权重：

```text
initial -2e-5 -> target -3e-5
```

当前阶段优先保证通过率，能耗不能压过楼梯动作链。

## 6. 当前训练设计背后的历史经验

### 6.1 为什么从 nan10 继续

nan10-8750 的基础能力较好，能走大多数地形，且没有明显学会沿对角线爬楼梯的路径依赖。但它只能上较简单的台阶，尤其反台阶高难度不足。

当前方案是在 nan10 上补高难度台阶能力，而不是直接继承 HJC 的行为。

### 6.2 HJC 代码的优点和问题

HJC 代码训练出了较强上楼梯能力，但观察到明显倾向于选择对角线/边界路径。当前代码吸收了“台阶动作塑形”的思想，但避免使用可能绑定世界轴或固定台阶方向的奖励。

### 6.3 WK 代码的启发

WK 代码对步态和基础移动更自然，有可参考的 feet clearance / command tracking 思路。当前代码借鉴了更直接的命令跟踪和步态塑形，但没有全量照搬权重。

### 6.4 已放弃或谨慎使用的方向

以下方向当前不建议重新启用：

- 未验证的 terrain label gate 专用 `inv_stair_climb_action`。
- 世界 x 轴 `forward_velocity` 式奖励。
- 固定楼梯轴/世界轴 alignment。
- 从 0 调度到非 0 的 reward term。
- 大幅提高纯 `feet_air_time`。
- 只靠 `track_lin_vel_xy` 解决高难度楼梯。

## 7. 监控日志怎么看

### 7.1 不要只看 reward_mean

`reward_mean` 是训练 reward，不是比赛总分。它可能因为某个奖励项变大而上涨，但通过率未必提升。

当前最重要的指标：

- `completed_count`
- `timeout_count`
- `abnormal_count`
- 非超时失败：`abnormal_count - timeout_count`
- 各地形 L0-L9 通过情况
- `total_score`、`forward_score`、`step_score`
- `pose_score`、`energy_score`
- 关键 reward term 的实际贡献

### 7.2 为什么曲线有锯齿

平台曲线经常出现周期性锯齿。原因通常是不同难度和不同 episode 终止步数混在一起：

- 简单地形可能跑满更多步。
- 高难度地形可能提前失败。
- 不同批次 episode 结束时机不同。

所以判断模型好坏时，不要只看一个尖峰或谷底，要看平滑后的趋势和保存点附近的综合表现。

### 7.3 如何选择 checkpoint

当前保存间隔是 50 轮。选择继续训练起点时，建议：

1. 优先看 50 轮结尾的模型。
2. 不只看总分，看高难度台阶/反台阶通过情况。
3. 看 timeout 是否上升。如果 timeout 上升，可能模型开始怯战。
4. 看非超时失败是否上升。如果上升，可能动作激进、姿态崩或下台阶冲太快。
5. 看姿态和能耗是否严重恶化。
6. 结合视频观察是否出现斜走、趴低、前脚跨太远、单腿异常。

### 7.4 手动读取平台曲线

工具目录：

```text
arena_frontend_monitor/
```

推荐使用：

```bash
arena_frontend_monitor/manual_metric_recorder.sh <monitor_url>
```

脚本结束后会生成 AI 可读的 summary、平滑曲线、分块分析。最新重要长训分析已写入：

```text
docs/change_records/2026-05-22_nan10_longrun_reward_regression_analysis.md
```

## 8. 接手后第一天 checklist

### 8.1 先读这几个文件

```text
docs/TRAINING_HANDOFF_GUIDE.md
docs/HANDOFF.md
docs/change_records/2026-05-22_nan10_longrun_reward_regression_analysis.md
agent_ppo/conf/conf.py
agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml
agent_ppo/feature/reward_process.py
agent_ppo/workflow/train_workflow.py
agent_ppo/conf/monitor_builder.py
```

### 8.2 确认当前代码状态

运行：

```bash
git status --short
```

如果存在未提交改动，先确认这些改动是否就是最新训练逻辑。不要只拿旧 commit 训练。

### 8.3 同步到容器

先 dry-run：

```bash
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dest /workspace/code --dry-run
```

确认文件列表后再 apply：

```bash
CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dest /workspace/code --apply --py-compile
```

注意：

- routine sync 不应上传 `agent_diy`、大日志、runtime、backup、token。
- 如果 RPC 不通，在容器内运行 `bash agent_diy/codex_rpc_bridge/start_rpc.sh`。

### 8.4 短训验证

新接手不要直接 5h 长训。建议先做 15-30 分钟短训，确认：

- 训练能正常启动。
- `[CommandMix]` 和 `[CommandMixMonitor]` 显示比例合理。
- `FineTuneSchedule` 有 apply 日志。
- reward 面板不是全 0。
- 没有 `NameError`、reward manager 注册异常、配置校验错误。

### 8.5 再决定是否长训

如果短训中出现以下问题，不要长训：

- L0-L3 大量失败。
- timeout 明显上升。
- 非超时失败明显上升。
- 高度计相关奖励全为 0。
- command mix `runtime_seen=0`。
- 机器人视频出现严重跛脚、单腿画圈、原地转向、趴低卡台阶。

## 9. 常见问题

### Q1: 为什么不直接大幅增加抬腿奖励？

因为历史上模型已经出现过“只会高抬腿，但落不准、卡住、失衡”的问题。当前目标不是单纯抬高脚，而是完整动作链：

```text
识别台阶 -> 抬够 -> 前伸 -> 落准 -> 支撑承重 -> 身体跟上 -> 控速通过
```

### Q2: 为什么不强迫机器人正对台阶？

任务要求不是必须正面上台阶，而是以各种合理角度穿越台阶。但不能让模型学会固定沿对角线/边界走捷径。因此当前只惩罚“沿边走”和“无指令转向”，不绑定世界坐标轴。

### Q3: 为什么不用 terrain label gate？

历史测试中，terrain label gate 在实际运行链路里无法稳定验证。当前更可靠的方案是基于 16x16 height scan 做局部几何判断。

### Q4: 为什么 velocity curriculum 还在，但速度 stage 都一样？

当前阶段希望固定速度范围收敛高难度楼梯动作，不再同一轮里同时改变速度难度。代码保留是为了以后恢复课程和保留监控。

### Q5: 为什么需要 command mix？

评估中经常只给 `vx`，机器人不应该自己转弯。但训练也不能完全没有 `vy/wz`，否则未来迁移到更复杂命令会弱。因此当前 50% `vx_only`，50% `full`。

## 10. 修改奖励时的原则

修改前先回答三个问题：

1. 这个 reward 是为了修复哪一个具体失败模式？
2. 它会不会和已有 reward 冲突？
3. 它的量级能不能从监控面板中看到？

推荐策略：

- 一次只改少量关键权重。
- 用短训验证，再长训。
- 优先固定非零启用 reward，不要从 0 调度到非 0。
- 不要把塑形奖励做得比主任务 reward 更大。
- 高难度台阶能力不足时，优先看动作链是否断了，而不是单纯加速度奖励。

## 11. 交接包生成

安全交接包脚本：

```bash
python3 scripts/build_handoff_bundle.py --dry-run
python3 scripts/build_handoff_bundle.py --output /tmp/fwwb-rl-dog-handoff.tar.gz
```

默认会包含：

- 当前训练代码。
- 交接文档。
- RPC/sync/monitor 工具。
- 精选监控摘要。
- nan10/hjc/wk 参考代码。

默认不会包含：

- `.git/`
- token
- RPC runtime
- raw monitor runtime 大目录
- checkpoint 二进制
- `__pycache__`

checkpoint 需要在腾讯平台模型列表中另行说明。
