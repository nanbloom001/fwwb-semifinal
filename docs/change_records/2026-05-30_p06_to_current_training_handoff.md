# p06 之后改动、训练思路与 xtrack9 继承审计

本文记录从 `代码归档/p06` 覆盖当前仓库之后，到当前版本为止的主要代码变化、当前训练思想，以及 `p06` 在排除官方模板后仍与 `代码归档/xtrack9` 相似的部分。本文用于后续交接、训练复盘和代码清理。

注意：本文中的“相似”不是法律或官方查重结论，只是基于本地文件存在性、文本相似度和是否属于官方模板的工程审计结果。后续优化应以行为正确、训练稳定、可维护为主，不建议做无语义的机械改名。

## 1. 当前主线

当前主线是 `agent_ppo` Track 训练，核心链路如下：

- `agent_ppo/conf/conf.py`
- `agent_ppo/conf/train_env_conf_track_nav.toml`
- `agent_ppo/feature/reward_process.py`
- `agent_ppo/feature/terrain_gate.py`
- `agent_ppo/feature/phase_command_adapter.py`
- `agent_ppo/feature/eval_command_adapter.py`
- `agent_ppo/workflow/train_workflow.py`
- `agent_ppo/conf/monitor_builder.py`

p06 之后，当前仓库已经从单纯继承 p06 的版本，逐步变成以下方向：

```text
完成率优先
建议速度辅助
直接评分奖励逐步替代辅助塑形奖励
重复度高的塑形奖励优先被直接奖励覆盖
独特监督函数低比例保留，不轻易删除
奖励平衡必须按日志中的实际贡献量级计算
高难度完成保护
迷宫/近终点安全监督
训练和评估侧 command 口径尽量一致
```

## 2. p06 之后的主要改动

### 2.1 旧实验模块清理

p06 中继承或保留了较多旧实验模块。当前仓库已经删除或不再使用其中一批：

- `agent_ppo/feature/command_mix.py`
- `agent_ppo/feature/goal_observation.py`
- `agent_ppo/feature/height_scan_features.py`
- `agent_ppo/feature/motion-primitives.py`
- `agent_ppo/feature/motion_primitives.py`
- `agent_ppo/feature/navigation.py`
- `agent_ppo/feature/path_planner.py`
- `agent_ppo/feature/track_observation_features.py`

当前新增/保留的替代模块更偏职责拆分：

- `goal_features.py`：目标相关特征。
- `phase_semantics.py`：阶段/地形语义。
- `phase_command_adapter.py`：训练侧建议速度写入。
- `eval_command_adapter.py`：评估侧建议速度口径对齐。
- `terrain_gate.py`：sticky 地形门控、建议速度、门控监控。
- `velocity_curriculum.py`：标准模式速度课程能力保留。
- `isaac_env_bridge.py`：环境访问和解包相关辅助。

这轮清理的目的不是简单减少文件，而是把训练实际依赖的机制集中到少数可解释入口中。

### 2.2 建议速度机制

当前版本保留建议速度机制，但语义从“覆盖平台命令”调整为“给模型一个稳定速度意图”。训练和评估都尽量使用一致口径：

```toml
phase_command_enabled = true
worker_phase_command_enabled = true
gate_test_mode = "sticky"
terrain_sticky_source = "nan"
gate_speed_advice_enabled = true

suggested_speed_fallback = 0.68
pre_maze_lin_vel_x = [0.75, 0.75]
slope_lin_vel_x = [0.75, 0.75]
stairs_lin_vel_x = [0.60, 0.60]
maze_lin_vel_x = [0.60, 0.60]
phase_maze_goal_dist_gate = 8.5
phase_maze_distance_mode = "longitudinal"
```

核心思想：

- 平地和缓坡可以给略快的建议速度。
- 台阶和迷宫使用更稳的建议速度。
- 速度建议写入 obs command 槽，策略仍然输出关节动作。
- 评估侧使用 `eval_command_adapter.py` 尽量复现训练侧的速度意图，避免训练/评估命令语义断裂。

### 2.3 奖励函数重平衡

当前奖励设计逐步从“辅助塑形函数堆叠”转向“直接评分相关奖励主导，塑形函数辅助”。

训练思想：

```text
1. Track 首要目标是 completion ratio。
2. 在完成率稳定的基础上，按评分公式优化 time / posture / energy。
3. 尽量使用与平台评分强相关或同公式的直接奖励。
4. 辅助塑形函数只保留必要监督，不再无限叠加。
5. 如果直接奖励可以表达同一目标，应逐步替代或弱化对应塑形项。
6. 如果某个辅助项与直接评分项高度重复，优先被直接评分项覆盖。
7. 如果某个辅助项提供独特监督，不能直接删除，只能低比例保留。
8. 所有权重调整必须参考日志中的实际 reward 均值/占比，而不是只看 TOML 权重。
```

当前直接评分相关项：

- `task_complete`：直接对应完成。
- `pose_score_formula`：与姿态评分公式对齐。
- `energy_score_formula`：与能耗评分公式指数核对齐。
- `goal_velocity_projection`、`forward_heading_velocity`、`track_lin_vel_xy`、`command_speed_advantage`：时间/推进代理，不能完全等同 time score，但用于推动完成和加速。

当前逐步弱化或保持克制的辅助项：

- `action_rate`
- `action_smoothness`
- `joint_acc`
- `joint_torques`
- `dof_vel`
- `flat_orientation`
- `correct_base_height`
- `score_guidance`
- 部分 maze/near-goal shaping 项

最新主线权重方向：

```toml
task_complete = 285.0
difficulty_pressure_complete = 30.0

goal_velocity_projection = 2.30
forward_heading_velocity = 0.74
track_lin_vel_xy = 1.08
command_speed_advantage = 0.75

pose_score_formula = 2.70

energy_score_formula = 1.55
energy = -4.8e-5
joint_torques = -2.4e-5

undesired_contacts = -0.62
termination = -9.0
```

这个方向不是让训练 reward 精确等于平台分数，而是让主要梯度更接近平台目标：

```text
完成率优先
time ≈ posture > energy
辅助塑形服务于完成率和安全，不抢主目标
```

### 2.4 昨晚到现在形成的调参原则

本轮从 p06 之后的训练复盘中，逐步形成了更明确的调参方法：

#### 原则 A：重复度高的项优先覆盖

如果一个辅助塑形函数和平台直接评分项高度重叠，就优先用直接评分项覆盖它，而不是继续叠加两个方向相近的奖励。

例子：

- `pose_score_formula` 与姿态评分公式直接相关，因此它可以逐步覆盖部分姿态塑形压力。
- `energy_score_formula` 与能耗评分公式直接相关，因此它可以逐步覆盖部分 `joint_acc`、`action_rate`、`action_smoothness`、`joint_torques` 类间接能耗压力。
- `task_complete` 直接表达完成，因此完成相关奖励应围绕它组织，旧的 `reach_goal` 类重复稀疏奖励可以合并或删除。

这条原则的目的是减少重复梯度，让模型更接近平台最终评分口径。

#### 原则 B：独特监督项低比例保留

不是所有塑形函数都应该被直接奖励替代。某些奖励虽然不是平台公式，但提供了直接评分项无法表达的中间监督，需要保留，只是权重不能太大。

例子：

- `near_goal_finish_drive`：解决接近终点时最后一段推进。
- `near_goal_retreat_penalty`：防止进入终点附近后反向离开。
- `near_goal_circling_penalty`：抑制终点附近绕圈。
- `goal_miss_penalty`：惩罚接近终点后又错过。
- `wall_collision`、`wall_stall_penalty`、`long_non_foot_contact`：处理迷宫贴墙、撞墙、卡墙，这些不是评分公式本身，但影响 completion ratio。
- `goal_heading_alignment`、`goal_distance`：给迷宫方向和距离提供稳定学习信号。

这些项不适合被完全删除。后续优化方向是低比例保留、减少重复、看日志决定是否有效。

#### 原则 C：奖励平衡必须按日志实际贡献计算

不能只看 TOML 的 `weight`。不同奖励函数的原始输出尺度差异很大，因此必须看监控日志里的实际值，例如：

```text
reward_pose_score_formula
reward_energy_score_formula
reward_goal_velocity_projection
reward_track_lin_vel_xy
reward_task_complete
reward_difficulty_pressure_complete
```

本轮讨论中使用的基本方法是：

```text
1. 读取最新 session 的 all_metric_series_summary.json。
2. 去掉前 8 分钟震荡期或使用 last20_avg。
3. 按语义分组计算实际贡献：
   completion
   time_proxy
   pose_direct + pose_proxy
   energy_direct + energy_proxy
   maze_safety
4. 对比平台评分比例：
   Track = completion_ratio * (0.4 time + 0.4 posture + 0.2 energy)
5. 如果某组实际贡献明显过强，优先小幅回调。
6. 如果某组实际贡献明显过弱，优先增强直接评分项。
```

关键经验：

- `task_complete` 是稀疏奖励，日志均值天然小，不能简单用均值和连续型奖励一比一判断。
- `pose_score_formula` 原始输出接近 0 到 1，乘权重后容易成为大项。
- 时间没有完全等价的平台公式，只能用代理项；这些代理项同时承担“完成赛道”和“加速”的职责，不能按表面量级大幅削弱。
- 能耗优化优先增强 `energy_score_formula`，不要直接大幅增加动作平滑类惩罚，否则可能影响楼梯和迷宫动作能力。

#### 原则 D：不要一次引入太多变量

低难度时间鼓励、高难度保护、能耗增强、姿态回调、速度代理回调都可能影响训练趋势。最终选择不加入低难度时间鼓励，是为了让长训变量更少，便于判断：

```text
这轮主要验证：
1. 直接奖励重平衡是否改善评分项；
2. episode mean pressure 是否让高难度完成保护真正生效；
3. 完成率是否能保持。
```

### 2.5 高难度完成保护

p06 之后新增并迭代了 `difficulty_pressure_complete`。当前实现已经从“完成瞬间 pressure”改成“episode mean pressure”：

```text
每个 env 一局内累计：
  energy_score_sum
  pose_score_sum
  step_count

完成时：
  episode_energy_mean = energy_score_sum / step_count
  episode_pose_mean = pose_score_sum / step_count

相对近期 EMA 基线越低，pressure 越高。
只有 complete 时给 pressure bonus。
done 后重置该 env 统计。
```

这项的语义是：如果一局整体表现出高难度特征，但最终完成，则给很小的额外完成保护。它不读取真实地形等级，也不鼓励坏姿态或高能耗本身。

当前参数：

```toml
threshold = 0.6
ema_decay = 0.995
warmup_steps = 80
min_std = 0.02
std_scale = 2.0
energy_weight = 0.6
pressure_start = 0.55
curve_power = 1.5
```

## 3. 当前训练思路

### 3.1 从昨晚到现在的决策脉络

本轮讨论不是一次性确定权重，而是逐步收敛出以下路线：

1. 先确认 p06 之后训练效果没有完全崩坏，当前策略有继续优化价值。
2. 再确认单纯继续堆速度奖励风险较高，因为时间分早期虚高会被完成率提升后的慢样本稀释。
3. 然后把优化目标从“继续追速度”调整为“按平台评分比例重平衡 reward”。
4. 进一步确认直接评分奖励更适合作为主梯度：`task_complete`、`pose_score_formula`、`energy_score_formula`。
5. 对与直接评分高度重叠的辅助塑形项，采用“直接奖励覆盖，辅助项降权”的思路。
6. 对迷宫、近终点、撞墙、防绕圈这类直接评分公式不能表达的独特监督，采用“低比例保留”的思路。
7. 对高难度保护，不读取地形标签，而是用一局平均姿态/能耗相对 EMA 基线来判断高压力完成。
8. 最后决定不加入低难度时间鼓励，避免变量过多，先验证当前 reward balance。

因此，当前版本的重点不是“某一个 reward 加大”，而是：

```text
用日志计算实际贡献；
削弱重复塑形；
保留独特监督；
提高直接评分项；
保护完成率和高难度完成。
```

### 3.2 直接奖励逐步替代塑形奖励

后续调参的核心原则：

```text
如果某个辅助塑形项和平台评分直接项表达同一目标，
优先提高直接项、降低辅助项。
```

举例：

- 姿态方面，优先使用 `pose_score_formula`，谨慎保留 `ang_vel_xy`、`flat_orientation`、`correct_base_height` 作为稳定性辅助。
- 能耗方面，优先使用 `energy_score_formula`，而不是过度依赖 `joint_acc`、`action_rate`、`action_smoothness`。
- 时间方面没有完全等价的直接公式，只能用 `goal_velocity_projection`、`forward_heading_velocity`、`track_lin_vel_xy` 等代理项，但要避免它们过强导致高难度异常率上升。
- 完成率方面，`task_complete` 和终止/接触惩罚是核心，其他 maze shaping 只应辅助模型找到出口和避免撞墙。

### 3.3 为什么不继续堆低难度时间鼓励

曾讨论过使用姿态/能耗特征判断低难度，再额外给 1% 时间奖励。最终暂不加入，原因是：

- 动态归一化会让“低难度”定义漂移。
- 固定归一化虽然可解释，但需要额外短训验证是否误伤 L4-L6。
- 当前变量已经较多，再加低难度专属鼓励不利于判断长训效果。
- 时间代理当前并不弱，主要问题仍是高难度完成率和能耗/姿态平衡。

### 3.4 当前后续调参优先级

建议后续优先级：

1. 保持完成率，不让 L8/L9 abnormal 上升。
2. 观察 `difficulty_pressure_complete` 是否有小读数。
3. 能耗优化优先继续调整 `energy_score_formula`，不要大幅增加动作平滑类惩罚。
4. 如果时间明显变慢，优先恢复一部分 `goal_velocity_projection`，而不是加入新低难度奖励。
5. 如果姿态继续上升但能耗不升，考虑略降 `pose_score_formula` 或部分姿态辅助项。

## 4. p06 与 xtrack9 相似性审计方法

审计对象：

- 参考代码：`代码归档/xtrack9`
- p06：`代码归档/p06`
- 官方模板：`代码归档/最新版本官方原始代码`

审计原则：

```text
1. p06 与 xtrack9 完全相同，但也与官方高度一致：通常认为是模板继承，低风险。
2. p06 与 xtrack9 完全相同，官方没有该文件：高优先级继承点。
3. p06 与 xtrack9 高度相似，官方差异很大：中高优先级继承点。
4. p06 与 xtrack9 相似，但当前仓库已经删除或重写：记录为已处理。
```

## 5. p06 排除官方模板后的 xtrack9 继承点

### 5.1 高优先级：p06 与 xtrack9 完全相同且官方没有

这些文件在 p06 与 xtrack9 中完全一致，且官方模板中没有对应文件，属于最明确的继承/遗留内容：

| p06 文件 | 状态 | 当前仓库状态 | 建议 |
|---|---|---|---|
| `agent_ppo/codex_rpc_bridge_runtime/codex_file_rpc.py` | p06=xtrack9，官方无 | 当前主同步不上传 runtime | 不纳入提交；继续排除 |
| `agent_ppo/conf/eval_env_conf_track_nav_wide.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/conf/train_env_conf_standard_locomotion.keep_66_38.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/conf/train_env_conf_standard_stair_conservative.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/conf/train_env_conf_standard_stair_inv_finetune.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/conf/train_env_conf_standard_standard_fast.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/conf/train_env_conf_track_track_nav.toml` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/feature/goal_observation.py` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/feature/motion-primitives.py` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/feature/motion_primitives.py` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/feature/navigation.py` | p06=xtrack9，官方无 | 当前已无 | 已处理 |
| `agent_ppo/feature/path_planner.py` | p06=xtrack9，官方无 | 当前已无 | 已处理 |

结论：p06 中最明显的“官方无、xtrack9 有”的旧文件，当前大部分已经删除或不再同步，这是本轮清理中最有效的部分。

### 5.2 高优先级：p06 与 xtrack9 完全相同，但官方差异很大

这些文件官方有对应文件，但 p06 与 xtrack9 完全一致，且与官方差异较大：

| 文件 | p06 vs xtrack9 | p06 vs 官方 | 当前仓库状态 | 审核结论 |
|---|---:|---:|---|---|
| `agent_ppo/agent.py` | 1.000 | 0.523 | 当前已改，仍有部分评估 command adapter 逻辑 | 需要保留功能但继续说明/简化 |
| `agent_ppo/feature/policy_observation_process.py` | 1.000 | 0.160 | 当前已改，但仍受 301 维协议限制 | 不建议为差异化乱改观测 |
| `agent_ppo/feature/critic_observation_process.py` | 1.000 | 0.274 | 当前已改，但仍受 316 维协议限制 | 不建议为差异化乱改观测 |
| `agent_ppo/conf/train_env_conf_standard_locomotion.toml` | 1.000 | 0.423 | 当前仍完全相同 | 若不训练 Standard，建议排除同步或归档 |

其中最值得处理的是 `train_env_conf_standard_locomotion.toml`。当前主线是 Track，如果 Standard 配置不参与训练和提交，最干净的方案是从 routine sync 中排除，或移到归档目录，不作为当前交付代码的一部分。

`agent.py` 的继承点主要是评估流程、模型加载和 obs 处理流程。当前我们已经加入 `eval_command_adapter.py`，后续可以进一步把评估侧 command 语义写成清晰的团队实现说明，而不是保留 p06/xtrack9 风格的隐式逻辑。

`policy_observation_process.py` 和 `critic_observation_process.py` 不建议为了降低相似度做结构性改动，因为 301/316 维观测协议是硬约束。可以做的是增加清晰注释、删除无用分支、保持断言明确。

### 5.3 中高优先级：p06 与 xtrack9 高度相似，官方差异很大

这些文件不是完全相同，但仍高度相似，并且明显不是官方模板：

| 文件 | p06 vs xtrack9 | p06 vs 官方 | 当前仓库状态 | 建议 |
|---|---:|---:|---|---|
| `agent_ppo/conf/monitor_builder.py` | 0.962 | 0.165 | 当前已大量改动，但仍有旧面板结构痕迹 | 继续清理 empty 面板，按评分语义重组 |
| `agent_ppo/workflow/train_workflow.py` | 0.968 | 0.241 | 当前已改，加入 phase command 和监控逻辑 | 保留核心训练流程，整理新增监控入口 |
| `agent_ppo/feature/reward_process.py` | 0.838 | 0.043 | 当前继续大量修改，已偏离 xtrack9 | 按语义分层，减少旧函数残留 |

#### `monitor_builder.py`

p06 中该文件与 xtrack9 高度相似，且官方差异大，说明监控面板基本来自自定义实验体系。当前仓库已经继续改过，但仍有可清理点：

- 删除长期 empty 的旧奖励面板。
- 将面板按以下组重排：
  - 完成率与失败
  - 评分直接项
  - 时间代理
  - 能耗/姿态
  - 高难度保护
  - 建议速度门控
  - 迷宫安全
- 对英文残留面板名做统一命名，但不要影响 metric key。

#### `train_workflow.py`

p06 与 xtrack9 高度相似，但训练 workflow 本身也容易保留大量框架代码。当前仓库已经加入：

- `phase_command_adapter` 调用。
- 速度建议状态 reset。
- 监控数据扩展。
- 训练/评估性能优化相关处理。

后续建议：

- 不改 PPO rollout 主流程。
- 将“监控采样”和“phase command 写入”的调用封装得更清楚。
- 删除与当前配置无关的旧 debug 入口。

#### `reward_process.py`

p06 和 xtrack9 相似度仍较高，但当前仓库已经基于 p06 做了较多实际训练相关调整。后续不建议简单改函数名，而应继续做语义整理：

- 平台评分公式类：
  - `pose_score_formula`
  - `energy_score_formula`
- 完成保护类：
  - `task_complete`
  - `difficulty_pressure_complete`
- 时间/目标推进类：
  - `goal_velocity_projection`
  - `forward_heading_velocity`
  - `approach_goal`
  - `near_goal_finish_drive`
- 迷宫安全类：
  - `wall_collision`
  - `wall_stall_penalty`
  - `long_non_foot_contact`
- 旧/弱/空读数类：
  - 后续确认长期无读数后删除或从 TOML 关闭。

这类整理既能降低旧实验痕迹，也能提高后续调参效率。

### 5.4 低优先级：p06 与 xtrack9 相同，但官方也高度相似

这些文件虽然 p06 与 xtrack9 完全一致，但也接近官方模板，通常不应作为优先处理对象：

| 文件 | p06 vs xtrack9 | p06 vs 官方 | 建议 |
|---|---:|---:|---|
| `agent_ppo/algorithm/algorithm_ppo.py` | 1.000 | 0.942 | 保持不动 |
| `agent_ppo/model/actor_critic.py` | 1.000 | 0.960 | 保持不动 |
| `agent_ppo/feature/definition.py` | 1.000 | 0.911 | 保持不动或仅补注释 |
| 各级 `__init__.py` | 1.000 | 1.000 | 不处理 |

这些属于训练底座或模板骨架。为了差异化改 PPO、网络结构或空包初始化文件，风险大于收益。

## 6. 当前仓库相对 p06 已经处理掉的继承点

当前仓库相比 p06 已经处理的重点包括：

1. 删除旧导航/路径规划/动作混合类实验文件。
2. 用 `phase_command_adapter.py`、`eval_command_adapter.py`、`phase_semantics.py` 等更明确的模块替代隐式逻辑。
3. 大幅改写 `train_env_conf_track_nav.toml`，当前与 xtrack9 相似度已经很低。
4. 改造奖励权重，使其从速度推进主导，转向直接评分项和完成率保护。
5. 增加 `difficulty_pressure_complete` 并改为 episode mean pressure。
6. 清理一部分同步和容器垃圾文件，避免上传 runtime、日志、压缩包、缓存等无关文件。

## 7. 后续建议

### 7.1 训练侧

继续保持当前训练思路：

```text
完成率第一；
直接评分奖励逐步替代辅助塑形；
时间代理只做必要推进，不要继续无限加速；
高难度完成保护只做小权重辅助；
能耗优化优先使用 energy_score_formula，而不是大幅增加动作平滑惩罚。
```

### 7.2 代码侧

建议后续按低风险顺序处理：

1. 如果当前只训练 Track，排除或归档 `train_env_conf_standard_locomotion.toml`。
2. 清理 `monitor_builder.py` 中长期 empty 或语义重复的面板。
3. 给 `reward_process.py` 做语义分区整理，删除不再启用的旧 reward。
4. 保持 `algorithm_ppo.py`、`actor_critic.py`、观测维度处理不动，除非有明确训练收益。
5. 所有重构必须先 `py_compile`，再短训验证，不要在长训前做大范围拆文件。

### 7.3 最终交付前的容器清理

最终提交、打包或交付前，需要额外清理开发容器中的 RPC 与同步痕迹。原因是这些文件只服务本地开发和同步，不属于比赛模型代码，也可能包含过长日志、上传分片、临时备份或调试信息。

需要重点检查和清理：

```text
agent_diy/codex_rpc_bridge_runtime/uploads/
agent_diy/codex_rpc_bridge_runtime/backups/
agent_diy/codex_rpc_bridge_runtime/*.log
agent_ppo/codex_rpc_bridge_runtime/
```

如果容器内存在以下开发辅助文件，也应确认不会进入最终提交包：

```text
agent_diy/codex_rpc_bridge/
agent_diy/codex_rpc_bridge_runtime/
__pycache__/
.cache/
*.log
*.zip
*.tar.gz
*.pth
*.pkl
```

当前同步脚本已经会跳过大部分 `agent_diy`、日志、缓存、压缩包和模型文件；但最终交付前仍应手动核查容器侧目录，防止早期实验残留没有被清掉。

推荐最终检查流程：

```text
1. 查看容器内 agent_diy/codex_rpc_bridge_runtime 是否还有 uploads/backups/log。
2. 查看容器内 agent_ppo 下是否误残留 codex_rpc_bridge_runtime。
3. 查看是否存在多余 zip、tar.gz、pth、pkl、jsonl、log。
4. 确认最终提交/训练目录只包含 agent_ppo、conf 等比赛需要文件。
5. 最终同步或打包前再跑一次 dry-run，确认 sample_files 中没有 RPC/runtime/日志/模型。
```

注意：RPC 脚本本身是开发工具，不应作为训练逻辑的一部分；最终版本应只依赖比赛允许的训练/评估代码路径。

## 8. 一句话总结

p06 中最明显、排除官方模板后仍与 xtrack9 高度相似的旧模块，当前已经删除了大部分；剩余主要集中在监控、workflow、reward 和少量配置。当前训练思路已经从 p06/xtrack9 的“速度和启发式推进”逐步调整为“完成率优先、建议速度辅助、直接评分奖励主导、高难度完成保护”的团队版本。
