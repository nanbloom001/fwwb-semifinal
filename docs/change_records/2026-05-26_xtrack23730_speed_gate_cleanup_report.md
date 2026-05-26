# xtrack-23730 Handoff 交接文档

日期：2026-05-26

本文只记录 `代码归档/xtrack9-23730` 基线代码的行为、当前仓库相对 23730 的修改、以及继续训练/评估前必须注意的潜在隐患。不要把本文理解为前面所有实验的流水账。

## 基线范围

23730 基线代码：

- `代码归档/xtrack9-23730/agent_ppo`
- 对应日志：`arena_frontend_monitor_runtime/manual_metric_recorder/sessions/xtrack-23730`

当前待交接代码：

- `agent_ppo`
- 当前活动配置：`Config.CURRENT = TrackNavConfig`
- 当前训练 TOML：`agent_ppo/conf/train_env_conf_track_nav.toml`
- 当前容器端 `agent_ppo/conf` 已清理到只保留 `train_env_conf_track_nav.toml`

## 23730 原始语义

### 1. Track 训练主配置

23730 使用 Track 模式：

```toml
[terrain]
mode = "track"
difficulty_range = [0.0, 1.0]
curriculum = false

[terrain.track]
sub_terrains = [
  "pyramid_slope",
  "pyramid_slope_inv",
  "pyramid_stairs",
  "pyramid_stairs_inv",
  "open_entry_maze",
]
```

等级比例由 `[terrain.level_mix]` 控制，不是课程逐级推进。

### 2. 23730 有两套速度口径

环境原始命令：

```toml
[commands.ranges]
lin_vel_x = [0.55, 0.80]
lin_vel_y = [0.0, 0.0]
ang_vel_yaw = [0.0, 0.0]
```

RL phase command：

```toml
[rl_navigation]
phase_command_enabled = true
pre_maze_lin_vel_x = [1.20, 1.44]
slope_lin_vel_x = [1.10, 1.32]
stairs_lin_vel_x = [0.86, 1.06]
maze_lin_vel_x = [1.24, 1.48]
terrain_phase_speed_enabled = true
phase_command_resample_steps = 160
```

关键结论：

- 23730 的原始环境命令范围是 `0.55-0.80 m/s`。
- phase command 的 `1.2+ m/s` 主要 patch policy/critic observation。
- 旧路径会 best-effort 写入底层 `command_manager`，但是否真实写成功依赖 env unwrap。
- 23730 日志没有可靠的 `rl_phase_command_vx` 或 `obs_cmd_vel_x` 直读面板；从 `obs_actual_vel_x≈0.80`、`obs_lin_vel_x_error≈0.22` 反推，更接近原始命令口径，而不是全程真实执行 `1.2+ m/s`。

这说明：如果当前代码把更高速度真实写进 worker command，训练语义可能比 23730 激进很多。

### 3. 23730 地形判断方式

23730 使用 `train_workflow.py` 里的 `_estimate_pre_maze_terrain_from_obs()`：

- 输入主要来自 policy observation 的 height scan。
- 地形分为 flat/slope/stairs/wall/maze 相关阶段。
- maze 主要靠到 goal 的距离门控：
  - `phase_maze_goal_dist_gate = 14.0`
- 没有独立的 `terrain_gate.py` worker 门控模块。
- 没有当前版本的 sticky raw-fused 门控、nav scanner 墙体融合、难度缓存、worker command diagnostics。

### 4. 23730 eval 行为

`agent.py` 中 eval 会读取：

- `pre_maze_lin_vel_x`
- `slope_lin_vel_x`
- `stairs_lin_vel_x`
- `maze_lin_vel_x`

并在 `eval_command_override = true` 时把平台随机 command 改成稳定前进锚点。23730 的：

```toml
eval_command = [1.22, 0.0, 0.0]
```

风险是评估侧也可能看到偏激进的前进锚点。

## 当前代码相对 23730 的主要修改

### 1. 训练阶段和保存频率

`agent_ppo/conf/conf.py`：

- `TrackNavConfig.model_save_interval` 从 `20` 改为 `10`。
- 增加 eval 模式下的 train-only reward 过滤逻辑。

用途：

- 继续训练时更密集保存节点，便于挑选模型。
- 评估时尽量剥离训练用 probe/reward，降低评估计算负担。

潜在影响：

- 保存更频繁会增加少量 I/O。
- eval reward 过滤依赖 `custom_parameters`；如果后续配置写错，可能错误保留或错误删除 reward。

### 2. 观测处理接入 worker gate

`agent_ppo/feature/policy_observation_process.py` 和 `critic_observation_process.py` 增加：

```python
obs = apply_worker_gate_command(self.env, obs, "policy")
obs = apply_worker_gate_command(self.env, obs, "critic")
```

用途：

- 在生成 policy/critic observation 时同步写入 terrain gate 计算出的 command。
- 保证 policy obs `obs[:, 6:9]` 和 critic obs command 段尽量与 worker target command 一致。

潜在影响：

- 这是训练语义变化点，不是纯监控改动。
- 如果 `terrain_gate.py` 判断异常，会直接影响模型看到的速度指令。
- 如果 worker command 写入底层 env 成功，`track_lin_vel_xy` 奖励口径也会变化。

### 3. 新增 worker terrain gate

新增 `agent_ppo/feature/terrain_gate.py`。

当前启用：

```toml
phase_command_enabled = true
worker_phase_command_enabled = true
gate_test_mode = "sticky"
terrain_sticky_source = "raw_fused"
raw_terrain_gate_enabled = true
track_difficulty_cache_enabled = true
```

功能：

- 基于 raw height scanner / nav scanner / 原 observation 特征做地形判断。
- sticky 输出最终地形，减少 slope/stairs/flat/maze 的快速跳变。
- maze 主要使用 goal 距离先验。
- 缓坡、台阶难度用连续 factor 估计。
- maze 可使用前面 slope/stairs 阶段缓存的难度。
- 通过 worker command 给不同地形发不同前进速度。

和 23730 差异：

- 23730 是旧 `_estimate_pre_maze_terrain_from_obs()`。
- 当前是独立 worker gate 主路径。
- 当前有 sticky、raw-fused、nav wall、难度缓存、command 写入诊断。

潜在影响：

- gate 判断错误会直接改变 command。
- raw scanner/nav scanner 如果平台返回异常，必须依赖 fallback。
- sticky 参数如果过强，可能在地形切换处滞后；如果过弱，可能抖动。

### 4. 速度配置改为保守单套区间 + 线性难度映射

23730：

```toml
pre_maze_lin_vel_x = [1.20, 1.44]
slope_lin_vel_x = [1.10, 1.32]
stairs_lin_vel_x = [0.86, 1.06]
maze_lin_vel_x = [1.24, 1.48]
```

当前：

```toml
fallback_lin_vel_x = 0.90
slope_lin_vel_x = [0.72, 1.20]
stairs_lin_vel_x = [0.65, 1.00]
maze_lin_vel_x = [0.72, 0.92]
phase_command_linear_speed_enabled = true
phase_command_jitter = 0.015
```

已删除/弃用：

- `pre_maze_lin_vel_x`
- `stairs_low_lin_vel_x`
- `stairs_mid_lin_vel_x`
- `stairs_high_lin_vel_x`
- `terrain_difficulty_speed_scales`

当前语义：

- flat/unknown 使用 `fallback_lin_vel_x = 0.90`。
- slope/stairs 使用各自单套速度区间。
- difficulty factor 线性映射：
  - 低难度接近速度上限。
  - 高难度接近速度下限。
- maze 使用较低速度区间 `0.72-0.92`。

潜在影响：

- 比 23730 phase command 慢，但可能更接近 23730 实际环境速度。
- 时间分可能短期下降，但完成率和姿态/能耗风险应更低。
- 如果 difficulty factor 偏高，模型会长期收到偏慢指令。
- 如果 difficulty factor 偏低，高难度台阶可能仍偏快。

### 5. maze 门控修改

23730：

```toml
phase_maze_goal_dist_gate = 14.0
```

当前：

```toml
phase_maze_goal_dist_gate = 8.0
phase_maze_distance_mode = "longitudinal"
phase_maze_confirm_steps = 3
phase_maze_release_margin = 2.0
phase_maze_sticky_until_done = true
```

用途：

- 避免过早把前面地形识别成 maze。
- 用目标方向上的距离差更贴近“最后一个地形是迷宫”的先验。
- maze 一旦确认后保持到 episode 结束。

潜在影响：

- 如果入口实际触发距离不是 8m 左右，maze 触发会提前或滞后。
- `sticky_until_done` 会让 maze 误触发的代价很高。

### 6. 训练 reward 增量

当前在 23730 基础上新增或调整：

- `pose_score_formula`: `1.35 -> 1.45`
- `high_difficulty_pose_pressure`
- `high_difficulty_ang_vel_xy`
- `high_difficulty_energy`
- `goal_near_wrong_direction_penalty`
- `goal_miss_penalty`
- `slope_time_pressure`
- `low_mid_stair_time_pressure`
- `nav_wall_impact_penalty`
- `nav_wall_stuck_push_penalty`
- `navigation_time`: `-0.026 -> -0.020`

保留了 23730 的主要 locomotion 和 navigation reward 框架，例如：

- `track_lin_vel_xy`
- `command_speed_advantage`
- `track_ang_vel_z`
- `forward_heading_velocity`
- `goal_velocity_projection`
- `approach_goal`
- `task_complete`
- 原 height-scan maze wall 系列 reward

潜在影响：

- 这些不是纯监控，都会影响训练。
- 高难度姿态/能耗压力依赖 gate 难度可靠性；如果难度误判，会错误施压。
- 迷宫新增惩罚依赖墙体检测质量；如果 nav/height 墙体检测误报，可能把正确转向压掉。
- 时间压力如果和完成率冲突，优先应回退时间压力。

### 7. gate probe 和监控面板

当前新增大量 `gate_*` probe reward，权重 `1e-9`，用于上报：

- current/nan/raw/sticky/final 地形占比。
- low/mid/high 难度占比。
- difficulty cache。
- maze using cache。
- command written。
- worker/target command。
- policy/critic command error。
- actual minus command。
- gate disagreement。

`custom_parameters` 当前写法：

```toml
all_train_rewards_train_only = true
train_only_reward_prefixes = ["gate_"]
keep_train_only_rewards_in_eval = false
```

注意：

- gate probe 在训练时用于监控。
- eval 模式会尝试过滤训练 reward，降低评估开销。
- 目前过滤逻辑比较激进：`all_train_rewards_train_only = true` 会把训练 TOML 里的所有 rewards 加入 train-only 集合，再叠加 prefix 过滤。若 eval 确实加载训练 TOML，这会尽量清掉 reward；若平台 eval 使用独立 eval TOML，则依赖 eval TOML 自身。

潜在影响：

- 训练时 gate probe 虽然权重极小，但 reward 函数仍会被调用，仍有计算成本。
- eval 过滤必须用日志确认：评估日志应出现 stripped train-only reward 的信息，或至少不再调用 gate probe。

### 8. 监控面板重构

`agent_ppo/conf/monitor_builder.py` 相对 23730 大幅增加：

- Track 结果诊断：完成率、异常率、超时率、各等级样本数。
- L 级异常率、L 级超时率。
- 高难度压力。
- 门控状态、地形占比、有效难度占比、难度覆盖。
- 命令同步。
- raw nav 墙体。
- 撞墙冲击/卡墙前推惩罚。

修复过的面板语义：

- `invalid` 改为 wall/invalid 语义，避免误读为非法地形。
- 难度占比使用有效 slope/stairs 分母。
- 命令同步优先用 reward-probe 指标，避免空白的直读 `gate_worker_*` 面板。

潜在影响：

- 面板本身不改变训练，但对应 probe reward 会增加训练计算。
- 平台面板命名有严格字符限制，后续新增指标要避免 `%`、过长英文名。

### 9. eval command 修改

23730：

```toml
eval_command = [1.22, 0.0, 0.0]
```

当前：

```toml
eval_command = [0.85, 0.0, 0.0]
```

用途：

- 降低评估时固定前进锚点，避免过快导致姿态/能耗/完成率恶化。

潜在影响：

- 如果模型是在更高 command 锚点下学出来的，评估 command 降低可能改变行为。
- 当前 eval side 仍不是完整 worker gate 语义，只是近似分类 + 区间中点。

### 10. 配置文件清理

当前本地和容器 `agent_ppo/conf` 仅保留：

- `train_env_conf_track_nav.toml`

已删除旧 standard、旧 track、eval wide TOML。

用途：

- 避免同步无关配置。
- 避免人工误选旧 TOML。

潜在影响：

- 如果以后要切回 Standard 或旧实验，需要从 `代码归档` 或 git 恢复对应 TOML。

## 当前最重要的潜在隐患

### 隐患 1：当前 command 写入语义可能比 23730 更真实

23730 的 phase command 很可能没有稳定写入底层 command manager，更多是 observation override。当前 worker gate 会更积极地尝试同步 policy/critic obs 和底层 command。

验证方式：

- 看 `command_written_pct` 是否接近 100%。
- 看 `policy_cmd_error_mps`、`critic_cmd_error_mps` 是否接近 0。
- 看 `worker_cmd_vx_mps` 与 `target_cmd_vx_mps` 是否一致。
- 看 `actual_minus_cmd_vx_mps` 是否处于合理范围。

如果真实写入后完成率下降，优先降低速度区间或关闭 worker command 写入，只保留 obs override 对齐 23730。

### 隐患 2：train/eval 语义仍不完全一致

训练主路径使用 `terrain_gate.py` worker gate；eval `agent.py` 使用 observation 近似分类和速度中点。

验证方式：

- 评估日志确认是否启用 `eval_command_override`。
- 对比训练中 `worker_cmd_vx_mps` 分布与评估固定/近似 command。

若评估和训练差异过大，优先考虑让 eval command 固定为 `0.85`，暂时不要在 eval 做复杂门控。

### 隐患 3：gate 难度如果不可靠，会污染 reward 和速度

当前高难度姿态/能耗压力、slope/stairs 时间压力、速度线性映射都依赖 gate 难度。

验证方式：

- `difficulty_valid_pct` 不应异常低。
- `valid_slope_stairs_raw` 应与 slope+stairs 时间占比大致匹配。
- low/mid/high 在有效难度分母下应可解释，不要求全局加起来覆盖 flat/maze。
- `maze_using_cache` 应在 maze 阶段较高，但 cache unknown 不应持续过高。

若难度长期 unknown 或高难度占比明显反常，先关闭难度相关 reward，只保留地形 gate。

### 隐患 4：maze wall 奖励仍依赖传感器判墙质量

当前同时有旧 height-scan 墙体 reward 和新增 nav wall impact/stuck reward。狭窄通道、高速移动、height/nav 视角差异都可能导致误报/漏报。

验证方式：

- 看 `raw nav墙体` 面板是否在 maze 阶段有合理读数。
- 看 `reward_nav_wall_impact_penalty` 和 `reward_nav_wall_stuck_push_penalty` 是否只在迷宫撞墙/卡墙时明显。
- 结合视频确认是否把正常贴边转弯误判成撞墙。

若误报明显，优先降低 nav wall 相关权重或只保留监控。

### 隐患 5：probe reward 训练开销

训练时大量 gate probe 会调用 reward 函数，虽然权重只有 `1e-9`，但不是零成本。

处理原则：

- 短训验证 gate 时保留。
- 长训如果面板已经稳定，可减少重复 probe。
- 评估必须确认这些 probe 被关闭或过滤。

### 隐患 6：清理 TOML 后切阶段会失败

当前只保留 track nav TOML。`LocomotionConfig`、`StairConservativeConfig` 等 stage 类还在，但对应 TOML 已删除。

如果切回 Standard，需要先恢复 TOML。

## 下一轮短训验证清单

建议先跑 10-20 分钟，不直接长训。

必须看：

- `completion_pct` 是否不低于 23730 同阶段明显太多。
- `abnormal_pct` 是否没有上升。
- `timeout_pct` 是否没有上升。
- `command_written_pct` 是否接近 100%。
- `worker_cmd_vx_mps` 是否落在预期：
  - fallback 约 `0.90`
  - slope `0.72-1.20`
  - stairs `0.65-1.00`
  - maze `0.72-0.92`
- `policy_cmd_error_mps`、`critic_cmd_error_mps` 是否接近 0。
- flat/slope/stairs/maze/wall_or_invalid 占比是否接近 100% 且可解释。
- 有效难度 low/mid/high 是否只在 slope/stairs 上统计。
- `maze_using_cache` 是否符合“进入 maze 后使用前段难度缓存”的预期。

视频重点看：

- L6-L9 台阶是否因为速度/高难度压力导致保守或摔倒。
- maze 是否仍绕柱子转圈。
- 贴墙转弯是否被 nav wall 惩罚压坏。
- 到终点附近是否出现越过目标或反向绕圈。

## 回退建议

若完成率明显下降：

1. 先关或减小新增训练 reward：
   - `slope_time_pressure`
   - `low_mid_stair_time_pressure`
   - `high_difficulty_*`
   - `nav_wall_*`
   - `goal_near_wrong_direction_penalty`
   - `goal_miss_penalty`
2. 保留监控，确认是否 gate 误判。
3. 如果 command 写入导致行为突变，关闭 `worker_phase_command_enabled` 或把速度全部压到接近 23730 实际口径。

若评估超时：

1. 确认 eval 过滤是否生效。
2. 确认评估没有调用 gate probe 和训练 reward。
3. 不要改平台评估逻辑，优先减少训练侧 probe/reward 在 eval 配置中的残留。

若 maze 仍不走最短路：

1. 不先动前段 slope/stairs 训练。
2. 只微调 maze 相关 reward。
3. 保持 23730/T8 的完成率优先，不用强时间压力硬拉速度。

## 当前操作状态

已完成：

- 本地代码已改到上述状态。
- 容器已同步过当前主要 `agent_ppo` 代码。
- 容器 `agent_ppo/conf` 已清理旧 TOML。
- 容器 RPC runtime uploads/backups/log 已清理。
- 本地语法检查已通过：
  - `agent_ppo/conf/monitor_builder.py`
  - `agent_ppo/feature/terrain_gate.py`
  - `agent_ppo/feature/reward_process.py`
  - `agent_ppo/agent.py`
  - `agent_ppo/workflow/train_workflow.py`

未完成/仍需实测：

- 当前 handoff 文档本身未按 routine sync 上传，因为常规 sync 排除 docs。
- 当前 worker gate + conservative speed 的短训效果仍需看最新日志确认。
- eval 侧是否完全关闭训练 probe，需要用一次评估日志确认。

