# 腾讯开悟机器狗竞赛 — 本会话全部改动记录

> 记录时间：2026-05-07  
> 阶段：Phase-0 平地步态塑形  
> 基础算法：agent_ppo (PPO + GAE)

---

## 一、`agent_ppo/conf/conf.py`

### 1.1 提高模型保存频率

| 字段 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `model_save_interval` | `500` | `250` | 更密集的检查点便于回滚；平地阶段 episode 短，250 约每 ~2h 保存一次 |

---

## 二、`agent_ppo/conf/train_env_conf_standard_locomotion.toml`

### 2.1 Phase-0 地形配置（关闭课程 / 锁定平地）

| 字段 | 值 | 说明 |
|---|---|---|
| `terrain.curriculum` | `false` | 关闭平台内置课程（同时冻结地形难度和速度范围扩展） |
| `terrain.max_init_terrain_level` | `0` | 所有机器人从最简单难度（坡度≈0）开始 |
| `pyramid_slope` / `pyramid_slope_inv` | 各 `0.50` | 100% 平地坡面，0% 楼梯/迷宫 |
| `push_robots` | `false` | 关闭外部推力干扰 |

### 2.2 奖励权重调整（Phase-0 平地步态塑形）

| 奖励项 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `flat_orientation` | `-0.5` | `-3.0` | Phase-0 核心目标，6× 加强，平地没有倾倒借口 |
| `correct_base_height` | `-0.3` | `-2.0` | posture 评分项，平地高度应非常稳定 |
| `ang_vel_xy` | `-0.1` | `-0.5` | 与 flat_orientation 联动，抑制 pitch/roll 抖动 |
| `lin_vel_z` | `-2.0` | `-0.5` | 平地上下跳跃少，释放梯度预算 |
| `undesired_contacts` | `-1.0` | `-0.3` | 平地意外接触少，减少无意义惩罚 |
| `dof_pos_limits` | `-2.0` | `-0.3` | 平地极限触发少 |
| `termination` | `-1.0` | `-3.0` | 平地摔倒是真正失败，加重惩罚 |
| `action_rate` | `-0.01` | `-0.03` | 3× 加强，保守加强动作平滑 |
| `action_smoothness` | `-0.01` | `-0.02` | 2× 加强，与 action_rate 联动 |
| `air_time_variance_penalty` | `-0.1` | `-0.3` | 步态对称性是 Phase-0 主要指标 |

### 2.3 新增奖励项（Phase-0 首次激活）

| 奖励项 | 权重 | 说明 |
|---|---|---|
| `hip_to_default` | ~~`-1.0`~~ **已移除** | 见 2.4 |
| `joint_position_penalty` | `-0.5` | 全身12关节 L2 惩罚，Phase-0 核心约束 |
| `dof_vel` | `-1e-3` | 关节速度平方和惩罚，抑制鬼畜步态 |
| `base_lateral_vel` | `-0.5` | 侧向速度误差 `(actual_vy - cmd_vy)²`，防螃蟹步 |
| `pivot_turning` | `-0.2` | 原地蹭脚转弯惩罚 |

### 2.4 移除 `hip_to_default`

**原因**：`hip_to_default` 仅惩罚髋关节 [0,3,6,9]，而 `joint_position_penalty` 已覆盖全部 12 关节（含髋）。同时保留两者会：
- 对髋关节双重惩罚（有效权重 `-1.5`），对膝/踝关节仅 `-0.5`
- 梯度偏向髋关节，欠约束膝/踝，破坏全链条姿态自然性

### 2.5 Bug 修复：`joint_position_penalty` 参数

| 参数 | 旧值 | 新值 | 问题 |
|---|---|---|---|
| `cmd_threshold` | `0.1` | `0.05` | `lin_vel_x=[0.0, 0.5]`，0.05~0.1 m/s 的低速指令被误判为"静止"，受 `stand_still_scale=1.5` 倍惩罚，压制低速跟踪 |

### 2.6 `feet_air_time` 调整

| 参数 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `weight` | `1.5` | `1.0` | 与新 threshold 匹配 |
| `threshold` | `0.5` s | `0.25` s | `0.5s` 鼓励"悬脚尽量长再落地"，导致两腿走路刷分；`0.25s` 鼓励轻快步态 |

---

## 三、`agent_ppo/conf/monitor_builder.py`

### 3.1 修复非法面板名称（校验失败 3 个错误）

| 原名称 | 修正后 | 非法字符 |
|---|---|---|
| `pitch/roll 角速度惩罚` | `pitch roll 角速度惩罚` | `/` |
| `动作变化率惩罚(一阶)` | `动作变化率惩罚 一阶` | `()` |
| `动作平滑惩罚(二阶)` | `动作平滑惩罚 二阶` | `()` |

平台要求：面板中文名称仅支持中英文、数字及 `_-空格`，1~20 字符。

### 3.2 移除 `髋关节偏离惩罚` 面板

与 2.4 同步，`hip_to_default` 已从奖励配置中移除，对应 monitor 面板一并删除。

### 3.3 新增 `速度课程等级` 面板（Group 2 速度跟踪末尾）

```python
.add_panel(name="速度课程等级", name_en="vel_curriculum_stage", type="line")
    .add_metric(metrics_name="vel_curriculum_stage",
                expr="avg(vel_curriculum_stage{})")
    .end_panel()
```

实时显示 `VelocityCurriculum` 当前处于哪个速度阶段（0-3），便于观察课程进展。

---

## 四、`agent_ppo/workflow/train_workflow.py`

### 4.1 新增 `VelocityCurriculum` 类

**设计原因**：`terrain.curriculum=false` 同时冻结了平台内置的地形难度扩展和速度范围扩展，无法独立控制两者。`VelocityCurriculum` 在 Python 层完全绕过该联动，独立实现速度范围的性能驱动式课程。

**逻辑（与地形课程相同）**：

| 行为 | 条件 | 默认阈值 |
|---|---|---|
| **升级** ↑ 至下一速度阶段 | `mean(reward_track_lin_vel_xy) ≥ promote_threshold` 连续 N 次检查 | threshold=2.0, N=5 |
| **降级** ↓ 至上一速度阶段 | `mean(reward_track_lin_vel_xy) < demote_threshold` 连续 M 次检查 | threshold=0.8, M=3 |
| **中性区** | 两阈值之间 | 双计数器各衰减 1（防振荡） |

每次检查 = 1 个 episode batch（`run_episodes_` 完成、`ep_infos.clear()` 之前）。

**4个速度阶段**：

| Stage | lin_vel_x | lin_vel_y | ang_vel_yaw | 阶段目标 |
|---|---|---|---|---|
| 0 | `[0.0, 0.5]` | `[-0.3, 0.3]` | `[-1.0, 1.0]` | 学会基本步态（与 TOML 初始值一致） |
| 1 | `[0.0, 1.0]` | `[-0.5, 0.5]` | `[-1.5, 1.5]` | 巩固 trot 步态 |
| 2 | `[0.0, 1.5]` | `[-0.8, 0.8]` | `[-1.5, 1.5]` | 稳定中速行走 |
| 3 | `[-0.5, 2.0]` | `[-1.0, 1.0]` | `[-1.5, 1.5]` | 完整范围（含倒退） |

### 4.2 主循环集成

```
run_episodes_()               ← 填充 ep_infos
episode += 1
vel_curriculum.check_and_update(ep_infos, ...)   ← ep_infos.clear() 之前读取
    ↓ 若阶段变化
    env.reset(usr_conf)       ← 应用新速度范围（地形不变）
    cur_reward_sum.zero_()    ← 清除未完成 episode 的脏数据（Bug修复）
    cur_episode_length.zero_()
agent.learn()
storage.clear()
report_monitor_data(ep_infos, ..., vel_stage=vel_curriculum.stage)
ep_infos.clear()
```

### 4.3 Bug 修复

| 问题 | 修复方式 |
|---|---|
| **`float \| None` Python 3.10+ 语法**：在 Python 3.8/3.9 上 `_mean_tracking_reward` 的返回类型注解会触发 `TypeError` | 改为 `Optional[float]`，并 `from typing import Optional` |
| **`env.reset` 后统计张量未清零**：阶段切换触发 `env.reset` 后，`cur_reward_sum`/`cur_episode_length` 保留上一轮未完成 episode 的值，下次 `dones` 触发时将脏数据写入 `rewbuffer` | `_apply_stage`/`check_and_update` 返回三元组 `(obs, critic_obs, reset_happened: bool)`，workflow 侧检测到 `vel_reset=True` 后立即调用 `.zero_()` 清零 |

### 4.4 `report_monitor_data` 更新

新增 `vel_stage: int = 0` 参数，写入 `monitor_data["vel_curriculum_stage"]`，供 monitor 面板展示。

---

## 五、TOML 与 VelocityCurriculum Stage-0 一致性确认

`commands.ranges` 初始值（TOML 中配置）与 `VelocityCurriculum.STAGES[0]` 完全匹配：

```toml
# TOML 初始值（无需修改）
lin_vel_x = [0.0, 0.5]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-1.0, 1.0]
```

```python
# VelocityCurriculum.STAGES[0]（Python 侧）
{"lin_vel_x": [0.0, 0.5], "lin_vel_y": [-0.3, 0.3], "ang_vel_yaw": [-1.0, 1.0]}
```

训练开始时 `_initialize_training_state` 从 TOML 读入 `usr_conf`，后续由 `VelocityCurriculum` 接管速度范围更新，TOML 无需添加任何新字段。

---

## 六、Phase-1 升级检查清单

满足以下条件后，切换到完整 standard 配置：

- [ ] `vel_curriculum_stage` 稳定在 2 或 3
- [ ] `reward_track_lin_vel_xy` 均值 > 2.0（即 `promote_threshold`）
- [ ] `reward_flat_orientation` 趋近 0（机身保持水平）
- [ ] `reward_termination` 接近 0（不再摔倒）
- [ ] 视觉确认：步态规整，无螃蟹步/跛行/鬼畜

Phase-1 切换操作：
1. `terrain.curriculum = true`
2. `terrain.max_init_terrain_level = 3`（逐步升至 5）
3. `pyramid_stairs` / `maze` 比例设为非零
4. `push_robots = true`
5. 删除 `VelocityCurriculum`（或保留，terrain.curriculum 此时主导）
