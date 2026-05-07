# 腾讯开悟机器狗竞赛当前版本改动记录

> 记录时间：2026-05-07  
> 当前状态：开放完整难度带、低难度起步的 Standard 课程训练  
> 基础算法：agent_ppo（PPO + GAE）

---

## 本轮改动摘要

本轮代码已经不再停留在“近似平地塑形”阶段，当前主线状态为：

- standard 地形课程开启
- 难度范围恢复到 `difficulty_range = [0.0, 1.0]`
- 保留 `num_rows = 10` 的完整课程档位
- 初始放置上限保持 `max_init_terrain_level = 0`，从最低档启动训练
- 地形中加入正反楼梯比例
- 监控面板补充训练进展与物理观测量
- 速度课程改为 tracking ratio 驱动
- 增加近零速站立稳定惩罚与观测维度断言

---

## 1. 训练配置更新

文件：`agent_ppo/conf/train_env_conf_standard_locomotion.toml`

### 1.1 地形课程恢复为完整难度带

| 项目 | 当前值 | 说明 |
|---|---|---|
| `terrain.curriculum` | `true` | 启用 standard 地形课程 |
| `num_rows` | `10` | 恢复完整 10 档课程难度 |
| `difficulty_range` | `[0.0, 1.0]` | 覆盖 standard 模式完整难度带 |
| `max_init_terrain_level` | `0` | 机器人仍从最简单档起步 |

### 1.2 地形配比改为坡面 + 楼梯混合

| 子地形 | 比例 |
|---|---|
| `pyramid_slope` | `0.35` |
| `pyramid_slope_inv` | `0.35` |
| `pyramid_stairs` | `0.15` |
| `pyramid_stairs_inv` | `0.15` |
| `maze` | `0.0` |

目的：在保持大部分样本仍可稳定学习步态的同时，让策略真正见到上台阶与下台阶场景。

### 1.3 速度课程阈值改为 tracking ratio

| 项目 | 当前值 | 说明 |
|---|---|---|
| `promote_threshold` | `0.55` | 连续 2 次达到后升级 |
| `promote_count` | `2` | 升级计数 |
| `demote_threshold` | `0.30` | 连续 3 次低于后降级 |
| `demote_count` | `3` | 降级计数 |

这套阈值不再直接依赖 reward 权重绝对值，避免改权重后课程推进逻辑失真。

### 1.4 奖励项针对楼梯与近零速稳定性重调

主要调整如下：

- 放松 `flat_orientation = -1.5`，避免上台阶时过度限制俯仰
- 放松 `correct_base_height = -0.8`，允许过台阶时机身高度自然变化
- 放松 `action_rate = -0.02` 与 `action_smoothness = -0.01`，保留平滑但不压制抬腿
- 放松 `dof_vel = -5e-4`、`feet_stumble = -0.05`
- 保留并降低 `joint_position_penalty = -0.3`，允许膝髋更大姿态偏移
- 新增 `stand_still_motion = -0.8`，专门抑制近零命令下的原地上下晃动与反复抬腿

---

## 2. 监控面板扩展

文件：`agent_ppo/conf/monitor_builder.py`

监控面板已扩展为 9 组、35 个指标。

### 2.1 新增训练进展分组

新增指标：

- `mean_episode_length`
- `mean_episode_reward`

这两个指标用于判断训练是否真正变稳，而不只是在看某个单独奖励项上升。

### 2.2 速度课程增加 ratio 面板

新增指标：

- `vel_curriculum_tracking_ratio`
- `vel_curriculum_stage`

用于直接观察速度课程推进依据与当前阶段。

### 2.3 新增物理观测量分组

新增指标：

- `obs_lin_vel_x_error`
- `obs_lin_vel_y_error`
- `obs_actual_vel_x`
- `obs_base_height`
- `obs_ang_vel_xy`

这些指标展示的是物理量本身，不受 reward 权重影响，可直接用于判断是否“真实收敛”。

### 2.4 面板命名合法化

此前包含斜杠和括号的中文标题已统一改成平台允许的命名格式，避免监控面板校验失败。

---

## 3. 工作流与健壮性修复

文件：`agent_ppo/workflow/train_workflow.py`

### 3.1 VelocityCurriculum 改为比例驱动

速度课程推进逻辑不再看绝对 tracking reward，而是使用 tracking ratio。这样 reward 权重变化不会再把课程阈值一起拖偏。

### 3.2 增加启动和运行期校验

新增保护逻辑：

- 启动时校验 Stage 0 与 `commands.ranges` 完全一致
- 校验 standard 子地形比例之和必须为 1.0
- 课程切换后若 `env.reset(usr_conf)` 失败，直接抛出异常

### 3.3 监控上报补充 episode 指标与物理量采样

`report_monitor_data` 现在会写入：

- `mean_episode_length`
- `mean_episode_reward`
- `vel_curriculum_tracking_ratio`
- `vel_curriculum_tracking_reward`
- `vel_curriculum_stage`
- `obs_*` 物理观测量

同时增加了环境解包与 robot asset 查找逻辑，用于从 env 中稳定采样基础物理状态。

---

## 4. 观测处理修复与断言

文件：

- `agent_ppo/feature/policy_observation_process.py`
- `agent_ppo/feature/critic_observation_process.py`

### 4.1 修复 policy 观测处理文件语法错误

本轮修复了 `policy_observation_process.py` 的缩进损坏问题。此前该文件曾出现 `return` 脱离函数体的语法错误，现已恢复正常。

### 4.2 加入运行时维度断言

| 文件 | 断言维度 |
|---|---|
| `PolicyObservationProcess` | `301` |
| `CriticObservationProcess` | `316` |

一旦高度扫描或 privileged observation 拼接异常，会在运行期直接报错，而不是静默带着错误观测继续训练。

---

## 5. 自定义奖励实现更新

文件：`agent_ppo/feature/reward_process.py`

### 5.1 `joint_position_penalty` 近静止判定细化

从单一命令阈值改为区分线速度与角速度阈值，避免低速转向或低速行走时被误当成“静止状态”。

### 5.2 新增 `stand_still_motion`

该项用于在近零速命令下惩罚：

- 机身上下速度
- pitch/roll 摆动
- 关节速度过大导致的抬腿抖动

目标是抑制“原地上下晃动、小腿抬起又放下”的失败模式。

---

## 6. 当前建议的观察重点

训练时建议优先同时看以下几类指标：

- `mean_episode_length` 是否稳定上升
- `mean_episode_reward` 是否持续改善
- `vel_curriculum_tracking_ratio` 是否能推动课程升级
- `obs_lin_vel_x_error` 是否在下降
- `obs_base_height` 与 `obs_ang_vel_xy` 是否稳定
- `reward_feet_stumble`、`reward_termination` 是否在楼梯场景下恶化

如果 reward 曲线好看，但 `obs_*` 物理量没有改善，通常意味着权重配置存在掩盖问题，而不是策略真的学会了。
