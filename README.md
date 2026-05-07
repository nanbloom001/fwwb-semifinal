# fwwb-RL-dog

> 腾讯开悟四足机器人运动控制竞赛训练仓库（legged_robot_competition_26 · IDE 22.0.3）

---

## 项目简介

本仓库用于 Unitree Go2 四足机器人在 Isaac Sim / KaiwuDRL 环境中的强化学习训练，当前主线方案为 PPO + GAE 的 asymmetric actor-critic。当前代码已经从早期“近似平地塑形”推进到 standard 模式下开放完整难度带、但仍从最低档起步的课程训练版本，并补齐了训练诊断、物理量监控、观测维度校验与近零速稳定性约束。

standard 地形模式比赛评分维度如下：

| 维度 | 权重 | 说明 |
|---|---|---|
| distance | 40% | 穿越地形的距离 |
| time | 20% | 完成用时 |
| energy | 20% | 能耗（扭矩 × 速度） |
| posture | 20% | 机身姿态稳定性 |

---

## 仓库结构

```
.
├── agent_ppo/                  # 当前主训练智能体
│   ├── algorithm/              # PPO + GAE
│   ├── conf/
│   │   ├── conf.py             # Stage 配置、网络维度、训练超参
│   │   ├── monitor_builder.py  # 监控分组与指标面板
│   │   └── train_env_conf_standard_locomotion.toml
│   │                           # 标准地形训练配置：地形 / 命令 / 奖励 / 速度课程
│   ├── feature/                # policy/critic 观测处理与 reward_process
│   ├── model/                  # Actor-Critic 网络
│   └── workflow/
│       └── train_workflow.py   # 主训练循环、监控上报、VelocityCurriculum
├── agent_diy/                  # 自定义智能体模板（未作为当前主线）
├── conf/                       # 平台级 app/algo 配置
├── isaac_env/                  # Isaac 环境接口
├── legged-robot/               # 赛题说明、开发指南、框架文档
├── CHANGES.md                  # 本轮代码与配置改动汇总
├── kaiwu.json                  # 平台元数据
└── train_test.py               # 本地快速测试入口
```

---

## 算法概览

| 组件 | 详情 |
|---|---|
| 算法 | PPO + GAE（λ=0.95, γ=0.99） |
| 网络结构 | Asymmetric Actor-Critic |
| Actor | MLP [301→512→256→128→12]，ELU 激活 |
| Critic | MLP [316→512→256→128→1]，含 LayerNorm |
| Policy 观测 | 301 维 = 45 proprio + 256 height scan |
| Critic 观测 | 316 维 = 60 critic proprio + 256 height scan |
| 动作 | 12 维连续动作，输出经 `action_scale=0.25` 映射到 PD 目标关节角 |

说明：当前 policy 与 critic 观测处理器都加入了运行时维度断言，用于尽早暴露 height scan 缺失或观测布局漂移问题。

---

## 当前训练配置

### 完整难度带开放、低难度起步的 Standard Curriculum（当前）

- 地形课程已开启：`terrain.curriculum = true`
- 课程档位：`num_rows = 10`
- 难度范围：`difficulty_range = [0.0, 1.0]`
- 初始放置上限：`max_init_terrain_level = 0`，即始终从最简单档起步
- 地形配比：35% 上坡、35% 下坡、15% 上台阶、15% 下台阶、0% maze
- 外部推力：当前仍关闭，先优先稳定通过完整标准难度带

当前版本的目标不再只是“学会平地走”，而是以较保守的课程方式逐步推进到完整难度带。当前训练分布仍以坡面为主，用来维持步态收敛，同时加入 30% 楼梯样本让策略开始学习真正的上台阶和下台阶动作。

当前重点同时兼顾：

- 完整 standard 难度带的课程推进
- 上下台阶适应能力
- 近零速度命令下的站立稳定性
- 步态对称、机身姿态、能耗与动作平滑之间的平衡

### 速度课程

速度命令扩展由 Python 侧 `VelocityCurriculum` 独立管理，与 terrain curriculum 解耦。课程判断不再使用绝对奖励值，而是使用 tracking ratio（当前 tracking reward 占理论上限的比例），从而避免奖励权重变化后课程门槛漂移。

| Stage | lin_vel_x | lin_vel_y | ang_vel_yaw | 用途 |
|---|---|---|---|---|
| 0 | [0.0, 0.5] | [-0.3, 0.3] | [-1.0, 1.0] | 启动阶段 |
| 1 | [0.0, 1.0] | [-0.5, 0.5] | [-1.5, 1.5] | 巩固基础 trot |
| 2 | [0.0, 1.5] | [-0.8, 0.8] | [-1.5, 1.5] | 中速稳定 |
| 3 | [-0.5, 2.0] | [-1.0, 1.0] | [-1.5, 1.5] | 完整速度范围 |

当前阈值：

- 升级：tracking ratio 连续 2 次 ≥ 0.55
- 降级：tracking ratio 连续 3 次 < 0.30

---

## 监控与诊断

当前监控面板已扩展为 9 组、35 个指标，核心分组如下：

- 训练进展：`mean_episode_length`、`mean_episode_reward`
- 速度跟踪：线速度/偏航跟踪奖励、`vel_curriculum_tracking_ratio`、`vel_curriculum_stage`
- 姿态质量：机身水平、高度、关节偏离、侧向漂移、pitch/roll 角速度
- 步态质量：脚部滞空、脚滑、脚撞台阶、关节速度、对称性、原地旋转
- 稳定接触：垂直速度、非预期接触、终止惩罚、关节限位
- 关节动作平滑：关节加速度、一阶动作变化、二阶动作平滑
- 能耗扭矩：`reward_energy`、`reward_joint_torques`
- 物理观测量：`obs_lin_vel_x_error`、`obs_lin_vel_y_error`、`obs_actual_vel_x`、`obs_base_height`、`obs_ang_vel_xy`

其中“物理观测量”分组直接展示奖励函数所依赖的物理量，而不是加权后的奖励值，用于区分“模型确实在物理上收敛”还是“只是奖励权重掩盖了问题”。

---

## 当前关键实现

### 1. 奖励与站立稳定

- 新增 `stand_still_motion`，专门惩罚近零速命令下的上下晃动、pitch/roll 摇摆和腿部反复抬放
- `joint_position_penalty` 改为分别处理线速度与角速度阈值，避免低速区间误判
- 针对楼梯适应，适度放松 `flat_orientation`、`correct_base_height`、`action_rate`、`action_smoothness`、`dof_vel` 等过强的平地先验

### 2. 观测与一致性校验

- `PolicyObservationProcess` 强制校验 301 维
- `CriticObservationProcess` 强制校验 316 维
- 观测维度不符时直接抛错，优先暴露 height scan 缺失或布局变更问题

### 3. 训练工作流健壮性

- `VelocityCurriculum` 启动时会校验 Stage 0 与 `commands.ranges` 一致
- 课程切换后若 `env.reset(usr_conf)` 失败，会直接抛出异常而不是静默继续
- 会校验 standard 子地形比例之和是否为 1.0
- 监控上报加入 episode 级指标与物理量采样

---

## 关键配置文件速查

| 文件 | 作用 |
|---|---|
| `agent_ppo/conf/conf.py` | Stage 定义、模型维度、训练超参 |
| `agent_ppo/conf/train_env_conf_standard_locomotion.toml` | 地形、命令范围、奖励权重、速度课程配置 |
| `agent_ppo/conf/monitor_builder.py` | 9 组监控面板与 35 个指标定义 |
| `agent_ppo/workflow/train_workflow.py` | 训练主循环、课程推进、监控上报、物理量采样 |
| `agent_ppo/feature/reward_process.py` | 自定义奖励函数实现 |
| `agent_ppo/feature/policy_observation_process.py` | Policy 观测拼接与 301 维断言 |
| `agent_ppo/feature/critic_observation_process.py` | Critic 观测拼接与 316 维断言 |
| `CHANGES.md` | 本轮实际改动清单与配置更新摘要 |

---

## 改动记录

本轮改动详见 [CHANGES.md](CHANGES.md)。

---

## 参考资料

- [腾讯开悟竞赛文档](legged-robot/README.md)
- [赛题综合分析报告](legged-robot/赛题综合分析报告.md)
- [开源项目调研报告](legged-robot/开源项目调研报告.md)
- [开发指南](legged-robot/开发指南)
- [腾讯开悟强化学习框架文档](legged-robot/腾讯开悟强化学习框架)
- [walk-these-ways (Improbable-AI)](https://github.com/Improbable-AI/walk-these-ways)
