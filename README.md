# fwwb-RL-dog 🐕

> 腾讯开悟·四足机器人运动控制竞赛（legged_robot_competition_26 · IDE 22.0.3）强化学习训练方案

---

## 项目简介

本仓库是参加**腾讯开悟四足机器人运动控制竞赛**的训练代码与文档。机器人平台为 Unitree Go2（12-DOF），在 Isaac Sim 仿真环境中通过 PPO 算法训练端到端运动控制策略。

**比赛评分**（standard 地形模式）：

| 维度 | 权重 | 说明 |
|---|---|---|
| distance | 40% | 穿越地形的距离 |
| time | 20% | 完成用时 |
| energy | 20% | 能耗（扭矩×速度） |
| posture | 20% | 机身姿态稳定性 |

---

## 仓库结构

```
.
├── agent_ppo/                  # 核心 PPO 智能体（主要开发区域）
│   ├── algorithm/              # PPO + GAE 算法实现
│   ├── conf/
│   │   ├── conf.py             # 训练超参数（LocomotionConfig）
│   │   ├── monitor_builder.py  # 监控面板配置
│   │   └── train_env_conf_standard_locomotion.toml  # 环境/奖励/课程配置
│   ├── feature/                # 观测处理、奖励计算
│   ├── model/                  # Actor-Critic 网络
│   └── workflow/
│       └── train_workflow.py   # 主训练循环 + VelocityCurriculum
│
├── agent_diy/                  # DIY 扩展框架（可选）
├── conf/                       # 平台级应用配置
├── isaac_env/                  # Isaac 环境接口
├── legged-robot/               # 赛题文档与分析报告
│   ├── README.md
│   ├── 赛题综合分析报告.md
│   ├── 开源项目调研报告.md
│   ├── 开发指南/               # 环境/观测/动作/协议说明
│   └── 腾讯开悟强化学习框架/   # KaiwuDRL 框架文档
│
├── CHANGES.md                  # 本训练方案全部改动记录
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
| Critic | MLP [316→512(LN)→256(LN)→128→1]，含 LayerNorm |
| 观测维度 | Actor 301-dim（45 proprioception + 256 height scan）；Critic 316-dim（+15 privileged） |
| 动作 | 12-dim 连续 [-1,1]，action_scale=0.25，PD 控制器目标关节角 |

---

## 训练阶段规划

### Phase-0：平地步态塑形（当前）

- 地形：`terrain.curriculum=true`，`num_rows=3`，`difficulty_range=[0,0.2]`（接近平地）
- 速度范围：由 `VelocityCurriculum` 性能驱动独立扩展（共 4 个速度阶段）
- 核心目标：正确步态节律、机身水平、无螃蟹步、无鬼畜

**速度课程阶段**（配置见 TOML `[velocity_curriculum]`）：

| Stage | lin_vel_x | lin_vel_y | ang_vel_yaw | 目标 |
|---|---|---|---|---|
| 0 | [0.0, 0.5] | [-0.3, 0.3] | [-1.0, 1.0] | 基础步态 |
| 1 | [0.0, 1.0] | [-0.5, 0.5] | [-1.5, 1.5] | 巩固 trot |
| 2 | [0.0, 1.5] | [-0.8, 0.8] | [-1.5, 1.5] | 中速稳定 |
| 3 | [-0.5, 2.0] | [-1.0, 1.0] | [-1.5, 1.5] | 完整范围 |

**Phase-0 升级标准**：
- `vel_curriculum_stage` 稳定在 ≥ 2
- `reward_track_lin_vel_xy` 均值 > 2.0
- `reward_flat_orientation` 趋近 0
- `reward_termination` 接近 0

### Phase-1（计划）

- 开启完整 terrain curriculum（num_rows=10，difficulty_range=[0,1]）
- 加入楼梯/迷宫地形比例
- 开启 `push_robots=true`

---

## 关键配置文件速查

| 文件 | 作用 |
|---|---|
| `agent_ppo/conf/conf.py` | `model_save_interval`、批次大小、PPO 超参 |
| `agent_ppo/conf/train_env_conf_standard_locomotion.toml` | 地形、域随机化、速度命令、奖励权重、速度课程 |
| `agent_ppo/conf/monitor_builder.py` | 监控面板分组（7组，共23个指标） |
| `CHANGES.md` | 本会话全部改动的详细记录与技术依据 |

---

## 改动记录

详见 [CHANGES.md](CHANGES.md)。

---

## 参考资料

- [腾讯开悟竞赛文档](legged-robot/README.md)
- [赛题综合分析报告](legged-robot/赛题综合分析报告.md)
- [开源方案调研报告](legged-robot/开源项目调研报告.md)
- [walk-these-ways (Improbable-AI)](https://github.com/Improbable-AI/walk-these-ways)
