# fwwb-RL-dog

> Unitree Go2 四足机器人强化学习训练仓库 —— PPO + GAE / Asymmetric Actor-Critic / Isaac Sim + Isaac Lab / KaiwuDRL

---

## 项目简介

本仓库用于 Unitree Go2 四足机器人在 Isaac Sim / KaiwuDRL 环境中的强化学习训练，支持 **standard（标准地形穿越）** 与 **track（路线导航）** 两种任务模式。

当前主线方案为 PPO + GAE 的 asymmetric actor-critic：

- Policy 观测 301 维 = 45 维本体感觉 + 256 维高度扫描
- Critic 观测 316 维 = 60 维特权本体感觉 + 256 维高度扫描
- 12 维连续动作，经 `action_scale=0.25` 映射到 PD 目标关节角
- 训练侧实现了速度课程（VelocityCurriculum）、地形课程、奖励退火、观测维度运行时断言等工程化能力

Standard 模式评分维度（满分 100）：

| 维度 | 权重 | 说明 |
|---|---|---|
| distance | 40% | 穿越地形的距离 |
| time | 20% | 完成用时 |
| energy | 20% | 能耗（扭矩 × 速度） |
| posture | 20% | 机身姿态稳定性 |

Track 模式评分：`total = completion_ratio × (0.4 × time + 0.4 × posture + 0.2 × energy)`，先保证完成率，再优化速度/姿态/能耗。

---

## 仓库结构

```
.
├── agent_ppo/                  # 当前主训练智能体（PPO + GAE）
│   ├── algorithm/              # PPO + GAE 实现（含 NaN/Inf 防护）
│   ├── conf/
│   │   ├── conf.py             # Stage 配置、模型维度、训练超参；Config.CURRENT 切换阶段
│   │   ├── monitor_builder.py  # 监控分组与指标面板
│   │   └── train_env_conf_track_nav.toml   # 当前 track 阶段配置（地形/命令/奖励/课程）
│   ├── feature/                # policy/critic 观测处理、自定义奖励、速度课程、地形门控等
│   ├── model/                  # Actor-Critic 网络
│   └── workflow/
│       └── train_workflow.py   # 主训练循环、速度课程、监控上报
├── agent_diy/                  # 备选智能体模板（含 standard / track 多阶段 TOML 配置模板）
├── conf/                       # 平台级 app/algo 配置
├── isaac_env/                  # Isaac 环境接口
├── arena_frontend_monitor/     # 训练监控数据抓取与后处理脚本
├── legged-robot/               # 赛题说明、开发指南与框架文档（公开材料归档）
├── docs/                       # 训练交接指南与调研记录
├── CHANGES.md                  # 代码与配置改动汇总
├── kaiwu.json                  # 平台元数据
└── train_test.py               # 本地快速测试入口
```

---

## 快速开始

项目依赖 KaiwuDRL 与 Isaac Lab，需在腾讯开悟（Tencent Kaiwu）训练环境或等价环境中运行。

```bash
# 本地快速冒烟测试（需 KaiwuDRL 环境）
python train_test.py

# 本地语法检查（无单元测试，用 py_compile 代替）
python3 -m py_compile agent_ppo/agent.py
python3 -m py_compile agent_ppo/workflow/train_workflow.py
python3 -m py_compile agent_ppo/feature/reward_process.py
```

`train_test.py` 中可切换 `algorithm_name`（`"ppo"` / `"diy"`），并设置了小批量快速验证参数。

**切换训练阶段**：修改 `agent_ppo/conf/conf.py` 中的 `Config.CURRENT`，并保证对应的
`train_env_conf_<task_type>_<stage.name>.toml` 存在。

---

## 算法概览

| 组件 | 详情 |
|---|---|
| 算法 | PPO + GAE（λ=0.95，γ=0.99） |
| 网络结构 | Asymmetric Actor-Critic |
| Actor | MLP [301→512→256→128→12]，ELU 激活 |
| Critic | MLP [316→512→256→128→1]，LayerNorm + ELU |
| Policy 观测 | 301 维 = 45 proprio + 256 height scan（track 阶段可附加 3 维目标特征） |
| Critic 观测 | 316 维 = 60 privileged proprio + 256 height scan |
| 动作 | 12 维连续动作，`[-1, 1]`，经 `action_scale=0.25` 映射到 PD 目标关节角 |

policy / critic 观测处理器内置**运行时维度断言**，用于尽早暴露 height scan 缺失或观测布局漂移。

### Track 观测补充

- `height_scanner` 已包含在默认 301 维 policy obs 的 `[45:301]`，是 16×16 前方高度扫描；
  自定义 reward/obs 中读取原始几何高度差：`sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]`。
- `nav_scanner` 默认不在 301 维 obs 中，约 13×11=143 rays 的前瞻遮挡扫描，适合迷宫转向/堵路判断；
  推荐先提取左/中/右 clearance 等紧凑特征，再决定是否扩展 actor 输入。
- Track 附加信息：`env.goal_positions`、`env.goal_yaw`、`env.scene.sensors["nav_scanner"]`。

---

## 环境协议

```python
obs, critic_obs = env.reset(usr_conf)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = env.step(actions)
dones = terminated | truncated
```

Policy 观测布局（301 维）：

| 区间 | 内容 |
|---|---|
| [0:3] | 机体角速度（×0.25） |
| [3:6] | 投影重力 |
| [6:9] | 速度指令 (vx, vy, wz) |
| [9:21] | 相对关节位置 |
| [21:33] | 相对关节速度（×0.05） |
| [33:45] | 上一时刻动作 |
| [45:301] | 16×16 高度扫描（clip [-5,5]，×2.5） |

Critic 观测（316 维）额外包含 `base_lin_vel` 与 `joint_effort` 特权信息。

---

## 当前训练配置

当前默认阶段：**TrackNavConfig**（路线导航微调），见 `agent_ppo/conf/conf.py` 的 `Config.CURRENT`。
切换回 `LocomotionConfig` 可进行 standard 模式训练（standard 配置模板见 `agent_diy/conf/`）。

训练配置要点：

- 地形课程（terrain curriculum）与速度课程（VelocityCurriculum）解耦，速度课程按 tracking ratio
  推进（升级：连续 2 次 ≥ 0.55；降级：连续 3 次 < 0.30），Stage 0 必须与 `[commands.ranges]` 一致。
- 自定义奖励遵循 `_reward_<name>` 模式，在 TOML 中通过 `[rewards.<name>]` 启用。
- standard 子地形比例之和必须为 1.0；track 子地形列表必须以 `open_entry_maze` 结尾。
- 模型每 `model_save_interval` 个 episode 保存一次 checkpoint。

> 注意：训练 reward 与竞赛评分（total_score / distance_score 等）不是同一指标，不要用 reward 绝对值直接对标评分。

---

## 监控与诊断

`arena_frontend_monitor/` 提供平台监控数据抓取与后处理脚本；监控面板分组由
`agent_ppo/conf/monitor_builder.py` 定义，覆盖训练进展、速度跟踪、姿态/步态质量、
稳定接触、能耗扭矩与物理观测量等 9 组指标。

---

## 改动记录与参考

- 改动记录见 [CHANGES.md](CHANGES.md)，训练交接说明见 [docs/TRAINING_HANDOFF_GUIDE.md](docs/TRAINING_HANDOFF_GUIDE.md)。
- 赛题与框架文档见 [legged-robot/README.md](legged-robot/README.md)（腾讯开悟公开文档归档）。
- 相关开源参考：[walk-these-ways (Improbable-AI)](https://github.com/Improbable-AI/walk-these-ways)

---

*本项目基于腾讯开悟四足机器人运动控制竞赛（legged_robot_competition_26）训练实践整理，仅供学习交流。*
