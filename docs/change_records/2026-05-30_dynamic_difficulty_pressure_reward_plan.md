# Dynamic Difficulty Pressure Reward Plan

## 背景

最新日志 `20260530-020000` 剔除前 8 分钟后，按 L0-L9 末段 20 点均值拟合发现：

- `energy_score` 与难度等级几乎线性反比：`R^2 ~= 0.99`
- `pose_score` 与难度等级强线性反比：`R^2 ~= 0.91`

这说明能耗分和姿态分可以作为“当前难度压力”的强代理信号。它不能代表真实地形等级标签，但可以表示当前 env 是否处于高压力、高风险状态。

## 结论

该方案技术上可行，适合作为高难完成率保护的备选方案，但不建议立刻作为主训练奖励启用。

原因：

- 优点：不依赖地形标签、不依赖门控，直接从模型实际表现中估计困难程度。
- 风险：低难度上如果姿态/能耗表现差，也会被判成高压力，可能强化低难度坏动作。
- 当前更稳的优先方案仍是权重微调，保护 completion 和高难稳定性。

## 推荐安全实现

新增一个弱奖励项，例如：

```text
difficulty_pressure_complete
```

语义：

```text
只在 task_complete 时，根据动态难度压力给额外完成奖励。
```

不要直接奖励 `difficulty_pressure` 本身，避免模型故意制造高能耗或坏姿态。

### 动态压力计算

使用 `energy_score_formula` 和 `pose_score_formula` 的 EMA 统计，而不是固定区间或当前 batch min/max。

推荐逻辑：

```python
energy_pressure = clamp((energy_mean - energy_score) / (2 * energy_std), 0, 1)
pose_pressure = clamp((pose_mean - pose_score) / (2 * pose_std), 0, 1)
difficulty_pressure = 0.6 * energy_pressure + 0.4 * pose_pressure
```

其中：

- `energy_mean/std`、`pose_mean/std` 使用 EMA 更新。
- EMA 更新必须 `detach()`。
- `std` 设置下限，避免除零。
- 冷启动阶段 pressure 默认为 0 或禁用 bonus。

### 高压阈值

只奖励高压力完成，不奖励低压/中压完成：

```python
bonus_pressure = clamp((difficulty_pressure - 0.55) / 0.45, 0, 1)
reward = bonus_pressure.detach() * task_complete
```

### 初始权重

建议保守启动：

```toml
[rewards.difficulty_pressure_complete]
weight = 10.0
```

不要初始使用 `20~30`，因为动态归一化存在冷启动和分布漂移风险。

## 风险控制

该 reward 必须满足：

- 只训练启用，评估不需要启用。
- 只在完成时触发。
- pressure 本身必须 detach。
- 不改变 actor/critic 观测维度。
- 不替代 `task_complete`，只作为弱额外奖励。

需要增加监控：

- `reward_difficulty_pressure_complete`
- `difficulty_pressure_mean`
- `difficulty_pressure_bonus_active`
- `difficulty_pressure_energy_component`
- `difficulty_pressure_pose_component`

## 当前不立即启用的理由

当前已完成一轮“直接评分奖励增强 + 高重叠代理降权”的调整。应先短训验证：

- `completion_pct >= 97%`
- `abnormal_pct <= 3%`
- `timeout_pct = 0`
- L8/L9 abnormal 不继续升高
- `pose_score`、`energy_score` 不明显下降

如果短训后高难度仍然明显拖累完成率，再考虑启用该弱 bonus。

## 更稳的短期替代方案

优先考虑以下低风险权重回调：

```toml
task_complete = 275.0
track_lin_vel_xy = 1.16
goal_velocity_projection = 2.50
ang_vel_xy = -0.60
joint_acc = -5.0e-7
```

这些改动不引入新状态，也不依赖动态统计，适合比赛前的短周期验证。
