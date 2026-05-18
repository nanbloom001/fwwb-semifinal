# 2026-05-16 Reset yaw 探测记录

## 背景

为了判断是否需要在训练侧额外实现“reset 时固定 90 度倍数 yaw”，我探测了容器中的环境定义和 reset 事件实现。

## 探测到的容器源码位置

- `tools/base_env/base_env.py`
- `tools/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py`
- `tools/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`

## 结论

### 1. `init_state.pos` 只定义位置，不定义朝向

容器侧 `base_env.py` 只把 TOML 的：

```toml
[init_state]
pos = [x, y, z]
```

映射到 `robot_cfg.init_state.pos`。

因此：

- `pos` 是初始位置
- 不是 yaw / roll / pitch
- 不能用它设置初始朝向

### 2. Standard 模式默认 reset 已经随机 yaw

Standard 模式默认使用 `reset_root_state_maze_random`。
在 `events.py` 中，reset 逻辑明确包含：

```python
yaw = sample_uniform(-3.14159, 3.14159, (len(env_ids),), device=asset.device)
```

也就是说，Standard reset 已经是**全角度随机 yaw**，不是固定朝向。

### 3. 当前容器代码里没有现成的 TOML 入口给 Standard reset 传 `pose_range`

`reset_root_state_track_start` / `reset_root_state_eval_level_aware` 支持 `pose_range`，但 Standard 默认的 `reset_root_state_maze_random` 不接受这个参数。

`base_env.py` 里也没有把 `pose_range` 从 TOML 透传到 Standard reset 的现成入口。

## 设计判断

基于上述探测，当前训练里**没有必要再额外写一个 reset yaw 量化逻辑**。

原因是：

1. Standard reset 已经在全角度随机 yaw。
2. 额外强制 0 / 90 / 180 / 270 度会把连续分布改成离散分布，未必带来收益。
3. 当前对角线挑路的问题，更可能来自 reward / command 分布 / 世界方向偏置，而不是 reset yaw 不够多样。

## 后续建议

- 保留命令课程和路径一致性奖励。
- 不再依赖 reset yaw 量化 patch。
- 如果后面还要控制朝向，多半应该在平台 reset event 里做，而不是在训练 workflow 里二次 patch。
