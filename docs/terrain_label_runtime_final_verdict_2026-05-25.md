# Terrain Label Runtime Final Verdict - 2026-05-25

## 结论

在当前限制下，也就是**只能修改 `agent_ppo`，不能修改 Kaiwu/IsaacLab 容器侧
`external/container_src`、base env、scorer、worker 或 RPC/infos 回传协议**，训练中
不能可靠读取 per-env 地形难度/类型标签，也不能可靠用于 reward 门控或难度分级调优。

更精确地说：

- 底层 Isaac 环境中很可能存在 `scene.terrain.terrain_levels` 和
  `scene.terrain.terrain_types`。
- 但 `agent_ppo` 的 aisrv/workflow 进程无法直接拿到底层 Isaac env。
- `agent_ppo` 内的 reward hook / module global / `_stair_gate_debug` 等尝试，都没有形成
  可验证、可回传、可监控的稳定链路。
- 因此后续不要再基于 `agent_ppo` 直接 terrain label 读取来做训练门控。

## 已验证证据

### 1. TOML 配置能加载

日志确认当前训练确实加载了 active TOML：

```text
[TerrainLevelRewardProbeConfig] conf_file=agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml,
hook=track_lin_vel_xy, configured=1, weight=2.4,
params={'std': 0.25, 'command_name': 'base_velocity'}
```

这排除了“训练没有加载当前配置文件”的可能。

### 2. workflow 侧无法 unwrap 到底层 Isaac env

最终诊断日志：

```text
[WorkflowTerrainProbe] available=0 reason=env_missing
candidates=root:kaiwudrl.components.environment.env_wrapper.EnvWrapper
keys=['logger', 'monitor_proxy', 'env_object', '_final_terrain_diag_logged']
| root.env_object:kaiwudrl.components.environment.isaac_env.IsaacEnv
keys=['ctx', 'cmd_queue', 'data_queue', 'worker']
```

这说明 aisrv/workflow 看到的是 Kaiwu 的 wrapper 和 worker 队列，真实 Isaac env 在
worker 侧，不在当前 Python 对象图里。

### 3. workflow 读取模块级状态失败

多轮训练日志持续为：

```text
[TerrainLevelRewardProbe] episode=9, calls=0, available=0,
reason=reward_not_called, level=(-1,-1,0.00), num_envs=0
```

后续 episode 17、25、32 仍然是 `calls=0`。

这不能证明 reward hook 一定没有在 worker 进程执行；更可能说明 worker/reward 执行环境
和 aisrv/workflow 不是同一个 Python 模块状态空间，workflow 读不到 worker 里的
module global。

### 4. reward 内部直接日志没有出现

最终版本还在 `_reward_track_lin_vel_xy()` 开头加入了直接日志：

```text
[TrackLinVelHookDirect] entered ...
```

运行日志中没有出现该行。结合 workflow 无法 unwrap 到 worker，可判断当前 aisrv 日志流
无法可靠观察 reward hook 内部执行，或者实际 reward override 没进入我们可观察的路径。

### 5. 独立 reward term 方案失败

曾尝试新增：

```toml
[rewards.terrain_level_probe]
weight = 1.0
```

并实现 `_reward_terrain_level_probe()`，但运行中始终没有调用证据，workflow 仍为
`calls=0 reason=reward_not_called`。该方案已删除，不应恢复。

## 尝试过但不可用的方案

- workflow 中直接 unwrap `env.scene.terrain`：失败，`env_missing`。
- `_sample_standard_terrain_label_stats()` 直接采样：失败，`unwrap_failed`。
- reward 写 `_stair_gate_debug`，workflow 扫 wrapper 链读取：不可靠。
- reward 写模块级 `TERRAIN_LEVEL_PROBE_STATE`，workflow import 读取：跨进程不可验证。
- 新增独立 reward term `_reward_terrain_level_probe()`：无调用证据。
- 将 terrain probe 挂到 `_reward_track_lin_vel_xy()`：workflow 仍读不到调用状态，且直接
  hook 日志未出现在 aisrv 日志流。

## 后续建议

### 不建议继续做

- 不要在只改 `agent_ppo` 的前提下继续尝试直接读取
  `self.env.scene.terrain.terrain_levels` / `terrain_types` 做训练门控。
- 不要基于 `_standard_terrain_mask()` 是否“代码存在”来认定运行时可用。
- 不要将旧静态源码报告中的“底层可能可读”误解为“agent_ppo 训练链路可用”。

### 推荐替代方案

继续使用 observation 几何推断：

- `height_scanner` / 16x16 height scan 判断上下台阶、墙、坡、平坦区域。
- `nav_scanner` 做更远距离的通行空间、障碍和目标方向辅助。
- 用 episode/failure/score 的平台指标做结果分析，而不是作为 per-step label gate。

如果未来允许修改容器侧环境，推荐在以下位置之一做官方回传：

- `external/container_src/current/isaac_env/base_env.py`
- `external/container_src/current/tools/base_env/base_scorer.py`
- `env.step()` 返回的 `infos`
- monitor/scorer 已有 terrain-level 统计通道

只有当 worker/base_env 主动把 per-env `terrain_levels` / `terrain_types` 回传到 aisrv 或
monitor 时，才适合做可靠的 label-gated reward 或难度分级调优。

## 对旧文档的修正

`docs/terrain_level_label_repository_search_2026-05-24.md` 是静态源码搜索报告，结论是
底层 runtime wrapper/scorer 很可能具备 terrain label。该结论没有错，但不等价于
`agent_ppo` 可直接读取。

本文件是运行时最终判定：**底层可能有标签，但只改 `agent_ppo` 时不可可靠访问和使用。**
