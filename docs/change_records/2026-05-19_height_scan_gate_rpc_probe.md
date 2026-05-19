# 2026-05-19 Height scan gate RPC 探测记录

## 背景

本轮训练中新增的“门控阈值测试”监控面板共 22 个，包含
`g_*` 和 `gx_*` 指标，但测试结束后这些面板全部为空。与此同时，
`hs_*` 指标有上报点，但数值均为 0。因此需要确认问题到底来自：

- 前端监控读取脚本漏抓；
- 平台后端没有上报这些指标；
- reward / gate probe 代码没有执行；
- 对 `height_scanner` 数据语义理解错误。

本记录来自对开发容器的只读 RPC 探测。

## 已确认的监控事实

最新本地采集目录：

```text
arena_frontend_monitor_runtime/manual_metric_recorder/sessions/20260519-132027
```

该采集本身正常：

- `request_count = 207`
- `unique_metric_count = 513`
- `unique_point_count = 18433`
- `errors = []`

平台确实请求过 `g_*` / `gx_*` 指标：

- `gate_requests = 32`
- 全部返回 `item_count = 0, point_count = 0`

因此 22 个面板为空不是手动读取脚本漏抓，而是平台后端没有这些时间序列。

同时，`hs_*` 指标有上报，但全部为 0：

- `hs_up_ratio = 0`
- `hs_down_ratio = 0`
- `hs_wall_ratio = 0`
- `hs_step_score = 0`
- `hs_step_delta = 0`
- `hs_clear_active = 0`

`reward_hs_clearance` 有上报点，但数量级约为 `1e-7`，不能证明门控有效激活。

## 容器中探测到的源码位置

RPC 根目录：

```text
/data/projects/legged_robot_competition_26
```

开发工作区映射：

```text
/workspace/code
```

关键环境源码：

```text
/data/projects/legged_robot_competition_26/tools/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
/data/projects/legged_robot_competition_26/tools/base_env/observation_process.py
/data/projects/legged_robot_competition_26/isaac_env/base_env.py
```

当前 `agent_ppo` 工作区源码：

```text
/workspace/code/agent_ppo/feature/reward_process.py
/workspace/code/agent_ppo/conf/monitor_builder.py
/workspace/code/agent_ppo/workflow/train_workflow.py
```

## height_scanner 配置语义

容器中 `velocity_env_cfg.py` 的 `height_scanner` 配置如下：

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

源码注释明确说明：

- `16x16 = 256` rays。
- 覆盖机器人前方 `0 ~ 1.5m`。
- 横向覆盖约 `±0.75m`。
- 网格分辨率 `0.1m`。
- `ray_alignment = "yaw"`，即跟随 yaw，但不跟随完整 pitch/roll。

这意味着门控代码中如果假设第二维/第三维方向，必须和该网格约定对齐：

- 16x16 reshape 后一般应理解为 `(num_envs, lateral_y, forward_x)`。
- x 方向索引递增通常代表更远的前方。
- y 方向索引代表横向。

## 默认 observation 中 height_scan 的公式

容器中 `tools/base_env/observation_process.py` 的 helper 定义：

```python
scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
```

默认参数：

```python
offset = 0.5
clip_range = (-1.0, 5.0)
```

在 `velocity_env_cfg.py` 的 policy / critic observation 中，height scan 配置为：

```python
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    scale=2.5,
    clip=(-5.0, 5.0),
)
```

因此默认 301 维 policy observation 中的 height scan 不是 raw hit z，而是：

```text
(scanner_z - hit_z - 0.5) * 2.5, then clipped to [-5, 5]
```

当前 reward 代码中的 `_height_scan_grid()` 使用：

```python
scan = sensor.data.pos_w[:, 2:3] - ray_hits[..., 2]
```

这和 policy observation 不是同一尺度：没有减 `0.5`，也没有乘 `2.5` 或 clip。
这不一定错误，但后续阈值必须基于该 raw relative height 重新校准。

## 关键语义：height_scan 不是世界 z 高度图

当前仓库的 `agent_ppo/feature/height_scan_features.py` 已经写明约定：

```text
The scan convention used by the environment is scanner_z - hit_z.
Values below zero mean the ray hit something above the scanner plane.
```

因此 height scan 是“扫描器高度减命中点高度”，不是“地面世界 z”。

在这个语义下：

- 前方地形更高时，`scanner_z - hit_z` 会变小。
- 前方地形更低时，`scanner_z - hit_z` 会变大。
- 如果用 `front - near` 判断台阶方向，正负号和“世界高度差”的直觉相反。

这直接影响当前 `_height_scan_semantic_gate()` 的上/下台阶判定。

## 为什么 22 个 g/gx 面板为空

当前 `agent_ppo/feature/reward_process.py` 中，`g_*` / `gx_*` 指标由
`_probe_height_scan_gate_variants()` 写入。

该函数存在静默返回路径：

```python
hit_grid = self._height_scan_hit_grid()
if hit_grid is None:
    return
```

如果 `_height_scan_hit_grid()` 不可用，函数不会写任何 `g_*` / `gx_*` key。
平台监控看到的就是这些指标从未上报，面板为空。

另外，`_height_scan_semantic_gate()` 在 `near_z`、`front_z` 或 `grid` 为空时也会提前返回，
此时不会调用 `_probe_height_scan_gate_variants()`。

因此：

- `g_* / gx_*` 面板为空表示“探针没有稳定上报指标”。
- 不能用这些空面板判断阈值好坏。
- `hs_* = 0` 表示主门控有上报，但当前判定没有识别出上/下台阶/墙体。

## 当前判断

当前门控测试没有达到“多阈值比较”的目的。

主要问题不是前端读取脚本，也不是训练曲线后处理，而是诊断 probe 本身不够可靠：

1. `g_* / gx_*` probe 有静默不上报路径。
2. probe 单独依赖 raw hit z，而不是和主门控使用同一 height scan 语义。
3. 上/下台阶方向可能按世界高度直觉写反。
4. 当前 `hs_*` 全 0，说明即使主门控已上报，也没有有效识别 stair-like geometry。

## 下一步修复建议

先修诊断层，不要继续盲调阈值。

建议修改：

1. `_probe_height_scan_gate_variants()` 不允许静默 return。
   即使不可用，也上报所有 `g_* / gx_* = 0`，并额外上报：

   ```text
   g_probe_available = 0/1
   g_probe_source = 0/1/2
   ```

2. 多阈值 probe 优先复用 `_height_scan_grid()` 的 raw relative height：

   ```python
   scanner_z - hit_z
   ```

   raw world `hit_z` 可以作为额外对照，但不应作为唯一 probe 数据源。

3. 加 warning 日志：

   ```text
   [height_scan_probe] call=... available=... source=... reason=...
   ```

4. 按 `scanner_z - hit_z` 语义重新定义上/下台阶方向。
   不能继续把 `front - near > 0` 直接理解为“前方更高”。

5. 在修复 probe 之前，暂时不要把 `height_scan_feet_clearance` 和
   `height_scan_wall_reject` 作为可靠训练塑形依据，只用于低权重测试或关闭。

## 关于把容器依赖代码复制到本地

为了便于后续分析，可以把容器中可读的依赖源码镜像到本地，但必须和常规训练代码分离。

推荐本地目录：

```text
external/container_src/
```

建议只复制只读源码：

```text
/data/projects/legged_robot_competition_26/tools/base_env/
/data/projects/legged_robot_competition_26/tools/unitree_rl_lab/
/data/projects/legged_robot_competition_26/isaac_env/
/data/projects/legged_robot_competition_26/kaiwudrl/
```

不建议复制：

```text
/workspace/code/
/data/projects/legged_robot_competition_26/log/
agent_diy/codex_rpc_bridge_runtime/
训练输出、模型、视频、上传包、备份
```

这些镜像源码应视为分析参考，不应参与平台提交或同步到容器。
如果引入该目录，需要确保同步脚本和 `.gitignore` / 打包规则排除：

```text
external/
external/container_src/
```

## 后续修复与门控实验最终结论

在后续修复中，训练 workflow 改为直接从 rollout `critic_obs[:, 60:316]`
计算 height scan probe，而不是依赖 reward 侧 `_stair_gate_debug`。这样绕过了
Kaiwu/Isaac wrapper 下 reward debug dict 难以稳定暴露的问题。

当前可靠 probe 链路：

- `g_probe_available = 1`
- `g_probe_reason_code = 10`
- `hs_probe_available = 1`
- `hs_probe_reason = 10`

其中 `reason_code = 10` 表示 rollout height scan probe 成功执行。

新增并验证的监控指标包括：

- 方向互斥门控：
  - `g_dom_up_02`
  - `g_dom_down_02`
  - `g_dom_both_02`
  - `g_dom_none_02`
  - 以及 margin 0.05 / 0.08 对照版本。
- 台阶/缓坡/墙体结构门控：
  - `g_step_edge_sharpness`
  - `g_step_edge_locality`
  - `g_slope_smoothness`
  - `g_struct_wall`
  - `g_stair_l/m/s`
  - `g_slope_l/m/s`
  - `g_stair_noslope_l/m/s`

### 单项地形测试结果

#### 纯台阶 + 反台阶

配置：

```toml
pyramid_slope = 0.0
pyramid_slope_inv = 0.0
pyramid_stairs = 0.50
pyramid_stairs_inv = 0.50
maze = 0.0
```

稳定段结果：

```text
g_stair_m          ≈ 0.65
g_stair_noslope_m  ≈ 0.65
g_step_edge_sharpness ≈ 0.054
g_dom_both_02      ≈ 0.04
```

结论：台阶结构可以被 `g_stair_noslope_m` 稳定识别；上/下方向互斥门控
没有明显双向冲突。

#### 纯缓坡 + 反缓坡

配置：

```toml
pyramid_slope = 0.50
pyramid_slope_inv = 0.50
pyramid_stairs = 0.0
pyramid_stairs_inv = 0.0
maze = 0.0
```

结果：

```text
g_stair_l/m/s         = 0
g_stair_noslope_l/m/s = 0
g_step_edge_sharpness ≈ 0.0026
g_slope_l             ≈ 0.012
g_slope_m             ≈ 0
```

结论：当前台阶 gate 不会在缓坡上误触发。`g_slope_*` 对缓坡的正向识别
偏弱，暂时不应作为可靠的缓坡分类器；但这不影响用
`g_stair_noslope_m` 作为台阶奖励 gate 的安全性判断。

#### 全迷宫

配置：

```toml
pyramid_slope = 0.0
pyramid_slope_inv = 0.0
pyramid_stairs = 0.0
pyramid_stairs_inv = 0.0
maze = 1.0
```

结果：

```text
hs_wall_ratio       ≈ 0.55
hs_wall_score       ≈ 0.10
g_struct_wall       ≈ 0.68
g_step_edge_sharpness ≈ 0.34
g_stair_l/m/s       = 0
g_stair_noslope_l/m/s = 0
g_dom_up/down/both_02 = 0
```

结论：迷宫/墙体能被 wall 相关指标识别；强高度突变没有污染台阶 gate。
这是 `g_stair_noslope_m` 可以继续作为候选 gate 的关键证据。

### 混合地形测试结果

#### 无迷宫四类混合

配置：

```toml
pyramid_slope = 0.25
pyramid_slope_inv = 0.25
pyramid_stairs = 0.25
pyramid_stairs_inv = 0.25
maze = 0.0
```

稳定段结果：

```text
g_stair_noslope_m ≈ 0.33
g_dom_up_02       ≈ 0.17
g_dom_down_02     ≈ 0.18
g_dom_both_02     ≈ 0.017
g_struct_wall     = 0
```

结论：在真实的 slope + stairs 混合分布中，台阶 gate 有稳定非零激活；
激活率低于纯台阶且高于纯缓坡，符合预期。

#### 含迷宫混合

配置：

```toml
pyramid_slope = 0.15
pyramid_slope_inv = 0.15
pyramid_stairs = 0.25
pyramid_stairs_inv = 0.25
maze = 0.20
```

记录点较少，当前结果：

```text
hs_wall_ratio       ≈ 0.11
hs_wall_score       ≈ 0.020
g_struct_wall       ≈ 0.13
g_stair_noslope_m   ≈ 0.20 mean, latest ≈ 0.35
g_dom_both_02       ≈ 0.012
```

结论：加入 20% maze 后，wall 指标有反应，台阶 gate 没有表现出明显
maze 污染。由于记录点偏少且 `g_stair_noslope_m` 仍在上升，建议后续
再补一段更稳定的采样，但当前结果没有发现阻断性问题。

### 最终判断

当前最可信的台阶奖励候选 gate 是：

```text
g_stair_noslope_m
```

方向判断可以优先使用：

```text
g_dom_up_02
g_dom_down_02
```

暂时不建议依赖：

```text
g_slope_l/m/s
```

原因是纯缓坡测试中 `g_slope_m` 基本为 0，说明当前 slope-like 正向分类
不够敏感。

### 对后续奖励设计的建议

可以进入“小权重台阶辅助奖励”阶段，但不要大幅塑形。

建议只把 `g_stair_noslope_m` 用作 gate，先接入低权重、容易监控的
height-scan 抬脚/台阶辅助奖励，例如：

```text
height_scan_stair_clearance: 0.03 -> 0.08
```

或等价的小权重台阶抬脚奖励。初始阶段应继续保留清晰监控：

```text
g_stair_noslope_m
g_dom_up_02
g_dom_down_02
g_dom_both_02
reward_height_scan_feet_clearance
```

若接入奖励后出现台阶前转向、内八/外八步态、姿态退化或缓坡异常抬腿，
应立即回退 reward 权重，而不是继续提高 stair reward。

### 仍需注意

这些 gate 目前是基于 rollout critic observation 的监控 probe。若未来要在
reward 函数中实际使用，应保证 reward 侧计算使用同一套 height scan 语义和
阈值，不能重新回到早期 `_stair_gate_debug` / reward-side raw hit z 的不稳定路径。
