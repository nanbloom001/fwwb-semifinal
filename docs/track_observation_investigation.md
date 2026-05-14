# Track Observation Investigation

## 背景

最终评估需要在 Track 模式完成赛道。Track 模式相比 Standard 模式额外提供目标点
和更宽的前瞻扫描，但这些额外输入不会自动进入默认 301 维 policy observation。
本文件记录本地仓库与开发容器中的实际探查结论，重点说明：

- `goal_position_in_robot_frame()` 当前的调用风险。
- `height_scanner` 的实际实现与调用方式。
- `nav_scanner` 的实际实现与推荐调用方式。
- 后续如果扩展 Track 观测，需要同步修改哪些位置。

## 默认 Observation

默认 policy observation 仍是 301 维：

```text
obs = [proprio(45) | height_scan(256)]
```

布局：

```text
[0:3]     base angular velocity, scale=0.25
[3:6]     projected gravity
[6:9]     velocity command (vx, vy, wz)
[9:21]    relative joint positions
[21:33]   relative joint velocities, scale=0.05
[33:45]   previous action
[45:301]  height_scan, 16x16 = 256
```

默认 critic observation 是 316 维：

```text
critic_obs = [critic_proprio(60) | height_scan(256)]
```

critic 额外包含 `base_lin_vel` 和 `joint_effort`。

## Track 额外原始输入

开发指南和容器代码确认，Track 模式额外提供：

```python
env.goal_positions              # (num_envs, 3), 赛道终点/出口世界坐标
env.goal_yaw                    # (num_envs,), 出口朝向
env.scene.sensors["nav_scanner"] # RayCaster, 前瞻遮挡扫描
```

这些输入不是默认 301 维 observation 的一部分。若要给 actor/critic 使用，必须在
`agent_ppo/feature/policy_observation_process.py` 和
`agent_ppo/feature/critic_observation_process.py` 中自行拼接或提取紧凑特征，并同步
更新 stage 维度、模型输入维度和运行时断言。

## goal_position_in_robot_frame 调用风险

当前多个分支中都调用了：

```python
self.goal_position_in_robot_frame()
```

当前 `agent_ppo` 的调用路径是：

```python
# agent_ppo/feature/policy_observation_process.py
if mode == "track_goal":
    return self.goal_position_in_robot_frame()
```

critic observation process 中也有同样调用。该调用在以下条件下会触发：

```text
stage.extra_obs_mode == "track_goal"
```

也就是启用 `TrackNavConfig`，例如：

```bash
FWWB_STAGE=nav python train_test.py
FWWB_STAGE=track_nav python train_test.py
```

探查结果：

- 本地所有 refs 中有多处分支调用该方法。
- 开发容器 `/data/projects` 下也有调用。
- 但本地和容器均未找到任何实际定义：

```python
def goal_position_in_robot_frame(...):
    ...
```

因此当前风险是：一旦真正进入 `track_goal` 路径，Python 很可能抛出：

```text
AttributeError: 'PolicyObservationProcess' object has no attribute 'goal_position_in_robot_frame'
```

这不表示 `env.goal_positions` 不存在。容器代码显示 Track 模式下
`env.goal_positions` 是由 `base_env.step()` 主动维护的环境基础设施。缺失的是把
`env.goal_positions` 和 `env.goal_yaw` 转换为 actor/critic 可用特征的 helper。

### 推荐实现语义

如果继续使用 4 维 `track_goal` 观测，建议显式实现以下语义：

```text
goal_obs = [
  relative_goal_x_in_robot_frame,
  relative_goal_y_in_robot_frame,
  goal_distance,
  wrapped_goal_yaw_error,
]
```

其中：

- `relative_goal_*_in_robot_frame`：目标相对机器人 base 的 2D 向量，旋转到机器人坐标系。
- `goal_distance`：机器人到目标的 2D 欧氏距离。
- `wrapped_goal_yaw_error`：目标/出口朝向和机器人 yaw 的差，归一化到 `[-pi, pi]`。

实现时不要依赖不存在的基类 helper，应在本地 observation process 或共享工具函数中
显式实现。

## Track 目标点维护机制

容器中 `tools/base_env/observation_process.py` 提供模块级函数：

```python
ensure_goal_positions_ready(env)
update_goal_positions(env)
```

容器中 `tools/base_env/base_env.py` 在 `task_type == "track"` 时每步调用它们：

```python
if self._task_type == "track":
    from tools.base_env.observation_process import (
        ensure_goal_positions_ready,
        update_goal_positions,
    )

    env_unwrapped = self._gym_env.unwrapped
    if ensure_goal_positions_ready(env_unwrapped):
        update_goal_positions(env_unwrapped)
```

所以 Track 模式下：

- `env.goal_positions` 不依赖用户是否把 goal 拼进 obs。
- scorer、termination、reward 都可以读取该目标点。
- 如果 `env.goal_positions` 缺失或全零，base scorer 会在 Track 首个 done 帧做强校验。

`TerrainExitManager.get_track_goal_positions()` 当前固定返回赛道最后一段出口：

```text
finish_row = track_length - 1
```

也就是 `open_entry_maze` 的出口。它不是逐段子目标，不会根据机器人 X 坐标逐段递进。

## height_scanner 实现与调用

容器中 `velocity_env_cfg.py` 的配置：

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

含义：

- RayCaster 挂在机器人 base 上。
- 网格中心在机器人前方 `0.75m`，高度 `20m`。
- 覆盖前方 `1.5m x 1.5m`。
- 分辨率 `0.1m`，共 `16x16 = 256` 条 ray。
- `ray_alignment="yaw"`：只跟随机器人 yaw，不跟随 pitch/roll。

默认 observation term：

```python
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    scale=2.5,
    clip=(-5.0, 5.0),
    noise=Unoise(n_min=-0.1, n_max=0.1),
)
```

因此默认 policy obs 中的 `obs[:, 45:301]` 已经是：

```text
height_scanner 原始高度差 -> scale=2.5 -> clip[-5, 5] -> 可选 observation noise
```

### 从 observation 中读取

在 policy observation process 中直接使用默认 obs：

```python
obs = self.default_observation()
height_scan = obs[:, 45:301]
height_grid = height_scan.view(obs.shape[0], 16, 16)
```

注意：这里读取的是已经缩放、裁剪、加噪后的 observation 值。

### 从 sensor 读取原始几何

在 reward 或自定义 observation 中直接访问 sensor：

```python
sensor = self.env.scene.sensors["height_scanner"]
scan_raw = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
height_grid_raw = scan_raw[:, :256].view(self.env.num_envs, 16, 16)
```

注意：

- `scan_raw` 未经过默认 observation term 的 `scale=2.5` 和 `clip`。
- 如果复用基于 `obs[:, 45:301]` 的阈值，需要把米制阈值乘以 `2.5`。
- 如果直接使用 raw sensor 值，阈值应使用米制尺度。

## nav_scanner 实现与调用

容器中 `velocity_env_cfg.py` 的配置：

```python
nav_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(1.25, 0.0, 20.0)),
    ray_alignment="base",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.5, 2.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

含义：

- RayCaster 挂在机器人 base 上。
- 网格中心在机器人前方 `1.25m`。
- 覆盖前方 `[0, 2.5m]`，横向 `[-1m, 1m]`。
- 分辨率 `0.2m`。
- 约 `13x11 = 143` 条 ray。
- `ray_alignment="base"`：跟随完整 base 姿态，包括 pitch/roll。

容器注释说明该设计用于：

- 迷宫前瞻转向。
- 约能看到 `1.5-2` 个 maze cell。
- 通过 `base` 对齐减少上下坡/楼梯时把地面误判成墙的概率。

默认 301 维 observation 不包含 `nav_scanner`。必须手动读取。

### 原始调用方式

```python
nav = self.env.scene.sensors["nav_scanner"]
ray_vec = nav.data.ray_hits_w - nav.data.pos_w.unsqueeze(1)

# XY 平面距离，shape: (num_envs, 143)
dist_xy = torch.linalg.norm(ray_vec[..., :2], dim=-1)

# 垂直高度差，shape: (num_envs, 143)
height = nav.data.pos_w[:, 2:3] - nav.data.ray_hits_w[..., 2]
```

当前 `agent_ppo/feature/reward_process.py` 中已有类似写法：

```python
nav_scanner = self.env.scene.sensors.get("nav_scanner")
if nav_scanner is not None and hasattr(nav_scanner.data, "ray_hits_w"):
    ray_vec = nav_scanner.data.ray_hits_w - nav_scanner.data.pos_w.unsqueeze(1)
    dist = torch.linalg.norm(ray_vec[..., :2], dim=-1)
```

### 推荐特征工程

不建议一开始把完整 143 维 `nav_scanner` 全量拼进 actor。更稳妥的第一步是提取紧凑特征：

```python
nav = self.env.scene.sensors["nav_scanner"]
height = nav.data.pos_w[:, 2:3] - nav.data.ray_hits_w[..., 2]

# 具体行列顺序建议在容器里 assert 一次；当前规格约 143 rays。
# 如果按 11x13 组织，可先用如下方式试验：
nav_grid = height.view(self.env.num_envs, 11, 13)

left = nav_grid[:, :, :4].mean(dim=(1, 2))
center = nav_grid[:, :, 4:9].mean(dim=(1, 2))
right = nav_grid[:, :, 9:].mean(dim=(1, 2))

nav_features = torch.stack([left, center, right], dim=1)
```

更实用的特征可以包括：

- 左/中/右前方最小距离或平均 clearance。
- 中心前方是否堵塞。
- 左右通道可行性差值，作为 turn bias。
- 前方近距离墙体分数。
- L 型拐角前兆特征。

`height_scan_features.py` 中已有面向 16x16 height scan 的墙体/台阶几何判定逻辑，可作为
`nav_scanner` 压缩特征的参考，但不要直接复用 16x16 reshape 假设。

## 扩展 Track Observation 时必须同步修改

如果只实现 4 维 goal obs：

- `agent_ppo/feature/policy_observation_process.py`
- `agent_ppo/feature/critic_observation_process.py`
- `agent_ppo/conf/conf.py`
  - `TrackNavConfig.num_extra_obs = 4`
  - `TrackNavConfig.num_critic_observations = 320`
- 运行时维度断言：
  - policy expected dim = `301 + 4 = 305`
  - critic expected dim = `316 + 4 = 320`

如果再追加 compact nav features，例如 3 维：

```text
policy obs = 301 + 4 goal + 3 nav = 308
critic obs = 316 + 4 goal + 3 nav = 323
```

对应需要同步：

- `TrackNavConfig.num_extra_obs = 7`
- `TrackNavConfig.num_critic_observations = 323`
- actor/critic storage 输入维度会由 `agent_ppo/agent.py` 从 stage 自动读取。
- policy/critic observation process 必须拼接相同任务特征，保持 actor/critic 的任务信息约定一致。

## 当前建议

Track 模式的第一优先级是完成率，而不是速度。建议按以下顺序推进：

1. 先显式实现 4 维 goal observation，修复 `goal_position_in_robot_frame()` 缺失风险。
2. 验证 `FWWB_STAGE=nav` 能 reset、step，并且 policy/critic 维度分别为 `305/320`。
3. 训练或评估观察是否能朝最终出口方向产生稳定进展。
4. 再加入 compact `nav_scanner` 特征，用于迷宫段转向和堵路判断。
5. 最后再考虑全量 143 维 nav scan 或更复杂的局部规划特征。

不要在尚未能稳定完成 Track 的情况下优先优化时间分。Track 总分先乘完成系数，完成为 0 时
时间、姿态和能耗都不会带来有效总分。
