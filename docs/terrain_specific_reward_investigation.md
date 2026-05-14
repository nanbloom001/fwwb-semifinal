# Terrain-Specific Reward Investigation

## 背景

问题：训练时能否直接根据当前地形类型设计只针对某个地形的 reward？

本文件记录开发容器中的只读源码探查结论。核心结论是：

- Standard 模式可以较直接地使用 `env.scene.terrain.terrain_types` 做 per-env 地形 mask。
- Track 模式不能把 `terrain_types` 当作当前子地形类型；它表示并行赛道列/难度档。
- Track 模式如果要做“只在某个子地形段启用”的 reward，应根据机器人世界坐标 `x`
  计算当前 track segment，再映射到 `sub_terrains_order`。

## 容器中确认存在的运行时字段

容器源码中多个位置明确使用：

```python
env_unwrapped.scene.terrain.terrain_types
env_unwrapped.scene.terrain.terrain_levels
```

这些不是推测字段，而是容器环境代码实际访问的字段。

关键证据位于：

```text
tools/base_env/base_scorer.py
tools/base_env/base_env.py
tools/unitree_rl_lab/unitree_rl_lab/terrains/track_generator.py
```

`base_scorer.py` 初始化时读取：

```python
terrain = env_unwrapped.scene.terrain
terrain_cfg = terrain.cfg
terrain_gen_cfg = terrain_cfg.terrain_generator

self.terrain_names = list(terrain_gen_cfg.sub_terrains.keys())

proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()])
proportions = proportions / proportions.sum()
num_cols = terrain_gen_cfg.num_cols

col_to_sub_idx = []
cumsum = np.cumsum(proportions)
for col_i in range(num_cols):
    sub_idx = int(np.min(np.where(col_i / num_cols + 0.001 < cumsum)[0]))
    col_to_sub_idx.append(sub_idx)
col_to_sub_idx_tensor = torch.tensor(col_to_sub_idx, dtype=torch.long, device=terrain.device)

self._env_terrain_name_idx = col_to_sub_idx_tensor[terrain.terrain_types.long()]
```

`base_scorer.py` 在 done 统计时也读取：

```python
done_cols = env_unwrapped.scene.terrain.terrain_types[done_env_ids]
```

源码注释说明：

```text
terrain.terrain_types: (num_envs,) 列索引 -> 映射到 sub_terrain 名称索引
terrain_types 由地形生成阶段固定分配，curriculum 不会在 step 内修改
```

`base_env.py` 也读取：

```python
terrain_types = getattr(terrain, "terrain_types", None)
terrain_levels = getattr(terrain, "terrain_levels", None)
```

并有注释说明首次 reset 后 `terrain_types` 会被填充。

## 训练时 reward 中能否访问

可以。`agent_ppo/feature/reward_process.py` 中的 reward 函数运行时有 `self.env`，
而容器 scorer/base_env 都是从同一个环境对象访问：

```python
self.env.scene.terrain.terrain_types
self.env.scene.terrain.terrain_levels
```

因此在自定义 reward 中可以尝试读取：

```python
terrain = self.env.scene.terrain
terrain_types = terrain.terrain_types
terrain_levels = terrain.terrain_levels
```

注意事项：

- `terrain_types` 通常在 reset 后才有真实分配。
- `terrain_types` 是固定列索引，容器注释说明 curriculum 不会在 step 内修改它。
- `terrain_levels` 会随 terrain curriculum 变化。scorer 为了 done 统计准确，会在
  `env.step()` 前拍 `pre_step_terrain_levels` 快照；reward 中直接读取的是当前 step
  环境状态，适合即时 mask，但不适合做严格 episode 归因统计。

## Standard 模式语义

Standard 地形是二维 grid：

- row / `terrain_levels`：难度级别。
- col / `terrain_types`：地形列。
- `terrain_gen_cfg.sub_terrains.keys()`：地形名称配置顺序。
- `sub_cfg.proportion`：各地形列比例。

Standard 模式下，`terrain_types` 可以通过比例映射到子地形名称。scorer 正是这样做的。

Standard 合法地形包括：

```text
pyramid_slope
pyramid_slope_inv
pyramid_stairs
pyramid_stairs_inv
maze
```

### Standard 地形 mask 推荐实现

可以在 `agent_ppo/feature/reward_process.py` 中增加 helper：

```python
def _standard_terrain_mask(self, target_name: str):
    terrain = self.env.scene.terrain
    terrain_gen_cfg = terrain.cfg.terrain_generator
    terrain_names = list(terrain_gen_cfg.sub_terrains.keys())

    if target_name not in terrain_names:
        return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    proportions = torch.tensor(
        [float(terrain_gen_cfg.sub_terrains[name].proportion) for name in terrain_names],
        device=self.env.device,
        dtype=torch.float32,
    )
    proportions = proportions / torch.clamp(proportions.sum(), min=1.0e-6)
    cumsum = torch.cumsum(proportions, dim=0)

    num_cols = int(terrain_gen_cfg.num_cols)
    col = terrain.terrain_types.long()
    choice = col.float() / float(num_cols) + 0.001
    sub_idx = torch.searchsorted(cumsum, choice).clamp(max=len(terrain_names) - 1)

    target_idx = terrain_names.index(target_name)
    return sub_idx == target_idx
```

示例：只在上楼梯地形启用某个 reward：

```python
def _reward_stairs_only_clearance(self):
    reward = self._some_clearance_reward()
    mask = self._standard_terrain_mask("pyramid_stairs")
    return reward * mask.float()
```

示例：只在迷宫地形启用避障 reward：

```python
def _reward_maze_only_obstacle_evasion(self):
    reward = self._some_obstacle_reward()
    mask = self._standard_terrain_mask("maze")
    return reward * mask.float()
```

## Track 模式语义

Track 地形不是 Standard 的按比例 grid 语义。

容器 `track_generator.py` 中明确说明：

```text
Track 模式下同一 col 的不同 row 本来就是不同的子地形
col 决定难度、row 决定类型
```

Track generator 的设计：

- `track_length -> num_rows`，沿 X 方向的子地形序列。
- `num_parallel_tracks -> num_cols`，沿 Y 方向的并行赛道/难度档。
- col 固定映射 difficulty。
- row 固定映射 track sequence 中的子地形类型。

因此 Track 模式下：

```python
terrain.terrain_types
```

表示并行赛道列/难度档，不表示机器人当前所在的子地形类型。

如果在 Track reward 中直接使用 `terrain_types` 做 `maze/stairs/slope` 判断，会得到错误语义。

## Track 当前子地形段 mask 推荐实现

Track 下应根据机器人世界坐标 `x` 推断当前所在 segment。

容器 `TrackTerrainGenerator` 构建边界时使用：

```python
size_x = terrain_size[0]
offset = -size_x * track_length * 0.5
boundaries = [offset + i * size_x for i in range(track_length + 1)]
```

所以 reward 中可以按同样公式计算 segment：

```python
def _track_segment_index(self):
    terrain = self.env.scene.terrain
    terrain_gen_cfg = terrain.cfg.terrain_generator

    size_x = float(terrain_gen_cfg.size[0])
    track_length = int(terrain_gen_cfg.track_length)
    offset = -size_x * track_length * 0.5

    robot = self.env.scene["robot"]
    x = robot.data.root_pos_w[:, 0]

    seg = torch.floor((x - offset) / size_x).long()
    return seg.clamp(0, track_length - 1)
```

再把 segment index 映射到 track sequence：

```python
def _track_segment_mask(self, target_name: str):
    terrain = self.env.scene.terrain
    terrain_gen_cfg = terrain.cfg.terrain_generator

    order = getattr(terrain_gen_cfg, "sub_terrains_order", None)
    if order is None:
        order = list(terrain_gen_cfg.sub_terrains.keys())

    track_length = int(terrain_gen_cfg.track_length)
    sequence = []
    while len(sequence) < track_length:
        sequence.extend(order)
    sequence = sequence[:track_length]

    seg = self._track_segment_index()
    mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
    for idx, name in enumerate(sequence):
        if name == target_name:
            mask |= seg == idx
    return mask
```

示例：只在 Track 终点迷宫段启用导航/避障 reward：

```python
def _reward_track_maze_only_navigation(self):
    reward = self._some_navigation_reward()
    mask = self._track_segment_mask("open_entry_maze")
    return reward * mask.float()
```

示例：只在 Track 楼梯段启用额外 foot clearance：

```python
def _reward_track_stairs_only_clearance(self):
    reward = self._some_clearance_reward()
    mask = (
        self._track_segment_mask("pyramid_stairs")
        | self._track_segment_mask("pyramid_stairs_inv")
    )
    return reward * mask.float()
```

## 推荐封装：统一地形 mask

可以封装一个按模式分派的 helper：

```python
def _terrain_mask(self, target_name: str):
    terrain = self.env.scene.terrain
    terrain_gen_cfg = terrain.cfg.terrain_generator

    if terrain_gen_cfg is None:
        return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    if hasattr(terrain_gen_cfg, "track_length"):
        return self._track_segment_mask(target_name)
    return self._standard_terrain_mask(target_name)
```

然后 reward 函数不必关心当前是 Standard 还是 Track：

```python
def _reward_maze_only(self):
    reward = ...
    mask = self._terrain_mask("maze") | self._terrain_mask("open_entry_maze")
    return reward * mask.float()
```

## 标签 mask 与几何 mask 的取舍

按地形标签做 reward 的优点：

- 目标明确。
- 对 Standard 的 `maze/stairs/slope` 专项训练很直接。
- 可以避免在非目标地形上产生多余梯度。

缺点：

- Track 下必须正确计算 segment，不能直接用 `terrain_types`。
- 边界附近可能出现机器人跨段、身体/脚位分布跨越两个子地形的问题。
- 评估环境若内部地形布局变化，标签逻辑可能需要同步更新。

几何 mask 的优点：

- 直接根据 `height_scan` / `nav_scanner` 判断墙、台阶、坡等局部结构。
- 对边界和未知布局更鲁棒。
- 当前 `agent_ppo` 已经有用 height scan 区分 maze wall 和 stair edge 的实现基础。

缺点：

- 阈值需要调。
- 容易把局部结构误判为另一个地形，需要和实际观测分布一起验证。

当前建议：

- Standard 专项训练可以使用 `terrain_types` 的标签 mask。
- Track 训练如果只是“某段启用某 reward”，使用 robot X segment mask。
- Track 迷宫避障/转向更推荐结合 `nav_scanner` 几何特征；即使使用 segment mask，
  也可以再叠加 “前方确实有墙/堵路” 的几何门控。

## 结论

训练时可以实现只针对某个地形的 reward，但实现方式取决于模式：

```text
Standard:
  terrain_types -> col -> proportion mapping -> terrain name

Track:
  robot x position -> track segment index -> sub_terrains_order -> segment terrain name
```

不要在 Track 模式里直接把 `terrain_types` 当作当前地形类型。
