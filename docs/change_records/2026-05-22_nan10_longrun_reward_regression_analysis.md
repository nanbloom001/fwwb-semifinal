# 2026-05-22 Nan10 长训退化与奖励函数归因分析

## 结论摘要

本次分析对象是长训人工监控记录：

```text
arena_frontend_monitor_runtime/manual_metric_recorder/sessions/20260522-101241
```

该轮从 `train_global_step ~= 11652` 跑到 `14486`，覆盖约 6 小时。
综合平滑后的曲线、周期聚合后的曲线、奖励占比和源码语义，结论是：

1. 训练不是“完全无效”，而是优化方向偏离了目标。
2. 模型后期更稳定、更少非超时失败，但高难倒台阶通过能力明显退化。
3. 退化不是平台锯齿造成的假象，按 4 分钟周期聚合后仍然成立。
4. 当前奖励的主导项是速度跟踪、yaw 跟踪、方向推进和姿态/关节稳定；台阶专项动作链奖励占比太低。
5. 现有奖励更容易学到“保守、少摔、拖到 timeout”，而不是“识别台阶 -> 抬脚 -> 落准 -> 承重 -> 身体跟上 -> 通过高难台阶”。

因此下一轮不应单纯延长训练时间。需要重新分配奖励预算，降低一部分通用速度/稳定项的支配地位，增强高难台阶动作链的直接信号。

## 数据处理说明

平台原始曲线存在明显周期锯齿。原始 `completed_count` 开头为：

```text
3, 33, 36, 3109,
82, 110, 107, 2871,
144, 179, 156, 2718,
...
```

`timeout_count` 也同相：

```text
0, 0, 0, 665,
28, 15, 25, 680,
39, 42, 27, 604,
...
```

每个点间隔 60 秒，周期相关性检测显示：

| 指标 | lag=4 相关性 |
|---|---:|
| `completed_count` | `0.934` |
| `timeout_count` | `0.879` |
| `completed_count_pyramid_stairs_inv_l9` | `0.902` |

因此锯齿周期更准确地说是 `4` 个采样点，即约 `240s`。
从低点到高点看起来像 3 分钟，但两个高峰之间约 4 分钟。

已在 `arena_frontend_monitor/postprocess_monitor_capture.py` 中增加：

```text
all_metric_series_cycle_smoothed.json
cycle_smoothing_diagnostics.json
all_metric_series_cycle_blocks.json
cycle_block_diagnostics.json
```

本报告主要使用 `all_metric_series_cycle_blocks.json` 做趋势判断。这个文件按检测到的完整周期聚合，避免把同一个统计周期拆成“低、低、低、高”的假趋势。

## 平滑后关键趋势

周期聚合后，整体指标如下：

| 指标 | 首段均值/首值 | 末段均值/末值 | 解释 |
|---|---:|---:|---|
| `completed_count` | first block `795.25` | last block `625.0` | 总完成数后段下降 |
| `timeout_count` | first block `166.25` | last block `287.5` | timeout 明显上升 |
| `ntfail_total` | first block `84.5` | last block `27.0` | 非超时失败下降 |
| `total_score` | first block `47.71` | last block `59.99` | 总分不代表高难倒台阶能力 |
| `step_avg` | first block `802.0` | last block `1466.0` | 后期大量 episode 接近最大步数 |

这说明后期策略确实更稳、更少摔，但也更容易拖到 timeout。

高难倒台阶变化更关键：

| 指标 | 周期聚合 first block | 周期聚合 last block | 变化 |
|---|---:|---:|---|
| `completed_count_pyramid_stairs_inv_l7` | `29.75` | `1.5` | 大幅下降 |
| `completed_count_pyramid_stairs_inv_l8` | `22.5` | `0.0` | 基本消失 |
| `completed_count_pyramid_stairs_inv_l9` | `13.5` | `0.0` | 基本消失 |
| `completed_count_pyramid_stairs_l9` | `33.5` | `35.5` | 普通台阶没有同等退化 |

普通台阶 L9 仍能维持，但倒台阶 L8/L9 消失。这说明退化有地形方向性，不是所有 locomotion 能力一起崩。

## 分阶段日志表现

按 `train_global_step` 分段：

| 阶段 | step 范围 | alpha 均值 | completed 均值 | timeout 均值 | ntfail 均值 | stairs_inv L7/L8/L9 completed 均值 |
|---|---:|---:|---:|---:|---:|---:|
| early_skill_alive | `11650-12050` | `0.265` | `744` | `207` | `128` | `22.9 / 8.6 / 2.7` |
| transition_mid | `12050-12550` | `0.856` | `749` | `260` | `96` | `15.2 / 1.7 / 0.05` |
| full_late | `12550-14500` | `1.0` | `744` | `273` | `71` | `8.0 / 0.86 / 0.04` |
| tail | `14050-14500` | `1.0` | `739` | `286` | `43` | `5.9 / 0.57 / 0.0` |

核心转折发生在 `alpha` 从低值过渡到中高值时。
`stairs_inv L8/L9` 不是慢慢波动，而是在过渡阶段快速消失。

## 奖励占比

按整轮平均绝对贡献排序，排除 `reward_mean/reward_std` 后：

| 奖励 | 均值贡献 | 绝对占比 |
|---|---:|---:|
| `reward_track_lin_vel_xy` | `+1.482` | `36.56%` |
| `reward_track_ang_vel_z` | `+0.713` | `17.59%` |
| `reward_command_direction_progress` | `+0.478` | `11.80%` |
| `reward_ang_vel_xy` | `-0.326` | `8.03%` |
| `reward_joint_acc` | `-0.236` | `5.82%` |
| `reward_joint_position_penalty` | `-0.210` | `5.17%` |
| `reward_action_rate` | `-0.098` | `2.42%` |
| `reward_command_direction_deviation` | `-0.085` | `2.10%` |
| `reward_feet_clearance` | `+0.078` | `1.92%` |
| `reward_command_path_progress` | `+0.055` | `1.36%` |
| `reward_height_scan_feet_clearance` | `+0.033` | `0.80%` |
| `reward_stair_base_clearance_penalty` | `-0.016` | `0.39%` |
| `reward_down_stair_speed_safety` | `-0.013` | `0.33%` |
| `reward_stair_forward_foot_placement` | `+0.007` | `0.18%` |
| `reward_stair_swing_step_targeting` | `+0.0028` | `0.07%` |
| `reward_stair_over_clearance_penalty` | `-0.0028` | `0.07%` |
| `reward_stair_stride_length_penalty` | `-0.0013` | `0.03%` |
| `reward_down_stair_touchdown_safety` | `-0.0009` | `0.02%` |
| `reward_stair_support_continuity_penalty` | `-0.0007` | `0.02%` |
| `reward_commanded_stall_penalty` | `~0` | `~0%` |

前 8 个奖励已经占据约 88% 的绝对贡献。台阶动作链奖励虽然都注册且有值，但梯度预算太小。

## 高占比奖励函数语义分析

### 1. `track_lin_vel_xy`

源码位置：`agent_ppo/feature/reward_process.py::_reward_track_lin_vel_xy`

语义：

```python
lin_vel_error = ||command_xy - root_lin_vel_b_xy||^2
reward = exp(-lin_vel_error / std^2)
```

问题：

- 这是身体坐标系速度跟踪奖励，不知道“是否正在上台阶”。
- 高难台阶需要短暂停顿、抬脚、调整支撑、身体越过台阶。
- 这些动作会牺牲瞬时速度跟踪，因此容易被该大权重压制。
- 如果模型选择保守低速或绕角度，仍可在部分时刻获得可观速度跟踪收益。

该项占比 `36.56%`，是绝对主导项。它本身不是错误，但它没有表达“完成高难台阶动作链”。

### 2. `track_ang_vel_z`

源码位置：`agent_ppo/feature/reward_process.py::_reward_track_ang_vel_z`

语义：

```python
ang_vel_error = (command_wz - root_ang_vel_b_z)^2
reward = exp(-ang_vel_error / std^2)
```

问题：

- 当前 command mix 中 full command 比例较高，训练中存在 yaw 命令。
- 评估时通常只给 `vx`，但训练阶段仍会强化“能转”的能力。
- 如果无指令转向惩罚/世界方向保持不够强，策略可能把转向当成通过台阶的常用动作。

该项占比 `17.59%`，远高于所有防转向奖励。

### 3. `command_direction_progress`

源码位置：`agent_ppo/feature/reward_process.py::_reward_command_direction_progress`

语义：

```python
projected_speed = dot(root_lin_vel_b_xy, command_dir_body)
progress = clamp(projected_speed / target_speed, 0, 1)
```

问题：

- 它是强 dense reward，鼓励沿当前命令方向有速度。
- 但它仍然不关心脚是否落到下一阶、不关心身体是否越过台阶。
- 如果身体方向逐渐旋转，身体坐标系下的命令方向也会随之改变，不能完全阻止世界方向上的偏航路径依赖。

该项占比 `11.80%`，明显高于台阶专项动作链奖励。

### 4. `ang_vel_xy`

该项来自基础 locomotion 稳定惩罚，监控贡献约 `-0.326`，占比 `8.03%`。

问题：

- 高难台阶需要短时 pitch/roll 调整。
- 过强会让模型更倾向于保持机身稳定，而不是积极跨越高阶。
- 日志上后期 `pose_score` 提升，同时高难倒台阶消失，符合该方向。

这不代表应该移除它，但它当前与“高难台阶探索”存在张力。

### 5. `joint_acc`、`joint_position_penalty`、`action_rate`

合计约 `13.4%`。

问题：

- 这些项共同压制快速摆腿、关节偏离默认姿态、动作变化。
- 对普通平地/低难地形是好事。
- 对高难台阶，尤其倒台阶，需要更激进的摆腿、落脚和承重调整。

后期 `ntfail_total` 降低说明它们确实提高稳定性；但 `timeout_count` 上升和 `stairs_inv L8/L9` 消失说明它们也促进了保守策略。

### 6. `command_direction_deviation`

源码位置：`agent_ppo/feature/reward_process.py::_reward_command_direction_deviation`

语义：

```python
angle = angle(actual_xy_velocity, command_xy)
penalty = ((angle - angle_limit) / max_angle)^2
```

配置：

```toml
angle_limit_deg = 0.0
max_angle_deg = 50.0
weight = -0.52 -> -0.62
```

问题：

- 语义正确，应该防止侧向/斜向实际速度。
- 但实际贡献只有约 `-0.085`，占比 `2.10%`。
- 它惩罚的是“实际速度方向 vs 命令方向”，不是“机器人朝向/路径相对 reset 世界方向”。
- 如果狗先转身再沿身体前方走，速度方向可能仍然看似合理。

因此它不足以阻止“上楼梯前慢慢转成斜向”的策略。

### 7. `command_path_progress`

源码位置：`agent_ppo/feature/reward_process.py::_reward_command_path_progress`

语义：

- 对无 yaw 命令，使用 reset/命令段 anchor 方向。
- 对有 yaw 命令，使用当前命令方向。
- 奖励每步沿 desired direction 的世界坐标位移。
- 有 `segment_progress_cap`，不是无限推进奖励。

问题：

- 语义上很有价值，因为它比身体坐标速度更接近“沿 reset/命令路径推进”。
- 但实际贡献只有约 `+0.055`，占比 `1.36%`。
- 它太弱，无法对抗速度跟踪、yaw 跟踪和姿态稳定项。

该项应是后续修正路径依赖的重要候选，但需要防止变成下台阶硬冲。

## 台阶专项奖励分析

### `feet_clearance`

源码位置：`agent_ppo/feature/reward_process.py::_reward_feet_clearance`

该奖励是 dense swing-foot clearance，贡献约 `+0.078`，占比 `1.92%`。

问题：

- 它能让模型抬腿，但不保证“落到下一阶”。
- 可能造成“会抬腿，但抬高后踩空/卡住/身体跟不上”。
- 日志和视频观察中的“腿抬起来了，但动作链不完整”与此一致。

### `height_scan_feet_clearance`

贡献约 `+0.033`，占比 `0.80%`。
门控有值，不是空项。它能针对台阶区域加强抬腿，但仍主要解决垂直 clearance，不解决承重和身体跟随。

### `stair_forward_foot_placement`

源码位置：`agent_ppo/feature/reward_process.py::_reward_stair_forward_foot_placement`

语义：

- 只在 first contact 时奖励。
- 要求落脚在命令方向前方、接近上台阶表面。

问题：

- 这是很关键的奖励，但触发稀疏。
- 实际贡献只有 `+0.007`，占比 `0.18%`。
- 量级太小，不足以指导高难台阶落脚。

### `stair_swing_step_targeting`

语义：

- swing phase 中奖励短步前伸到下一阶附近。
- 比 first-contact 更 dense，理论上很有价值。

问题：

- 实际贡献只有 `+0.0028`，占比 `0.07%`。
- 这说明它几乎没有参与主导策略。

### `stair_stride_length_penalty`

语义：

- 惩罚前脚伸太远、后脚落后、前后跨度过大。

问题：

- 这是针对“前脚跨两个台阶，后脚还在地面”的关键项。
- 实际贡献只有 `-0.0013`，占比 `0.03%`。
- 量级远低于 `joint_acc`、`joint_position_penalty` 等通用项。

### `stair_support_continuity_penalty`

语义：

- 惩罚台阶区域接触脚数不足、只有前脚/后脚支撑。

问题：

- 这应该帮助“脚上去了之后身体跟上”和防止单腿/两腿不稳定支撑。
- 实际贡献只有 `-0.0007`，几乎可以忽略。

### `stair_base_clearance_penalty`

语义：

- 惩罚腹部/躯干下沿距离台阶高侧表面太近。
- 现在上下台阶都激活。

贡献约 `-0.016`，比其他台阶动作链项稍大，但仍只有 `0.39%`。
它能防止趴低蹭台阶，但不是完成动作链的正向奖励。

## 为什么奖励会导致上不了高难台阶

当前奖励实际优化目标更接近：

```text
速度跟踪 + yaw 跟踪 + 有方向速度 + 少晃 + 少关节激烈动作 + 有一点抬腿
```

而不是：

```text
识别台阶
-> 抬脚到合适高度
-> 短步前伸
-> 落到下一阶
-> 稳定承重
-> 后脚跟上
-> 身体越过台阶
-> 沿 reset/命令世界方向通过
```

高难倒台阶需要完整动作链。当前每个动作链环节的奖励都存在，但量级太低；强奖励项反而鼓励了更容易达成的局部目标：

- 在简单地形稳定速度跟踪；
- 减少 pitch/roll；
- 降低关节加速度；
- 避免大动作；
- 避免摔倒；
- 必要时慢走或拖到 timeout。

这解释了日志现象：

- `pose_score` 后期提升；
- `energy_score` 后期不差；
- `ntfail_total` 下降；
- `timeout_count` 上升；
- 普通台阶 L9 仍有能力；
- 倒台阶 L8/L9 消失。

## 当前最可能的失败机制

### 机制 1：高难倒台阶的冒险动作被稳定项压制

高难倒台阶需要主动进入不稳定瞬间。
但 `ang_vel_xy + joint_acc + joint_position_penalty + action_rate` 的合计贡献远大于 `stair_swing/stride/support/place`。
模型发现“不冒险”更容易避免大惩罚。

### 机制 2：抬腿奖励强于落脚/承重奖励

`feet_clearance + height_scan_feet_clearance` 约 `0.11`，而：

```text
stair_forward_foot_placement ~= 0.007
stair_swing_step_targeting ~= 0.003
stair_support_continuity_penalty ~= -0.001
```

这会产生“能抬腿，但不能完成上阶”的偏差。

### 机制 3：路径保持约束不足

`command_direction_deviation` 和 `command_path_progress` 方向正确，但贡献不足。
如果狗通过慢慢 yaw 来改变身体坐标系，`track_lin_vel_xy` 和 `command_direction_progress` 仍可能给出可观奖励。

### 机制 4：反 timeout 信号几乎无效

`reward_commanded_stall_penalty` 全轮均值约 `-0.000002`，基本没有训练作用。
因此当模型发现高难倒台阶风险大时，拖到 timeout 的代价不够直接。

### 机制 5：下台阶安全奖励改善安全，但可能强化保守

`down_stair_speed_safety` 和 `down_stair_touchdown_safety` 的贡献并不大，但方向上会鼓励慢下。
如果没有足够强的“完成/通过”信号配平，它们会支持保守策略。

## 下一轮修改方向

建议不要继续从后期模型训，也不要继续只延长训练时间。
应该从 `11700` 或 `11750` 附近继续，并调整奖励预算。

### 建议 1：降低强主导速度项的支配性

不是完全降低速度跟踪，而是避免其压制台阶动作链：

- `track_lin_vel_xy`: 小幅降或保持但增大 std，降低瞬时速度误差惩罚。
- `command_direction_progress`: 降低 dense 推进奖励，避免高难倒台阶硬要求持续速度。
- `track_ang_vel_z`: 若评估以 vx-only 为主，降低 full/yaw 训练占比或降低 yaw reward 权重。

### 建议 2：显著增强落脚/承重/身体跟上

重点不是再加强纯抬腿，而是加强：

- `stair_forward_foot_placement`
- `stair_swing_step_targeting`
- `stair_stride_length_penalty`
- `stair_support_continuity_penalty`
- `stair_base_clearance_penalty`

其中 `stair_stride_length_penalty` 和 `stair_support_continuity_penalty` 当前几乎无效，应至少提高到能在监控中达到 `0.01` 量级，否则难以改变步态。

### 建议 3：把 `commanded_stall_penalty` 修成真正有效

当前该项均值几乎为 0。需要检查其激活条件是否过苛刻：

- `stall_time_s = 0.80`
- `min_projected_speed = 0.10`
- `progress_ratio = 0.25`
- down-step 和 contact grace 会关闭部分惩罚

目标不是惩罚所有慢动作，而是惩罚长期无进展。
如果它不能产生 `0.005~0.02` 量级，就无法对 timeout 形成训练压力。

### 建议 4：路径保持奖励应更靠近世界/segment anchor

`command_path_progress` 语义较好，但量级偏小。
应考虑让它成为 vx-only 评估模式下的核心方向约束，而不是附属项。

同时要保留 cap/grace，避免下台阶硬冲。

### 建议 5：学习率要保守

这轮退化发生在 `alpha` 从低到中高的阶段，同时学习率已到 `2e-4`。
继续训练已有能力时，建议使用更保守学习率，如 `8e-5 ~ 1e-4`，避免把已有倒台阶技能快速覆盖。

## 本轮模型选择建议

周期聚合后，最有潜力的模型仍是早期：

| checkpoint 附近 | global pass rate | completed | timeout | ntfail | stairs_inv L7/L8/L9 | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| `11700` | `0.743` | `792.5` | `187.0` | `87.5` | `37.2 / 30.0 / 7.5` | 最适合恢复高难倒台阶 |
| `11750` | `0.755` | `799.2` | `178.0` | `82.0` | `40.2 / 22.0 / 7.2` | 综合稳定性略好 |
| `12150` | `0.712` | `783.2` | `231.0` | `85.5` | `34.2 / 4.8 / 0.0` | 普通台阶更好，但高难倒台阶已明显丢失 |

若目标是恢复高难倒台阶，应从 `11700` 或 `11750` 继续。
不建议从 `12500+` 继续追高难倒台阶，因为 `stairs_inv L8/L9` 基本已归零。
