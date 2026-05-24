# Terrain Level / Label Repository Search

Date: 2026-05-24

> Runtime correction, 2026-05-25:
> This static search found strong evidence that the lower-level container runtime
> and scorer can access `terrain_levels` / `terrain_types`.  A later runtime
> probe showed that this does **not** make the labels reliably accessible from
> `agent_ppo` alone, because aisrv/workflow only sees a Kaiwu worker wrapper and
> cannot unwrap the Isaac env.  Treat
> `docs/terrain_label_runtime_final_verdict_2026-05-25.md` as the operational
> verdict for future training changes.

This report is based on a fresh repository search. It does not use older analysis
reports as evidence. Prior reports under `docs/` or generated comparison folders
were intentionally excluded from the conclusions below.

## Scope

Searched areas:

- `agent_ppo/`
- `agent_diy/`
- `external/container_src/`
- `legged-robot/`
- `train_test.py`

Missing local areas:

- `common_python/`

`tools/` is not present as a top-level runtime package in this checkout, but
`external/container_src/current/tools/base_env/base_env.py` is available as a
container-source snapshot. The strongest platform-side evidence below comes from
`external/container_src/current/isaac_env/base_env.py`, with matching logic also
visible under `external/container_src/current/tools/base_env/base_env.py`.

## Search Commands

```bash
rg -n "scene\.terrain|terrain_levels|terrain_types|terrain_origins|terrain_generator|sub_terrains|track_length|num_rows|num_cols" agent_ppo agent_diy legged-robot train_test.py -S
rg -n "env\.reset|env\.step|usr_conf|is_eval|RUN_MODE_EVAL|RUN_MODE_EXAM|eval_env_conf|check_usr_conf|scene|terrain" agent_ppo agent_diy legged-robot train_test.py -S
rg -n "completed_count_|abnormal_count_|timeout_count_|total_score_|energy_score_|pose_score_|_l\{0~9\}|l[0-9]" agent_ppo agent_diy legged-robot -S
rg -n "terrain_levels|terrain_types|terrain_origins|scene\.terrain|terrain_generator|sub_terrains|num_rows|num_cols|track_length|difficulty|level" external/container_src -S
```

## Confirmed Findings

### 1. Active PPO training can access the unwrapped Isaac environment

The active training path is `agent_ppo`.

`agent_ppo/conf/conf.py` sets the active config to `Nan10StairBridgeConfig`.
During non-eval training, `Config.load_conf()` loads:

```text
agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml
```

During eval/exam mode, it attempts to load:

```text
tools/eval/conf/eval_env_conf.toml
```

The active training workflow loads the config, calls `env.reset(usr_conf)`, then
uses the returned `obs` and `critic_obs` in the PPO loop. The workflow also has
helpers that unwrap the runtime environment and look for objects with
`command_manager` and `scene`, which means training-side code is already written
with access to `env.scene`.

Relevant files:

- `agent_ppo/conf/conf.py`
- `agent_ppo/workflow/train_workflow.py`

### 2. Standard-mode config explicitly defines difficulty rows

In `agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml`, the terrain
comments and settings define:

```toml
mode = "standard"
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 9
```

The same file documents that, in Standard mode:

- `num_rows` means difficulty tiers.
- `num_cols` means parallel terrain plots, not difficulty.

So for the current active Standard training config, difficulty is intended to be
represented by row/level, with levels 0 through 9.

Relevant file:

- `agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml`

### 3. Track-mode config uses columns as difficulty lanes

The checked Track config under `agent_diy/conf/train_env_conf_track_nav.toml`
documents a different convention:

```toml
mode = "track"
num_rows = 10
num_cols = 10
curriculum = true
max_init_terrain_level = 0
track_length = 5
```

Its comments state:

- In Track mode, `num_rows` is overridden by `track_length`.
- `num_cols` is the difficulty tier / parallel lane count.
- `l0` is easiest and `l9` is hardest.

The developer guide also describes eval placement:

- Standard eval maps envs over selected `level` values and subterrain types.
- Track eval places robots on tracks whose columns match assigned difficulty
  levels.

Relevant files:

- `agent_diy/conf/train_env_conf_track_nav.toml`
- `legged-robot/开发指南/03-智能体详述.md`

### 4. Current reward code already probes terrain labels

`agent_ppo/feature/reward_process.py` contains `_standard_terrain_mask()`. This
function reads:

```python
terrain = self.env.scene.terrain
terrain_types = getattr(terrain, "terrain_types", None)
terrain_gen_cfg = terrain.cfg.terrain_generator
sub_terrains = terrain_gen_cfg.sub_terrains
```

It then maps terrain type indices back to subterrain names by using the terrain
generator proportions. This is used for Standard terrain-specific reward masks.

The same code path also reads:

```python
terrain_levels = getattr(terrain, "terrain_levels", None)
```

and logs min/max/mean if that field exists.

This is the strongest source-level evidence in the repository: the active PPO
reward code already expects that `env.scene.terrain.terrain_levels` may exist at
runtime.

Important limitation: `_standard_terrain_mask()` explicitly rejects terrain
generators with `track_length`, so the current terrain-type mask helper is for
Standard mode only. Track mode needs its own label-reading helper or a generic
helper that understands both conventions.

Relevant file:

- `agent_ppo/feature/reward_process.py`

### 5. Monitoring already assumes per-level metrics exist

`agent_ppo/conf/monitor_builder.py` creates monitor expressions such as:

```text
abnormal_count_pyramid_stairs_l0 - timeout_count_pyramid_stairs_l0
...
abnormal_count_pyramid_stairs_l9 - timeout_count_pyramid_stairs_l9
```

for Standard subterrain names. The developer guide documents Track metrics such
as:

```text
completed_count_track_l{0~9}
abnormal_count_track_l{0~9}
timeout_count_track_l{0~9}
total_score_track_l{0~9}
energy_score_track_l{0~9}
pose_score_track_l{0~9}
time_score_track_l{0~9}
```

This confirms that the platform monitoring layer distinguishes difficulty
levels. It does not by itself prove the Python training environment exposes the
same labels as tensors, but it strongly supports the interpretation that level
information exists in the runtime.

Relevant files:

- `agent_ppo/conf/monitor_builder.py`
- `legged-robot/开发指南/02-环境详述.md`

### 6. Container-side base environment confirms runtime terrain labels

`external/container_src/current/isaac_env/base_env.py` directly confirms that
the runtime environment exposes terrain labels.

Confirmed fields:

```python
terrain = env_unwrapped.scene.terrain
terrain.terrain_levels
terrain.terrain_types
```

Key evidence:

- The wrapper snapshots `terrain.terrain_levels` before `env.step()` because
  Isaac Lab may auto-reset envs and then update terrain levels during the step.
- Every 500 steps it prints `terrain_level` mean/max/min/distribution from
  `terrain.terrain_levels`.
- In Track mode it also prints `terrain_type(col/difficulty)` distribution from
  `terrain.terrain_types`.
- Video naming resolves per-env terrain names using both `terrain_types` and
  `terrain_levels`.

Relevant file:

- `external/container_src/current/isaac_env/base_env.py`

### 7. Container-side eval logic defines deterministic level placement

For eval configs with `[terrain] level = [...]`,
`external/container_src/current/isaac_env/base_env.py` stores:

```python
env_cfg._eval_level_list = list(level_list_int)
```

Then:

- Standard eval:
  - `num_rows = max(level) + 1`
  - `num_cols = len(sub_terrains)`
  - subterrain columns are rebuilt in the explicit eval order
  - reset uses `(row=level, col=subterrain)`
- Track eval:
  - `num_parallel_tracks` / `num_cols` is expanded to cover `max(level)+1`
  - reset uses `col=level`
  - Track labels are named like `L{level}_track_{sub_terrains_chain}`

The file also explicitly keeps `terrain_generator.curriculum=True` during eval
level-list placement so Isaac Lab generates a deterministic grid, while disabling
the training-time terrain-level curriculum term to prevent promotion/demotion
from overriding deterministic reset placement.

Relevant file:

- `external/container_src/current/isaac_env/base_env.py`

## Answers

### Can training read terrain difficulty labels?

Yes, based on the container-side runtime source snapshot.

What is confirmed from source:

- Active PPO code can access `env.scene`.
- Reward code already attempts to read `env.scene.terrain.terrain_levels`.
- Reward code already reads `env.scene.terrain.terrain_types`.
- Config and docs define terrain difficulty levels.
- Container-side `base_env.py` directly reads `terrain.terrain_levels` before
  step, during monitoring, and during debug printing.
- Container-side `base_env.py` directly reads `terrain.terrain_types`.

What still deserves a runtime sanity check:

- Exact dtype/device/shape in the current launched platform version.
- Whether the snapshot under `external/container_src/current/` is perfectly
  identical to the active remote container version for a given run.

### Can every env read its own level independently?

Yes in the container-side implementation.

The container wrapper indexes and snapshots `terrain.terrain_levels` as a
per-env tensor, passes `pre_step_terrain_levels` into monitor code, and builds
per-env terrain names. The debug distribution also uses `torch.bincount` over
`terrain.terrain_levels`, which confirms this is a vector of environment labels.

The useful sanity check is still to print:

```python
terrain_levels.shape
terrain_levels.dtype
terrain_levels.device
terrain_levels[:16]
```

after reset and after at least one step.

### Can eval read terrain difficulty labels?

Yes, container-side eval code is explicitly level-aware.

`Config.load_conf()` switches eval/exam runs to
`tools/eval/conf/eval_env_conf.toml`, and the container-side base environment has
specific code for `[terrain] level = [...]`. It stores the level list on
`env_cfg`, rewrites Standard eval grid dimensions/subterrain ordering, expands
Track eval lanes if necessary, and hooks `reset_root_state_eval_level_aware`.

The remaining caveat is version parity: verify that the active online container
matches `external/container_src/current/`.

### Can terrain labels be used to test gate accuracy?

Yes.

Recommended use:

- Keep labels out of actor observations at first.
- Use labels as diagnostics only.
- Compare geometry-based gate predictions against runtime labels.
- Report confusion by terrain type and difficulty level.

This avoids changing the policy input contract while validating whether the
gate is actually detecting terrain difficulty correctly.

### Can terrain labels be used for difficulty-specific tuning?

Yes, with caution.

Safe first uses:

- Per-level reward diagnostics.
- Per-level success/failure summaries.
- Per-level curriculum or reward-weight scheduling outside the actor input.
- Evaluation breakdown by difficulty.

Riskier uses:

- Feeding labels directly into policy observations.
- Strongly changing rewards by level before verifying label correctness.
- Making Track and Standard share one label interpretation without handling
  their different row/column conventions.

## Standard vs Track Label Semantics

The repository suggests different terrain-layout semantics:

| Mode | Difficulty axis | Evidence |
| --- | --- | --- |
| Standard | `num_rows` / terrain row | Active Standard TOML comments |
| Track | `num_cols` / track column | Track TOML comments and eval guide |

This means a generic helper should not blindly interpret rows and columns the
same way in both modes. It should read exposed runtime fields first, then fall
back to mode-specific inference only if necessary.

## Recommended Runtime Probe

Add a temporary or gated debug probe that runs after `env.reset(usr_conf)` and
again after the first `env.step(actions)`.

Probe fields:

```python
terrain = getattr(getattr(env, "scene", None), "terrain", None)
terrain_levels = getattr(terrain, "terrain_levels", None)
terrain_types = getattr(terrain, "terrain_types", None)
terrain_origins = getattr(terrain, "terrain_origins", None)
terrain_cfg = getattr(terrain, "cfg", None)
terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
track_length = getattr(terrain_generator, "track_length", None)
num_rows = getattr(terrain_generator, "num_rows", None)
num_cols = getattr(terrain_generator, "num_cols", None)
sub_terrains = getattr(terrain_generator, "sub_terrains", None)
```

Log for tensor fields:

```text
exists
shape
dtype
device
min/max/mean
first 16 values
```

Run the probe in four cases:

1. Standard train
2. Track train
3. Standard eval
4. Track eval

## Proposed Implementation Path

1. Add a small helper in `agent_ppo/feature/reward_process.py` or a shared
   utility module:

   ```python
   def get_terrain_label_context(env):
       ...
   ```

   It should return a structured object or dict with optional fields:

   ```text
   terrain_levels
   terrain_types
   terrain_names
   mode
   num_rows
   num_cols
   track_length
   available
   reason
   ```

2. Use the helper for diagnostics first:

   - log availability once;
   - log shape and first values;
   - do not change policy observations;
   - do not change reward behavior by default.

3. Add gate-validation diagnostics:

   - geometry gate result from height scanner / nav scanner;
   - runtime label result from terrain labels;
   - mismatch rate by terrain type and level.

4. Add difficulty-aware tuning only after label validation:

   - per-level reward scalars;
   - per-level curriculum thresholds;
   - per-level eval summaries.

5. Keep Standard and Track conventions separate:

   - Standard: row/level based difficulty;
   - Track: column/lane based difficulty;
   - prefer runtime `terrain_levels` over inferred row/column math.

## Current Verdict

The repository plus `external/container_src` provide enough evidence to say that
terrain labels are available to the runtime wrapper:

- `terrain.terrain_levels` is a per-env difficulty-level tensor.
- `terrain.terrain_types` is a per-env terrain column/type tensor.
- Standard difficulty is row/level based.
- Track difficulty is column/lane based.
- Eval level-list mode is deterministic and explicitly handled.

The practical next step is not to prove existence from scratch, but to add a
small version/parity probe in the active training run so logs show the exact
shape, dtype, device, distribution, and first values for the current container.
