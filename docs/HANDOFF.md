# Tencent Kaiwu RL Dog Handoff

This document lists the assets that should be handed to the next maintainer for
continuing training from the current repository state.

## Current Active Training Entry

- Main implementation: `agent_ppo`
- Active stage: `Nan10StairBridgeConfig`
- Active config selector: `agent_ppo/conf/conf.py`
- Active training TOML:
  `agent_ppo/conf/train_env_conf_standard_nan10_stair_bridge.toml`
- Current task mode: Standard locomotion/stair bridge training
- Current continuation target: continue from nan10-derived stair checkpoints,
  with stairs and inverse stairs as the main acquisition target.

Before handoff, keep the current uncommitted changes or commit them. The latest
reward, command-mix, monitoring, and postprocess logic is not represented by the
old clean git state.

## Required Assets

Hand these off for a continuation-training handoff:

- `agent_ppo/`: PPO agent, active config, reward functions, command mix,
  monitor builder, model, and training workflow.
- `conf/`, `kaiwu.json`, `train_test.py`: platform and local quick-run
  entrypoints.
- `AGENTS.md`, `README.md`, `CHANGES.md`, `docs/`, `legged-robot/`: operating
  notes, competition docs, and analysis history.
- `agent_diy/codex_rpc_bridge/`: RPC bridge, safe sync script, keepalive helper,
  and platform operating notes.
- `arena_frontend_monitor/`: manual monitor recorder and postprocess tools.
- Selected monitor summaries from:
  - `arena_frontend_monitor_runtime/manual_metric_recorder/sessions/20260520-192626-nan10-8750`
  - `arena_frontend_monitor_runtime/manual_metric_recorder/sessions/20260522-101241`

The selected monitor summaries should include AI-readable JSON, metric summaries,
cycle smoothing/block files when available, and `analysis_report.md`. Do not
hand off raw request captures by default unless the receiver explicitly wants a
research archive.

## Optional Reference Assets

These are useful for understanding why the current training strategy was chosen:

- `nan10-8750/`: baseline code for the checkpoint that could traverse most
  terrain but had weak high-level inverse-stair ability.
- `hjc3-6/`: reference code that learned stronger stair climbing but tended to
  develop diagonal/path-selection behavior.
- `wk/`: reference code with useful gait/reward ideas.
- `external/container_src/`: small local mirror of container dependency source
  used for semantic inspection.

The historical zip files are not required if the extracted folders above are
present.

## Do Not Hand Off By Default

Exclude these from a continuation-training package:

- `.git/` when making an archive package. Use the remote repository instead for
  git history.
- `.codex_rpc/`, `agent_diy/codex_rpc_bridge_runtime/`, `codex_rpc_bridge_runtime/`.
- RPC uploads/backups, token files, browser auth/session files, and logs.
- `__pycache__/`, `*.pyc`, `.DS_Store`, editor metadata, temporary images, HAR
  files, and local runtime scratch outputs.
- Full `arena_frontend_monitor_runtime/` raw data. It is large and mostly not
  needed for continuing training.
- Model checkpoint binaries are not stored in this repository. Handoff the
  checkpoint choice separately via the Tencent platform model list.

## Recommended Handoff Procedure

1. Inspect `git status --short` and make sure the receiver gets all current
   uncommitted changes, either through a commit or a patch.
2. Run:

   ```bash
   python3 scripts/build_handoff_bundle.py --dry-run
   ```

3. Review the included and excluded paths.
4. Build the archive:

   ```bash
   python3 scripts/build_handoff_bundle.py --output /tmp/fwwb-rl-dog-handoff.tar.gz
   ```

5. Separately tell the receiver which Tencent platform checkpoint should be used
   as the next pretrain source.

## Notes For The Next Maintainer

- The training reward is not the same as platform score. Use platform completion,
  timeout, non-timeout abnormal failure, stair level metrics, and smoothed curves
  together.
- The current repository intentionally avoids previously unverified
  terrain-label-gated inverse-stair rewards.
- Height-scan stair shaping is allowed, but its reward magnitude must be checked
  against the monitor curves before long runs.
- The safe sync target is `/workspace/code`; routine sync should not upload docs,
  raw runtime data, backups, or `agent_diy`.
