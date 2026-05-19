# Arena Frontend Monitor

This module is separate from `agent_diy/codex_rpc_bridge`.

It is for Tencent Arena's web frontend only:

- reuses an already logged-in `agent-browser` session
- reads the training monitor page
- reads the training log page
- exports frontend API responses for logs and metric curves
- writes local snapshot files
- renders a small local HTML dashboard with auto-refresh

It does not depend on the development container RPC bridge.

## Why this exists

Tencent Arena already exposes useful training signals in the browser:

- monitor overview groups and current values
- training log entries
- task metadata visible on the page

Even when we cannot directly consume a TensorBoard event file, we can still
build a TensorBoard-like workflow from frontend-visible data:

- periodic snapshots
- structured log history
- current monitor overview
- local HTML view for quick inspection

## Files

- `frontend_monitor.py`: collector + HTML renderer + local server
- `network_export.py`: structured exporter for `GetTrainLog` and `GetTrainMetricRange`
- `collect_monitor_overview.py`: one-shot collector for lazy-loaded monitor
  overview groups plus important training logs

Runtime output is written to:

```text
arena_frontend_monitor_runtime/
```

Manual live metric recording with a visible browser:

```bash
arena_frontend_monitor/manual_metric_recorder.sh '<monitor-url>'
arena_frontend_monitor/manual_metric_recorder.sh -c
```

The wrapper opens the monitor URL with a headed `agent-browser` session, waits
for Enter in the terminal, then starts `manual_metric_recorder.py` with
`--no-clear-network`. This is useful for live training pages: manually expand,
scroll, or refresh cards first, then press Enter so already-loaded curve
requests are kept. Extra recorder arguments can be appended after the URL. Use
`-c` or `--clipboard` to read the monitor URL from the system clipboard.
When the recorder exits, the wrapper automatically runs
`postprocess_monitor_capture.py` on the latest manual session and prints a short
terminal summary. Preprocessed files are written into the same session
directory. These files are factual data views only; they do not classify
training quality or make reward/code recommendations.

- `ai_readable_metrics.json`
- `metric_inventory.json`
- `label_series_summary.json`
- `all_metric_series_summary.json`
- `all_metric_series_lttb.json`
- `all_metric_series_smoothed.json`
- `analysis_report.md`
- `postprocess_summary.json`

## Requirements

- Python 3.10+
- `agent-browser` installed and available on `PATH`
- a valid logged-in browser session, usually:
  - `AGENT_BROWSER_SESSION=tencent-arena`
  - `AGENT_BROWSER_SESSION_NAME=tencent-arena`

## Quick start

Collect one snapshot from the current Tencent Arena monitor page:

```bash
python arena_frontend_monitor/frontend_monitor.py collect-once
```

Export structured frontend data from the currently open monitor page:

```bash
python arena_frontend_monitor/network_export.py
```

This writes a capture under:

```text
arena_frontend_monitor_runtime/network_capture/sessions/<timestamp>/
```

Key outputs:

- `summary.json`
- `views/latest_logs_compact.json`
- `views/latest_metrics_compact.json`
- `views/page_metric_inventory.json`
- `views/coverage_report.json`
- `requests/*.request.json`
- `requests/*.response.json`
- `requests/*.meta.json`

Collect lazy-loaded monitor overview groups and important training logs in one
shot:

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
python3 arena_frontend_monitor/collect_monitor_overview.py
```

For a quick targeted probe, limit the scan to one or more group names:

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
python3 arena_frontend_monitor/collect_monitor_overview.py \
  --group "地形-斜坡" \
  --group "训练进展"
```

Recommended wrapper:

```bash
arena_frontend_monitor/collect_monitor.sh
arena_frontend_monitor/collect_monitor.sh --group "步态质量" --group "地形高度剖面"
```

By default, the wrapper does not pass a monitor URL. It requires the currently
active tab to already be the intended Tencent Arena monitor page. This avoids
accidentally switching away from the task that the operator has selected.

If the active tab is not a monitor page, the script stops with an error instead
of guessing which open monitor tab is correct. For legacy behavior, pass
`--allow-monitor-tab-fallback` to allow switching to another already open
monitor tab.

Only set an explicit target page when you intentionally want to switch tasks:

```bash
MONITOR_URL="<monitor-url>" arena_frontend_monitor/collect_monitor.sh
```

This collector:

- reuses the existing monitor tab; it does not open a new health or IDE tab
- applies CSS page zoom by default (`--page-zoom 0.75`) so more monitor cards
  are exposed per viewport; pass `--page-zoom 0` only for debugging
- switches to `监控总览`
- enables `每 5 秒自动刷新`
- uses Python sleeps instead of `agent-browser wait`, because short
  `agent-browser wait` calls have been observed to hang the shared browser
  session
- reads the task status near the top of the page before capture; by default
  `进行中` uses running timestamp freshness, while completed/stopped/failed
  tasks use historical timestamp comparison
- collapses all overview groups with DOM `aria-expanded=true` clicks and a
  top-to-bottom scan; a single current-viewport snapshot is not enough on this
  page
- expands each group one by one
- drags the page's right-side scrollbar downward after expanding each group;
  this is necessary because `agent-browser scroll` and mouse wheel may not move
  Tencent Arena's monitor panel, while dragging the scrollbar has been verified
  by screenshot to reveal lower cards such as `斜坡-能耗分数` and `斜坡-步数`
- checks filtered `GetTrainMetricRange` responses after each drag until the
  effective card count reaches the number shown next to the group name
- keeps scrolling a group until it is covered, or until the scroll container
  reaches the bottom and no new metric cards appear for
  `--no-new-data-patience` rounds
- only waits for an extra fresh auto-refresh batch when `--confirm-refresh` is
  passed; normal collection moves on after coverage and timestamp checks pass
- reads each request with `agent-browser network request <id> --json`, because
  HAR files can expose request metadata while omitting response text
- switches to `训练日志`, reads `GetTrainLog`, and filters important entries by
  keywords such as `ERROR`, `WARNING`, `MonitorDebug`,
  `VelocityCurriculumDebug`, `EnvMonitor`, `Episode`, `terrain`, `reward`,
  `tracking`, and `curriculum`

Output is written under:

```text
arena_frontend_monitor_runtime/overview_capture/sessions/<timestamp>/
```

Key files:

- `summary.json`: group-level coverage, important log count, and errors
- `groups/*.json`: per-group metric requests, query names, and latest values
- `training_logs.json`: parsed training logs plus important filtered entries
- `capture.json`: full raw capture bundle

Some cards legitimately stay at `暂无数据`; the script records the captured query
coverage and continues. A group number such as `地形-斜坡( 8 )` is the number of
cards, not necessarily the number of returned query series. Terrain cards often
expand into l0-l9 query names.

For each overview group, prefer `data_status` and `exact_coverage_ok` in
`summary.json` or the group JSON:

- `data_status=ok`: the group was covered and has fresh data
- `data_status=empty`: the group was covered but every response had no points;
  this is expected for configured-but-not-reported metric groups such as
  `地形高度剖面`
- `data_status=stale`: the group was covered but latest timestamps are older
  than `--running-timestamp-max-age-ms`
- `data_status=incomplete`: the visible cards or effective metric card count
  did not reach the number next to the group name
- `data_status=polluted`: more effective metric cards were captured than the
  group declares, usually because adjacent lazy-loaded charts entered view

The script records both `request_signature_count` and `effective_card_count`.
Most groups use unique request signatures, but some groups have multiple curves
inside one request; for example `Reward指标(4)` can be covered by two request
signatures containing four returned result curves. Some groups are filtered by
query allowlists derived from `agent_ppo/conf/monitor_builder.py` to avoid
counting adjacent-group requests as current-group data.

The group JSON also includes `view_refs`, `trigger_trace`,
`filtered_out_request_signatures`, and `missing_allowlist_signatures`. If
`data_status=incomplete`, increase `--max-scrollbar-drags-per-group`, adjust
`--scrollbar-x`, or increase `--drag-wait-ms`. If `data_status=polluted`, lower
`--small-group-scrollbar-drag-dy` or add/refine the group's query allowlist.

After capture, the script returns to `监控总览` and performs a final collapse by
default. If a run is interrupted externally, check for leftover processes:

```bash
ps -axo pid,etime,command | rg 'collect_monitor_overview.py|agent-browser mouse|agent-browser wait' | rg -v rg
```

By default `--task-state-mode auto` reads the page status before capture.
Detected `进行中` tasks validate latest metric timestamps against wall-clock
time and report `timestamp_ok`. Completed, stopped, failed, or otherwise
non-running tasks are treated as historical records, so the script checks
whether captured group timestamps are close to one another instead of expecting
current wall-clock freshness. Use `--task-state-mode running` or
`--task-state-mode history` to override detection.

Recommended explicit capture command:

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"

python D:\fwwb-RL-dog\arena_frontend_monitor\network_export.py `
  --session tencent-arena `
  --session-name tencent-arena `
  --monitor-url "<monitor-url>" `
  --capture-mode auto `
  --auto-refresh-interval "每 5 秒自动刷新"
```

`network_export.py` supports four capture modes:

- `auto`: try HAR capture first, then fall back to `agent-browser network requests`
- `har`: capture and parse a browser HAR only
- `network`: use `agent-browser network requests` only
- `both`: combine both sources

Start a local auto-refresh dashboard:

```bash
python arena_frontend_monitor/frontend_monitor.py serve --interval 15 --port 8877
```

Then open:

```text
http://127.0.0.1:8877
```

## Notes

- This collector uses visible page text. It is conservative and stable, but it
  is not yet a full raw-API exporter.
- The exporter in `network_export.py` is the preferred path for AI-friendly
  reading of historical logs and metric curves.
- Historical curve loading is viewport-sensitive. The monitor page may request
  additional `GetTrainMetricRange` batches only after the relevant sections are
  scrolled into view.
- Do not reload or navigate the monitor page during capture; use an already
  open monitor tab or pass `--monitor-url`.
- Keep the IDE tab open if the development container keepalive script is also
  running.
- It is designed to be independent of the development container.
- It expects the Tencent Arena training monitor page to already be open in the
  reused browser session. If not, pass `--monitor-url`.
