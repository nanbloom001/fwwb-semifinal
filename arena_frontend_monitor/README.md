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

Runtime output is written to:

```text
arena_frontend_monitor_runtime/
```

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
