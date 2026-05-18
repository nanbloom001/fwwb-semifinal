#!/usr/bin/env python3
"""Build compact, analysis-friendly views from a monitor overview capture.

The collector stores full frontend responses so no information is lost, but the
raw capture is too large for routine comparison. This postprocessor extracts all
metric series, keeps a full normalized copy, and adds shape-preserving downsampled
and rolling-smoothed views.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path, help="overview_capture/sessions/<timestamp> directory")
    parser.add_argument("--downsample-points", type=int, default=80)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument(
        "--no-full-series",
        action="store_true",
        help="skip writing all_metric_series.json; raw group responses still remain unchanged",
    )
    return parser.parse_args()


def _metric_name(result_id: str | None) -> str:
    if not result_id:
        return ""
    stem, sep, suffix = result_id.rpartition("_")
    if sep and suffix.isdigit():
        return stem
    return result_id


def extract_series(capture_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, dict[str, Any]]]:
    groups_dir = capture_dir / "groups"
    if not groups_dir.is_dir():
        raise FileNotFoundError(f"missing groups directory: {groups_dir}")

    series: dict[str, list[tuple[int, float]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    for group_file in sorted(groups_dir.glob("*.json")):
        data = json.loads(group_file.read_text())
        group_name = data.get("group", {}).get("name") or group_file.stem

        query_expr_by_id: dict[str, str] = {}
        query_name_by_id: dict[str, str] = {}
        for detail in data.get("metric_details", []):
            post_data = detail.get("post_data") or {}
            for query in post_data.get("queries") or []:
                qid = query.get("id")
                if qid:
                    query_expr_by_id[qid] = query.get("expr", "")
                    query_name_by_id[qid] = query.get("name", _metric_name(qid))

            response_json = detail.get("response_json") or {}
            for result in response_json.get("data", {}).get("results", []) or []:
                result_id = result.get("id") or ""
                name = query_name_by_id.get(result_id) or _metric_name(result_id)
                if not name:
                    continue

                values: list[tuple[int, float]] = []
                for item in result.get("items") or []:
                    for point in item.get("values") or []:
                        try:
                            values.append((int(point["timestamp"]), float(point["value"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                if not values:
                    continue

                series.setdefault(name, []).extend(values)
                meta.setdefault(
                    name,
                    {
                        "group": group_name,
                        "expr": query_expr_by_id.get(result_id, ""),
                        "result_ids": [],
                    },
                )
                if result_id and result_id not in meta[name]["result_ids"]:
                    meta[name]["result_ids"].append(result_id)

    for name, values in list(series.items()):
        dedup = {timestamp: value for timestamp, value in values}
        series[name] = sorted(dedup.items())

    return series, meta


def lttb_downsample(points: list[tuple[int, float]], threshold: int) -> list[tuple[int, float]]:
    """Largest-Triangle-Three-Buckets downsampling.

    Keeps visual shape substantially better than uniform sampling while still
    preserving the first and last points.
    """
    data_length = len(points)
    if threshold <= 0 or threshold >= data_length:
        return list(points)
    if threshold < 3:
        return [points[0], points[-1]][:threshold]

    sampled = [points[0]]
    bucket_size = (data_length - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_range_start = int(math.floor((i + 1) * bucket_size)) + 1
        avg_range_end = int(math.floor((i + 2) * bucket_size)) + 1
        avg_range_end = min(avg_range_end, data_length)
        avg_range = points[avg_range_start:avg_range_end]
        if avg_range:
            avg_x = sum(p[0] for p in avg_range) / len(avg_range)
            avg_y = sum(p[1] for p in avg_range) / len(avg_range)
        else:
            avg_x, avg_y = points[-1]

        range_offs = int(math.floor(i * bucket_size)) + 1
        range_to = int(math.floor((i + 1) * bucket_size)) + 1
        range_to = min(range_to, data_length - 1)

        point_a_x, point_a_y = points[a]
        max_area = -1.0
        next_a = range_offs
        for idx in range(range_offs, range_to):
            point_x, point_y = points[idx]
            area = abs(
                (point_a_x - avg_x) * (point_y - point_a_y)
                - (point_a_x - point_x) * (avg_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                next_a = idx

        sampled.append(points[next_a])
        a = next_a

    sampled.append(points[-1])
    return sampled


def centered_moving_average(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(points) <= 2:
        return list(points)
    half = window // 2
    smoothed: list[tuple[int, float]] = []
    values = [value for _, value in points]
    for idx, (timestamp, _) in enumerate(points):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)
        smoothed.append((timestamp, sum(values[start:end]) / (end - start)))
    return smoothed


def summarize(points: list[tuple[int, float]], meta: dict[str, Any]) -> dict[str, Any]:
    values = [value for _, value in points]
    first20 = values[: min(20, len(values))]
    last20 = values[-min(20, len(values)) :]
    max_idx = max(range(len(values)), key=values.__getitem__)
    min_idx = min(range(len(values)), key=values.__getitem__)

    def ts_text(timestamp: int) -> str:
        return datetime.fromtimestamp(timestamp / 1000).isoformat(timespec="seconds")

    return {
        "group": meta.get("group", ""),
        "expr": meta.get("expr", ""),
        "count": len(points),
        "start_time": ts_text(points[0][0]),
        "end_time": ts_text(points[-1][0]),
        "first": values[0],
        "mid": values[len(values) // 2],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "first20_avg": statistics.fmean(first20),
        "last20_avg": statistics.fmean(last20),
        "delta": values[-1] - values[0],
        "max_time": ts_text(points[max_idx][0]),
        "min_time": ts_text(points[min_idx][0]),
    }


def encode_points(points: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    return [{"ts": timestamp, "value": value} for timestamp, value in points]


def main() -> None:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    series, meta = extract_series(capture_dir)

    summary = {
        name: summarize(points, meta.get(name, {}))
        for name, points in sorted(series.items())
    }
    downsampled = {
        name: encode_points(lttb_downsample(points, args.downsample_points))
        for name, points in sorted(series.items())
    }
    smoothed = {
        name: encode_points(centered_moving_average(points, args.smooth_window))
        for name, points in sorted(series.items())
    }

    (capture_dir / "all_metric_series_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (capture_dir / "all_metric_series_lttb.json").write_text(
        json.dumps(downsampled, ensure_ascii=False)
    )
    (capture_dir / "all_metric_series_smoothed.json").write_text(
        json.dumps(smoothed, ensure_ascii=False)
    )
    if not args.no_full_series:
        (capture_dir / "all_metric_series.json").write_text(
            json.dumps(
                {name: encode_points(points) for name, points in sorted(series.items())},
                ensure_ascii=False,
            )
        )

    group_counts: dict[str, int] = {}
    for item in summary.values():
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
    overview = {
        "capture_dir": str(capture_dir),
        "series_count": len(series),
        "downsample_points": args.downsample_points,
        "smooth_window": args.smooth_window,
        "group_counts": dict(sorted(group_counts.items())),
        "outputs": {
            "summary": "all_metric_series_summary.json",
            "full_series": None if args.no_full_series else "all_metric_series.json",
            "lttb": "all_metric_series_lttb.json",
            "smoothed": "all_metric_series_smoothed.json",
        },
    }
    (capture_dir / "postprocess_summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2)
    )
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
