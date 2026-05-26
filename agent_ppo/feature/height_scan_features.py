# -*- coding: UTF-8 -*-
"""Shared geometry features derived from the 16x16 height scan.

The scan convention used by the environment is ``scanner_z - hit_z``. Values
below zero mean the ray hit something above the scanner plane. Maze walls are
tall, laterally continuous, and have sharper jumps than stair steps. Stairs are
lower, forward-progressive discontinuities.
"""

from __future__ import annotations

import torch


def maze_wall_stair_features(
    grid: torch.Tensor,
    *,
    y_bands: tuple[tuple[int, int], ...] = ((0, 5), (5, 11), (11, 16)),
    x_start: int = 1,
    x_end: int = 11,
    wall_height_threshold: float = 0.18,
    body_clearance_threshold: float = 0.30,
    wall_jump_threshold: float = 0.16,
    stair_min_delta: float = 0.03,
    stair_max_delta: float = 0.24,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Return compact wall/stair geometry features for each env.

    Args:
        grid: Tensor shaped ``(num_envs, 16, 16)``. Negative values represent
            obstacles above the scanner plane.
        y_bands: Left/center/right lateral bands. The second grid dimension is
            treated as lateral, and the third as forward.
        x_start/x_end: Forward columns used for near-front obstacle analysis.
        wall_height_threshold: Minimum above-scanner height for a tall blocker.
        body_clearance_threshold: Stronger "higher than body/shoulder" cue.
        wall_jump_threshold: Large forward discontinuity cue for vertical walls.
        stair_min_delta/stair_max_delta: Moderate deltas typical of stair edges.
    """
    if grid.dim() != 3:
        raise ValueError(f"height scan grid must be 3-D, got shape={tuple(grid.shape)}")
    if grid.shape[1] < 16 or grid.shape[2] < 16:
        raise ValueError(f"height scan grid must contain at least 16x16 samples, got shape={tuple(grid.shape)}")

    x0 = max(0, min(int(x_start), 15))
    x1 = max(x0 + 2, min(int(x_end), 16))

    band_wall_scores = []
    band_free_scores = []
    band_stair_scores = []
    band_wall_distances = []

    for y_start, y_end in y_bands:
        y0 = max(0, min(int(y_start), 15))
        y1 = max(y0 + 1, min(int(y_end), 16))
        region = grid[:, y0:y1, x0:x1]
        obstacle_height = torch.clamp(-region, min=0.0)

        high = obstacle_height > wall_height_threshold
        body_high = obstacle_height > body_clearance_threshold
        high_density = high.float().mean(dim=(1, 2))
        body_high_density = body_high.float().mean(dim=(1, 2))

        # A maze wall tends to occupy a continuous lateral slice at some
        # forward distance; stair edges are less likely to fill a whole band.
        lateral_continuity = high.float().mean(dim=1).max(dim=1).values
        peak_height = obstacle_height.amax(dim=(1, 2))
        tallness = torch.clamp((peak_height - wall_height_threshold) / max(body_clearance_threshold, eps), 0.0, 2.0)

        if region.shape[-1] > 1:
            forward_delta = torch.abs(region[:, :, 1:] - region[:, :, :-1])
            large_jump = forward_delta > wall_jump_threshold
            moderate_step = (forward_delta > stair_min_delta) & (forward_delta < stair_max_delta)
            jump_density = large_jump.float().mean(dim=(1, 2))
            stair_edge_density = moderate_step.float().mean(dim=(1, 2))
        else:
            jump_density = torch.zeros(grid.shape[0], device=grid.device, dtype=grid.dtype)
            stair_edge_density = torch.zeros_like(jump_density)

        # Continuity alone must not classify one isolated high ray as a wall.
        # It only amplifies density/jump/body-height evidence.
        wall_evidence = 0.50 * high_density + 0.30 * jump_density + 0.20 * body_high_density
        wall_score = wall_evidence * (0.35 + 0.65 * lateral_continuity) * (1.0 + 0.35 * tallness)
        stair_score = stair_edge_density * torch.clamp(1.0 - body_high_density, min=0.0)

        # Free score is intentionally conservative: tall/body-high blockers
        # make the corridor expensive even when only part of the band is hit.
        free_score = torch.clamp(1.0 - 0.65 * high_density - 0.35 * lateral_continuity - body_high_density, 0.0, 1.0)

        high_any_per_x = high.any(dim=1)
        first_hit = torch.argmax(high_any_per_x.float(), dim=1).to(grid.dtype)
        has_hit = high_any_per_x.any(dim=1)
        normalized_distance = first_hit / max(float(x1 - x0 - 1), 1.0)
        wall_distance = torch.where(has_hit, normalized_distance, torch.ones_like(normalized_distance))

        band_wall_scores.append(wall_score)
        band_free_scores.append(free_score)
        band_stair_scores.append(stair_score)
        band_wall_distances.append(wall_distance)

    wall_scores = torch.stack(band_wall_scores, dim=1)
    free_scores = torch.stack(band_free_scores, dim=1)
    stair_scores = torch.stack(band_stair_scores, dim=1)
    wall_distances = torch.stack(band_wall_distances, dim=1)

    left_wall, center_wall, right_wall = wall_scores[:, 0], wall_scores[:, 1], wall_scores[:, 2]
    left_free, center_free, right_free = free_scores[:, 0], free_scores[:, 1], free_scores[:, 2]
    front_wall_score = torch.maximum(center_wall, torch.maximum(left_wall, right_wall))
    stair_score = stair_scores.mean(dim=1)
    turn_bias = torch.clamp(right_free - left_free, -1.0, 1.0)
    center_wall_distance = wall_distances[:, 1]

    return {
        "left_wall_score": left_wall,
        "center_wall_score": center_wall,
        "right_wall_score": right_wall,
        "front_wall_score": front_wall_score,
        "stair_score": stair_score,
        "turn_bias": turn_bias,
        "center_free_score": center_free,
        "center_wall_distance": center_wall_distance,
        "wall_scores": wall_scores,
        "free_scores": free_scores,
        "stair_scores": stair_scores,
    }


def compact_maze_scan_features(
    grid: torch.Tensor,
    *,
    wall_height_threshold: float = 0.18,
    body_clearance_threshold: float = 0.30,
    wall_jump_threshold: float = 0.16,
    stair_min_delta: float = 0.03,
    stair_max_delta: float = 0.24,
) -> torch.Tensor:
    """Six observation features kept compatible with ``num_extra_obs = 6``."""
    features = maze_wall_stair_features(
        grid,
        wall_height_threshold=wall_height_threshold,
        body_clearance_threshold=body_clearance_threshold,
        wall_jump_threshold=wall_jump_threshold,
        stair_min_delta=stair_min_delta,
        stair_max_delta=stair_max_delta,
    )
    return torch.stack(
        (
            features["left_wall_score"],
            features["center_wall_score"],
            features["right_wall_score"],
            features["stair_score"],
            features["turn_bias"],
            features["center_wall_distance"],
        ),
        dim=1,
    )
