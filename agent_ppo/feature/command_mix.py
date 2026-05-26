# -*- coding: UTF-8 -*-
"""PPO-side command mixing for velocity-command specialization.

This does not modify Isaac Lab's command generator.  It rewrites the command
seen by the PPO observation and by custom rewards so the policy/reward contract
is internally consistent while keeping all changes inside ``agent_ppo``.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _command_mix_conf(env) -> dict:
    usr_conf = getattr(env, "usr_conf", None)
    if not isinstance(usr_conf, dict):
        return {}
    conf = usr_conf.get("command_mix", {})
    return conf if isinstance(conf, dict) else {}


def _normalized_ratios(conf: dict) -> tuple[float, float, float, float]:
    spin = max(0.0, _as_float(conf.get("spin_only_ratio", 0.0), 0.0))
    vx_only = max(0.0, _as_float(conf.get("vx_only_ratio", 0.0), 0.0))
    vx_vy = max(0.0, _as_float(conf.get("vx_vy_only_ratio", 0.0), 0.0))
    full = max(0.0, _as_float(conf.get("full_ratio", 0.0), 0.0))

    total = spin + vx_only + vx_vy + full
    if total <= 1.0:
        full += 1.0 - total
        total = 1.0

    return spin / total, vx_only / total, vx_vy / total, full / total


def _ratio(mask: torch.Tensor) -> float:
    if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
        return 0.0
    return float(mask.detach().float().mean().item())


def _mean_abs(values: torch.Tensor) -> float:
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        return 0.0
    return float(torch.abs(values.detach().float()).mean().item())


def _command_resample_hash(raw_command: torch.Tensor) -> torch.Tensor:
    """Stable per-command hash used to assign mix modes.

    The hash includes env id and quantized command values.  This keeps the mode
    stable while a sampled command is unchanged, but lets the same env move to a
    different mix mode after the command generator resamples velocity.
    """
    num_envs = int(raw_command.shape[0])
    device = raw_command.device
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    quantized = torch.round(raw_command[:, :3].detach().float() / 0.02).to(torch.long)
    hashed = env_ids * 1103515245 + 12345
    hashed = hashed + quantized[:, 0] * 374761393
    hashed = hashed + quantized[:, 1] * 668265263
    hashed = hashed + quantized[:, 2] * 2246822519
    return torch.remainder(hashed, 1000003).float() / 1000003.0


def _log_command_mix(env, debug: dict, conf: dict, site: str) -> None:
    counter_name = "_command_mix_log_count"
    count = int(getattr(env, counter_name, 0)) + 1
    setattr(env, counter_name, count)
    interval = max(1, int(_as_float(conf.get("log_interval", 200), 200)))
    if count != 1 and count % interval != 0:
        return

    message = (
        "[CommandMix] call=%d site=%s enabled=%s reason=%s "
        "target=(spin %.3f, vx_only %.3f, vx_vy %.3f, full %.3f) "
        "active=(spin %.3f, vx_only %.3f, vx_vy %.3f, full %.3f) "
        "raw_abs=(vy %.3f, wz %.3f) mixed_abs=(vy %.3f, wz %.3f)"
    ) % (
        count,
        site,
        bool(debug.get("enabled", 0.0)),
        debug.get("reason", "unknown"),
        debug.get("target_spin_ratio", 0.0),
        debug.get("target_vx_only_ratio", 0.0),
        debug.get("target_vx_vy_ratio", 0.0),
        debug.get("target_full_ratio", 0.0),
        debug.get("spin_ratio", 0.0),
        debug.get("vx_only_ratio", 0.0),
        debug.get("vx_vy_ratio", 0.0),
        debug.get("full_ratio", 0.0),
        debug.get("raw_cmd_vy_abs_mean", 0.0),
        debug.get("raw_cmd_wz_abs_mean", 0.0),
        debug.get("mixed_cmd_vy_abs_mean", 0.0),
        debug.get("mixed_cmd_wz_abs_mean", 0.0),
    )

    logger = getattr(env, "logger", None)
    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def get_mixed_command_from_raw(owner, raw_command, conf: dict | None = None, site: str = "workflow"):
    """Return a mixed copy of an existing command tensor.

    This is the reliable PPO-side path: workflow code already has obs tensors
    containing command slots, even when the platform bypasses custom observation
    processors.  ``owner`` is only used for logging/debug attachment.
    """
    if conf is None:
        conf = _command_mix_conf(owner)
    if not isinstance(conf, dict):
        conf = {}

    enabled = bool(conf.get("enabled", False))
    if not enabled:
        debug = {
            "enabled": 0.0,
            "reason": "disabled",
        }
        setattr(owner, "_command_mix_debug", debug)
        _log_command_mix(owner, debug, conf, site)
        return raw_command

    if raw_command is None or not isinstance(raw_command, torch.Tensor) or raw_command.ndim != 2 or raw_command.shape[1] < 3:
        debug = {
            "enabled": 0.0,
            "reason": "invalid_command",
        }
        setattr(owner, "_command_mix_debug", debug)
        _log_command_mix(owner, debug, conf, site)
        return raw_command

    spin_ratio, vx_only_ratio, vx_vy_ratio, full_ratio = _normalized_ratios(conf)
    num_envs = int(raw_command.shape[0])
    device = raw_command.device

    hashed = _command_resample_hash(raw_command)
    spin_cut = spin_ratio
    vx_only_cut = spin_cut + vx_only_ratio
    vx_vy_cut = vx_only_cut + vx_vy_ratio

    spin_mask = hashed < spin_cut
    vx_only_mask = (hashed >= spin_cut) & (hashed < vx_only_cut)
    vx_vy_mask = (hashed >= vx_only_cut) & (hashed < vx_vy_cut)
    full_mask = ~(spin_mask | vx_only_mask | vx_vy_mask)

    mixed = raw_command.clone()
    mixed[spin_mask, 0] = 0.0
    mixed[spin_mask, 1] = 0.0
    mixed[vx_only_mask, 1] = 0.0
    mixed[vx_only_mask, 2] = 0.0
    mixed[vx_vy_mask, 2] = 0.0

    debug = {
        "enabled": 1.0,
        "reason": "ok",
        "target_spin_ratio": float(spin_ratio),
        "target_vx_only_ratio": float(vx_only_ratio),
        "target_vx_vy_ratio": float(vx_vy_ratio),
        "target_full_ratio": float(full_ratio),
        "spin_ratio": _ratio(spin_mask),
        "vx_only_ratio": _ratio(vx_only_mask),
        "vx_vy_ratio": _ratio(vx_vy_mask),
        "full_ratio": _ratio(full_mask),
        "raw_cmd_vy_abs_mean": _mean_abs(raw_command[:, 1]),
        "raw_cmd_wz_abs_mean": _mean_abs(raw_command[:, 2]),
        "mixed_cmd_vy_abs_mean": _mean_abs(mixed[:, 1]),
        "mixed_cmd_wz_abs_mean": _mean_abs(mixed[:, 2]),
    }
    setattr(owner, "_command_mix_debug", debug)
    _log_command_mix(owner, debug, conf, site)
    return mixed


def get_mixed_command(env, command_name: str = "base_velocity", raw_command=None, site: str = "reward"):
    """Return the command after PPO-side per-env masking.

    Modes:
      - spin_only: vx=0, vy=0, wz=raw
      - vx_only: vx=raw, vy=0, wz=0
      - vx_vy_only: vx=raw, vy=raw, wz=0
      - full: raw vx/vy/wz
    """
    if raw_command is None:
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return raw_command
        raw_command = command_manager.get_command(command_name)

    return get_mixed_command_from_raw(env, raw_command, conf=_command_mix_conf(env), site=site)


def apply_command_mix_to_observation(
    env,
    obs: torch.Tensor,
    command_slice: slice,
    command_name: str = "base_velocity",
    site: str = "obs",
) -> torch.Tensor:
    """Overwrite the command slot in an observation tensor with mixed commands."""
    if obs is None or not isinstance(obs, torch.Tensor):
        return obs
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return obs
    raw_command = command_manager.get_command(command_name)
    mixed = get_mixed_command(env, command_name=command_name, raw_command=raw_command, site=site)
    if mixed is None or not isinstance(mixed, torch.Tensor):
        return obs
    out = obs.clone()
    out[:, command_slice] = mixed[:, :3].to(device=out.device, dtype=out.dtype)
    return out


def apply_command_mix_to_obs_pair(
    owner,
    obs: torch.Tensor,
    critic_obs: torch.Tensor | None,
    usr_conf: dict | None = None,
    site: str = "workflow",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply command mix directly to workflow obs and critic_obs tensors."""
    conf = {}
    if isinstance(usr_conf, dict):
        raw_conf = usr_conf.get("command_mix", {})
        conf = raw_conf if isinstance(raw_conf, dict) else {}

    mixed_command = None
    if isinstance(obs, torch.Tensor) and obs.ndim == 2 and obs.shape[1] >= 9:
        mixed_command = get_mixed_command_from_raw(owner, obs[:, 6:9], conf=conf, site=f"{site}_policy")
        if isinstance(mixed_command, torch.Tensor):
            obs = obs.clone()
            obs[:, 6:9] = mixed_command[:, :3].to(device=obs.device, dtype=obs.dtype)

    if isinstance(critic_obs, torch.Tensor) and critic_obs.ndim == 2 and critic_obs.shape[1] >= 12:
        if isinstance(mixed_command, torch.Tensor) and mixed_command.shape[0] == critic_obs.shape[0]:
            critic_mixed = mixed_command
        else:
            critic_mixed = get_mixed_command_from_raw(
                owner,
                critic_obs[:, 9:12],
                conf=conf,
                site=f"{site}_critic",
            )
        if isinstance(critic_mixed, torch.Tensor):
            critic_obs = critic_obs.clone()
            critic_obs[:, 9:12] = critic_mixed[:, :3].to(device=critic_obs.device, dtype=critic_obs.dtype)

    return obs, critic_obs
