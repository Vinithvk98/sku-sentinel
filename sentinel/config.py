"""Central configuration: config.yaml at the project root, with safe defaults.

Every tunable in one place means users adjust behavior without reading source,
and reviewers can see the whole operating point at a glance.
"""
from __future__ import annotations

import pathlib

DEFAULTS = {
    "simulation": {"n_skus": 200, "days": 730, "events_per_type": 4, "seed": 42},
    "monitor": {
        "window_days": 28,
        "checkpoint_every": 7,
        "wape_factor": 1.5,
        "bias_floor": 0.30,
        "persist_checkpoints": 3,
        "psi_base_threshold": 0.25,
        "ks_p_value": 0.001,
    },
    "triage": {"horizon_days": 28},
}


def _load() -> dict:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    if path.exists():
        try:
            import yaml

            user = yaml.safe_load(path.read_text()) or {}
            for section, values in user.items():
                if section in cfg and isinstance(values, dict):
                    cfg[section].update(values)
        except Exception as exc:  # bad YAML should not brick the pipeline
            print(f"[sentinel] warning: could not read config.yaml ({exc}); using defaults")
    return cfg


CFG = _load()
