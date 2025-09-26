"""Runtime configuration knobs for the fantasy ranking toolkit.

Historically the project relied on module-level constants (e.g., ``FUZZY_CUTOFF``)
that were imported throughout the codebase. To support runtime overrides and an
API layer, we now expose a thread-safe :class:`SettingsManager` that stores the
same knobs in a mutable dictionary. Modules should call ``settings.get(...)``
instead of importing the constants directly, but the legacy constants are still
defined (as the original default values) so existing CLI scripts continue to
work while the migration is in progress.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict


class SettingsManager:
    """Thread-safe accessor for mutable scoring knobs.

    The manager stores a copy of the default settings and exposes ``get``/``set``
    helpers. ``snapshot`` returns an immutable dictionary that can be embedded in
    API responses without risking mid-request mutation.
    """

    def __init__(self, defaults: Dict[str, Any]) -> None:
        self._defaults = dict(defaults)
        self._settings = dict(defaults)
        self._lock = RLock()

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._settings.keys())

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._settings:
                raise KeyError(f"Unknown setting '{name}'")
            return self._settings[name]

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            if name not in self._settings:
                raise KeyError(f"Unknown setting '{name}'")
            self._settings[name] = value

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._settings = dict(self._defaults)
                return
            if name not in self._defaults:
                raise KeyError(f"Unknown setting '{name}'")
            self._settings[name] = self._defaults[name]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._settings)


_DEFAULT_SETTINGS: Dict[str, Any] = {
    "fuzzy_cutoff": 0.84,
    "bench_ovar_beta": 0.25,
    "combined_starters_weight": 0.80,
    "combined_bench_weight": 0.20,
    "projection_scale_beta": 0.50,
    "bench_z_fallback_threshold": 3.0,
    "bench_percentile_clamp": 0.05,
}

SETTINGS_HELP: Dict[str, str] = {
    "fuzzy_cutoff": "Minimum similarity required when matching roster names to ranking entries (higher = stricter).",
    "bench_ovar_beta": "Weight applied to overall rank advantage (oVAR) when scoring bench players.",
    "combined_starters_weight": "Proportion of the combined score attributed to starter strength.",
    "combined_bench_weight": "Proportion of the combined score attributed to bench depth.",
    "projection_scale_beta": "Influence of projection z-scores when rescaling raw rankings.",
    "bench_z_fallback_threshold": "If bench-score variance drops below this threshold, percentile z-scores are used instead of standard z-scores.",
    "bench_percentile_clamp": "Clamp applied to percentile-based z-scores to limit outlier influence.",
}

settings = SettingsManager(_DEFAULT_SETTINGS)


def set_knob(name: str, value: Any) -> None:
    settings.set(name, value)


def get_knob(name: str) -> Any:
    return settings.get(name)


def all_knobs() -> Dict[str, Any]:
    return settings.snapshot()


# Legacy constants (retain initial defaults for backwards compatibility). These
# should be phased out in favour of ``settings.get``.
FUZZY_CUTOFF = _DEFAULT_SETTINGS["fuzzy_cutoff"]
BENCH_OVAR_BETA = _DEFAULT_SETTINGS["bench_ovar_beta"]
COMBINED_STARTERS_WEIGHT = _DEFAULT_SETTINGS["combined_starters_weight"]
COMBINED_BENCH_WEIGHT = _DEFAULT_SETTINGS["combined_bench_weight"]
PROJECTION_SCALE_BETA = _DEFAULT_SETTINGS["projection_scale_beta"]
BENCH_Z_FALLBACK_THRESHOLD = _DEFAULT_SETTINGS["bench_z_fallback_threshold"]
BENCH_PERCENTILE_CLAMP = _DEFAULT_SETTINGS["bench_percentile_clamp"]

# Projection input file remains a simple constant; individual requests can pass
# alternate paths without mutating global state.
PROJECTIONS_CSV = "projections.csv"


# Slots used in optimization (K/DST omitted from scoring)
SLOT_DEFS = [
    ("QB", {"QB"}),
    ("RB", {"RB"}),
    ("RB/WR", {"RB", "WR"}),
    ("WR", {"WR"}),
    ("WR/TE", {"WR", "TE"}),
    ("TE", {"TE"}),
    ("WR/RB/TE", {"WR", "RB", "TE"}),
]

# Positions we actually score
SCORABLE_POS = {"QB", "RB", "WR", "TE"}

# Columns expected in rankings CSV (rename map)
CSV_RENAME = {"RankRk.": "Rank", "PositionPos.": "Position"}
