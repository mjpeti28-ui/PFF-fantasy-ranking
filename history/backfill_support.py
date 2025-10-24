from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tools.backfill_snapshots import BackfillOptions, InputResolver, run_backfill


ARCHIVED_RANKINGS_MAP: Dict[int, str] = {
    1: "history/rankings/20250924T185619_PFF_rankings.csv",
    2: "history/rankings/20250924T185619_ROS_week_2_PFF_rankings.csv",
    3: "history/rankings/20250927T091433_PFF_rankings.csv",
    4: "history/rankings/20250930T110428_ROS_week_2_PFF_rankings.csv",
    5: "history/rankings/20251007T132710_PFF_rankings.csv",
    6: "history/rankings/20251007T142116_PFF_rankings.csv",
    7: "history/rankings/20251014T083003_PFF_rankings.csv",
    8: "history/rankings/20251021T162627_PFF_rankings.csv",
}

ARCHIVED_PROJECTIONS_MAP: Dict[int, str] = {
    1: "history/projections/20250924T185619_projections.csv",
    2: "history/projections/20250924T192028_projections.csv",
    3: "history/projections/20250927T091433_projections.csv",
    4: "history/projections/20250930T110428_projections.csv",
    5: "history/projections/20251007T132710_projections.csv",
    6: "history/projections/20251007T142116_projections.csv",
    7: "history/projections/20251014T083003_projections.csv",
    8: "history/projections/20251021T162627_projections.csv",
}

DEFAULT_WEEKS = sorted(ARCHIVED_RANKINGS_MAP.keys())


class _MappingResolver(InputResolver):
    def __init__(self, rankings_map: Dict[int, str], projections_map: Dict[int, str]):
        super().__init__()
        self.rankings_map = rankings_map
        self.projections_map = projections_map

    def resolve_rankings(self, week: int, default_path: str) -> str:
        path = self.rankings_map.get(week)
        if path is None:
            return super().resolve_rankings(week, default_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Rankings file not found for week {week}: {path}")
        return path

    def resolve_projections(self, week: int, default_path: str | None) -> str | None:
        path = self.projections_map.get(week)
        if path is None:
            return super().resolve_projections(week, default_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Projections file not found for week {week}: {path}")
        return path


def ensure_backfill_snapshots(*, weeks: List[int] | None = None) -> None:
    target_weeks = weeks or DEFAULT_WEEKS

    historical_options = BackfillOptions(
        weeks=target_weeks,
        mode="historical",
        source="backfill.historical",
        base_tags=("backfill", "archived"),
        resolver=_MappingResolver(ARCHIVED_RANKINGS_MAP, ARCHIVED_PROJECTIONS_MAP),
    )
    run_backfill(historical_options)

    current_options = BackfillOptions(
        weeks=target_weeks,
        mode="historical",
        source="backfill.historical-current",
        base_tags=("backfill", "current"),
    )
    run_backfill(current_options)
