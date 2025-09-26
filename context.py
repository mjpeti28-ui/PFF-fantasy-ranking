"""Shared league data context and reload management utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

import pandas as pd

from alias import build_alias_map
from config import PROJECTIONS_CSV, settings
from data import build_lookups, load_rankings, load_rosters
from optimizer import flatten_league_names


DEFAULT_RANKINGS_PATH = Path(__file__).with_name("ROS_week_2_PFF_rankings.csv")
DEFAULT_SUPPLEMENTAL_PATH = Path(__file__).with_name("PFF_rankings.csv")


@dataclass(frozen=True)
class LeagueDataContext:
    """Immutable snapshot of the inputs required to evaluate the league."""

    dataframe: pd.DataFrame
    rank_by_name: Dict[str, int]
    pos_by_name: Dict[str, str]
    posrank_by_name: Dict[str, int]
    proj_by_name: Dict[str, Optional[float]]
    alias_map: Dict[str, Optional[str]]
    rosters: Dict[str, Dict[str, list[str]]]
    csv_max_per_pos: Dict[str, int]
    created_at: datetime
    rankings_path: Path
    projections_path: Optional[Path]
    supplemental_path: Optional[Path]
    settings_snapshot: Dict[str, Any]


def build_context(
    rankings_path: Path,
    *,
    projections_path: Optional[Path] = None,
    supplemental_path: Optional[Path] = None,
    projection_scale_beta: Optional[float] = None,
    custom_rosters: Optional[Dict[str, Dict[str, list[str]]]] = None,
) -> LeagueDataContext:
    """Construct a fresh :class:`LeagueDataContext`.

    Parameters mirror :func:`data.load_rankings`. ``custom_rosters`` may be used
    to evaluate hypothetical leagues (trades, waivers, etc.).
    """

    rankings_path = Path(rankings_path)
    projections_path = Path(projections_path) if projections_path else None
    supplemental_path = Path(supplemental_path) if supplemental_path else None

    df = load_rankings(
        str(rankings_path),
        str(projections_path) if projections_path else None,
        projection_scale_beta=projection_scale_beta,
        supplemental_path=str(supplemental_path) if supplemental_path else None,
    )
    rank_by_name, pos_by_name, posrank_by_name, proj_by_name = build_lookups(df)
    csv_max_per_pos = df.groupby("Position")["Rank"].max().to_dict()

    rosters = load_rosters() if custom_rosters is None else copy.deepcopy(custom_rosters)
    league_names = flatten_league_names(rosters)
    alias_map = build_alias_map(league_names, df)

    snapshot = LeagueDataContext(
        dataframe=df,
        rank_by_name=rank_by_name,
        pos_by_name=pos_by_name,
        posrank_by_name=posrank_by_name,
        proj_by_name=proj_by_name,
        alias_map=alias_map,
        rosters=rosters,
        csv_max_per_pos=csv_max_per_pos,
        created_at=datetime.now(timezone.utc),
        rankings_path=rankings_path,
        projections_path=projections_path,
        supplemental_path=supplemental_path,
        settings_snapshot=settings.snapshot(),
    )
    return snapshot


class ContextManager:
    """Manage the active :class:`LeagueDataContext` with atomic reloads."""

    def __init__(
        self,
        *,
        rankings_path: Path | str | None = None,
        projections_path: Path | str | None = None,
        supplemental_path: Path | str | None = None,
    ) -> None:
        self._lock = RLock()
        self._rankings_path = Path(rankings_path) if rankings_path else DEFAULT_RANKINGS_PATH
        self._projections_path = Path(projections_path) if projections_path else Path(PROJECTIONS_CSV)
        if supplemental_path is None and DEFAULT_SUPPLEMENTAL_PATH.exists():
            self._supplemental_path = DEFAULT_SUPPLEMENTAL_PATH
        else:
            self._supplemental_path = Path(supplemental_path) if supplemental_path else None
        self._context: Optional[LeagueDataContext] = None

    def get(self) -> LeagueDataContext:
        """Return the current context, loading it lazily if needed."""

        with self._lock:
            if self._context is None:
                self._context = build_context(
                    self._rankings_path,
                    projections_path=self._projections_path,
                    supplemental_path=self._supplemental_path,
                )
            return self._context

    def reload(
        self,
        *,
        rankings_path: Path | str | None = None,
        projections_path: Path | str | None = None,
        supplemental_path: Path | str | None = None,
        projection_scale_beta: Optional[float] = None,
    ) -> LeagueDataContext:
        """Reload inputs and swap in a brand-new context atomically."""

        new_rankings = Path(rankings_path) if rankings_path else self._rankings_path
        new_projections = Path(projections_path) if projections_path else self._projections_path
        if supplemental_path is None:
            new_supp = self._supplemental_path
        else:
            new_supp = Path(supplemental_path)

        fresh = build_context(
            new_rankings,
            projections_path=new_projections,
            supplemental_path=new_supp,
            projection_scale_beta=projection_scale_beta,
        )

        with self._lock:
            self._rankings_path = new_rankings
            self._projections_path = new_projections
            self._supplemental_path = new_supp
            self._context = fresh
            return self._context

    def metadata(self) -> Dict[str, Any]:
        """Return lightweight info about the active context."""

        ctx = self.get()
        return {
            "last_reload": ctx.created_at.isoformat(),
            "rankings_path": str(ctx.rankings_path),
            "projections_path": str(ctx.projections_path) if ctx.projections_path else None,
            "supplemental_path": str(ctx.supplemental_path) if ctx.supplemental_path else None,
            "settings": ctx.settings_snapshot,
            "team_count": len(ctx.rosters),
            "player_count": int(ctx.dataframe.shape[0]),
        }


# Global singleton used by the CLI/API layers.
context_manager = ContextManager()

