from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import PlayerDetail, PlayerListResponse, PlayerSummary
from context import ContextManager
from data import normalize_name

router = APIRouter(prefix="/players", tags=["players"], dependencies=[Depends(require_api_key)])


_METRIC_CONFIG = {
    "rank": ("Rank", True),
    "posrank": ("PosRank", True),
    "proj": ("ProjPoints", False),
    "projz": ("ProjZ", False),
}


def _safe_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    try:
        return value.item()
    except AttributeError:
        return value


def _series_get(row: pd.Series, key: str):
    if key not in row.index:
        return None
    return _safe_value(row[key])


def _build_summary(row: pd.Series) -> PlayerSummary:
    name = row.get("Name", "")
    position = row.get("Position", "")
    team = _series_get(row, "Team")
    rank = _series_get(row, "Rank")
    if isinstance(rank, float) and rank is not None:
        rank = int(rank)
    pos_rank = _series_get(row, "PosRank")
    if isinstance(pos_rank, float) and pos_rank is not None:
        pos_rank = int(pos_rank)
    proj_points = _series_get(row, "ProjPoints")
    proj_z = _series_get(row, "ProjZ")
    player_id = row.get("name_norm") or normalize_name(str(name))
    return PlayerSummary(
        id=str(player_id),
        name=str(name),
        position=str(position),
        team=team if team is None else str(team),
        rank=rank,
        pos_rank=pos_rank,
        proj_points=proj_points,
        proj_z=proj_z,
    )


def _build_detail(row: pd.Series) -> PlayerDetail:
    summary = _build_summary(row)
    rank_original = _series_get(row, "RankOriginal")
    if isinstance(rank_original, float) and rank_original is not None:
        rank_original = int(rank_original)
    proj_points_csv = _series_get(row, "ProjPointsCsv")
    bye_week = _series_get(row, "ByeWeek")
    if isinstance(bye_week, float) and bye_week is not None:
        bye_week = int(bye_week)
    adp = _series_get(row, "ADP")
    auction_value = _series_get(row, "AuctionValue")

    used_keys = {
        "name_norm",
        "Name",
        "Position",
        "Team",
        "Rank",
        "PosRank",
        "ProjPoints",
        "ProjZ",
        "RankOriginal",
        "ProjPointsCsv",
        "ByeWeek",
        "ADP",
        "AuctionValue",
    }
    extras = {}
    for key in row.index:
        if key in used_keys:
            continue
        value = _series_get(row, key)
        if value is not None:
            extras[key] = value

    return PlayerDetail(
        **summary.dict(),
        rank_original=rank_original,
        bye_week=bye_week,
        adp=adp,
        auction_value=auction_value,
        proj_points_csv=proj_points_csv,
        extras=extras,
    )


def _query_players_response(
    manager: ContextManager,
    *,
    pos: Optional[str],
    team: Optional[str],
    contains: Optional[str],
    metric: str,
    limit: int,
    offset: int,
) -> PlayerListResponse:
    column, ascending = _METRIC_CONFIG.get(metric.lower(), (None, True))
    if column is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown metric")

    ctx = manager.get()
    df = ctx.dataframe

    subset = df
    if pos:
        subset = subset[subset["Position"].str.upper() == pos.upper()]
    if team:
        subset = subset[subset["Team"].fillna("").str.upper() == team.upper()]
    if contains:
        subset = subset[subset["Name"].str.contains(contains, case=False, na=False)]

    if column not in subset.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Metric column '{column}' missing")

    total = int(len(subset))
    if total == 0:
        return PlayerListResponse(items=[], total=0, limit=limit, offset=offset, metric=metric.lower())

    sorted_df = subset.sort_values(column, ascending=ascending, na_position="last")
    window = sorted_df.iloc[offset : offset + limit]
    items = [_build_summary(row) for _, row in window.iterrows()]
    return PlayerListResponse(items=items, total=total, limit=limit, offset=offset, metric=metric.lower())


@router.get("/", response_model=PlayerListResponse, summary="Search players")
async def list_players(
    manager: ContextManager = Depends(get_context_manager),
    pos: Optional[str] = Query(None, description="Filter by fantasy position (QB/RB/WR/TE)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    contains: Optional[str] = Query(None, description="Substring match on player name"),
    metric: str = Query("rank", description="Sort metric: rank, posrank, proj, projz"),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PlayerListResponse:
    return _query_players_response(
        manager,
        pos=pos,
        team=team,
        contains=contains,
        metric=metric,
        limit=limit,
        offset=offset,
    )


@router.get("/{player_id}", response_model=PlayerDetail, summary="Get player detail")
async def get_player(
    player_id: str,
    manager: ContextManager = Depends(get_context_manager),
) -> PlayerDetail:
    ctx = manager.get()
    df = ctx.dataframe

    norm_id = normalize_name(player_id)
    candidate = df[df["name_norm"] == norm_id] if "name_norm" in df.columns else pd.DataFrame()
    if candidate.empty:
        candidate = df[df["Name"].str.lower() == player_id.lower()]
    if candidate.empty and ctx.alias_map:
        canonical = ctx.alias_map.get(player_id) or ctx.alias_map.get(player_id.replace(" (IR)", ""))
        if canonical:
            norm = normalize_name(canonical)
            candidate = df[df["name_norm"] == norm]
    if candidate.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Player not found")

    row = candidate.sort_values("Rank").iloc[0]
    return _build_detail(row)
