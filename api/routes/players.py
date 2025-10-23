from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    PlayerComparison,
    PlayerComparisonRequest,
    PlayerComparisonResponse,
    PlayerDetail,
    PlayerInfo,
    PlayerInfoRequest,
    PlayerInfoResponse,
    PlayerListResponse,
    PlayerNewsItem,
    PlayerOwnership,
    PlayerSummary,
    ESPNPlayerSnapshot,
)
from context import ContextManager
from data import normalize_name
from espn_client import fetch_player_detail, POSITION_ID_MAP

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


@router.get("", response_model=PlayerListResponse, summary="Search players")
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


def _series_to_dict(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in row.items():
        payload[key] = _safe_value(value)
    return payload


def _compute_name_series(df: pd.DataFrame) -> pd.Series:
    if "name_norm" in df.columns:
        return df["name_norm"].astype(str).str.lower()
    if "Name" in df.columns:
        return df["Name"].fillna("").astype(str).map(lambda x: normalize_name(x)).str.lower()
    return pd.Series(["" for _ in range(len(df))], index=df.index)


def _ownership_lookup(rosters: Dict[str, Dict[str, list[str]]], norm_name: str) -> PlayerOwnership:
    for team, slots in rosters.items():
        for slot, names in slots.items():
            for raw in names:
                cleaned = raw.replace(" (IR)", "")
                if normalize_name(cleaned) == norm_name:
                    is_ir = "(IR)" in raw
                    return PlayerOwnership(
                        team=team,
                        roster_slot=slot,
                        raw_name=raw,
                        is_ir=is_ir,
                        is_free_agent=False,
                    )
    return PlayerOwnership(team=None, roster_slot=None, raw_name=None, is_ir=False, is_free_agent=True)


def _build_espn_player_map(league) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    if not league:
        return mapping
    for team in league.teams.values():
        for entry in team.roster:
            norm = normalize_name(entry.full_name)
            player_payload = (entry.raw.get("playerPoolEntry") or {}).get("player", {})
            mapping.setdefault(norm, []).append(
                {
                    "team": team,
                    "entry": entry,
                    "player": player_payload,
                }
            )
    return mapping


def _match_stats_row(df: pd.DataFrame, norm: str, name: str) -> Optional[Dict[str, Any]]:
    lower_norm = norm.lower()
    search_columns = ["name_norm", "Name", "player", "Player"]
    for column in search_columns:
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str)
        if column == "name_norm":
            matches = df[series.str.lower() == lower_norm]
        else:
            matches = df[series.str.lower() == name.lower()]
        if matches.empty:
            continue
        return _series_to_dict(matches.iloc[0])
    return None


def _gather_aliases(alias_map: Dict[str, str], canonical: str) -> list[str]:
    if not alias_map:
        return []
    aliases = [alias for alias, target in alias_map.items() if target == canonical]
    return sorted(set(aliases))


def _coerce_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return str(value)
    return str(value)


@router.post("/compare", response_model=PlayerComparisonResponse, summary="Compare multiple players")
async def compare_players(
    payload: PlayerComparisonRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> PlayerComparisonResponse:
    ctx = manager.get()
    df = ctx.dataframe
    alias_map = ctx.alias_map or {}
    rosters = ctx.rosters or {}
    projections_df = ctx.projections_df if payload.include_projections else None
    stats_data = ctx.stats_data if payload.include_stats else {}

    items: list[PlayerComparison] = []
    unresolved: list[str] = []

    name_series = _compute_name_series(df)

    for query in payload.players:
        norm_query = normalize_name(query.replace(" (IR)", "")).lower()
        matches = df[name_series == norm_query]

        if matches.empty and alias_map:
            canonical = alias_map.get(query) or alias_map.get(query.replace(" (IR)", ""))
            if canonical:
                norm_alias = normalize_name(canonical).lower()
                matches = df[name_series == norm_alias]
                if not matches.empty:
                    norm_query = norm_alias

        if matches.empty:
            approx = df[df["Name"].str.contains(query, case=False, na=False)] if "Name" in df.columns else pd.DataFrame()
            unresolved.append(query)
            suggestions = approx["Name"].head(5).tolist() if not approx.empty else []
            items.append(
                PlayerComparison(
                    query=query,
                    canonical=None,
                    matches=suggestions,
                    ownership=PlayerOwnership(is_free_agent=True),
                )
            )
            continue

        row = matches.sort_values("Rank", na_position="last").iloc[0]
        canonical_name = str(row.get("Name", query))

        ownership = _ownership_lookup(rosters, norm_query)

        projections_payload: Dict[str, Any] = {}
        if projections_df is not None and not projections_df.empty:
            proj_row = _match_stats_row(projections_df, norm_query, canonical_name)
            if proj_row:
                projections_payload = proj_row

        stats_payload: Dict[str, Dict[str, Any]] = {}
        if stats_data:
            for dataset_name, dataset_df in stats_data.items():
                if dataset_df is None or dataset_df.empty:
                    continue
                stat_row = _match_stats_row(dataset_df, norm_query, canonical_name)
                if stat_row:
                    stats_payload[dataset_name] = stat_row

        aliases: list[str] = _gather_aliases(alias_map, canonical_name) if payload.include_aliases else []

        comparison = PlayerComparison(
            query=query,
            canonical=canonical_name,
            matches=matches["Name"].tolist(),
            position=row.get("Position"),
            team=row.get("Team"),
            rank=int(_series_get(row, "Rank")) if _series_get(row, "Rank") is not None else None,
            pos_rank=int(_series_get(row, "PosRank")) if _series_get(row, "PosRank") is not None else None,
            proj_points=_series_get(row, "ProjPoints"),
            proj_z=_series_get(row, "ProjZ"),
            vor=_series_get(row, "VOR") or _series_get(row, "vor"),
            ownership=ownership,
            rankings=_series_to_dict(row),
            projections=projections_payload,
            stats=stats_payload,
            aliases=aliases,
            notes={},
        )
        items.append(comparison)

    return PlayerComparisonResponse(items=items, unresolved=unresolved)


@router.post("/insights", response_model=PlayerInfoResponse, summary="Get comprehensive player insights")
async def player_insights(
    payload: PlayerInfoRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> PlayerInfoResponse:
    ctx = manager.get()
    df = ctx.dataframe
    alias_map = ctx.alias_map or {}
    rosters = ctx.rosters or {}
    projections_df = ctx.projections_df if payload.include_projections else None
    stats_data = ctx.stats_data if payload.include_stats else {}
    espn_map = _build_espn_player_map(ctx.espn_league)

    name_series = _compute_name_series(df)

    items: List[PlayerInfo] = []
    unresolved: List[str] = []
    news_requests: Dict[int, List[PlayerInfo]] = {}

    for query in payload.players:
        norm_query = normalize_name(query.replace(" (IR)", "")).lower()
        matches = df[name_series == norm_query]

        if matches.empty and alias_map:
            canonical = alias_map.get(query) or alias_map.get(query.replace(" (IR)", ""))
            if canonical:
                norm_alias = normalize_name(canonical).lower()
                matches = df[name_series == norm_alias]
                if not matches.empty:
                    norm_query = norm_alias

        if matches.empty:
            approx = df[df["Name"].str.contains(query, case=False, na=False)] if "Name" in df.columns else pd.DataFrame()
            unresolved.append(query)
            suggestions = approx["Name"].head(5).tolist() if not approx.empty else []
            items.append(
                PlayerInfo(
                    query=query,
                    notes={"suggestions": suggestions} if suggestions else {},
                )
            )
            continue

        row = matches.sort_values("Rank", na_position="last").iloc[0]
        canonical_name = str(row.get("Name", query))

        ownership = _ownership_lookup(rosters, norm_query)

        rank_val = _series_get(row, "Rank")
        if isinstance(rank_val, float) and rank_val is not None:
            rank_val = int(rank_val)
        pos_rank_val = _series_get(row, "PosRank")
        if isinstance(pos_rank_val, float) and pos_rank_val is not None:
            pos_rank_val = int(pos_rank_val)

        proj_points_val = _series_get(row, "ProjPoints")
        proj_z_val = _series_get(row, "ProjZ")

        projections_payload: Dict[str, Any] = {}
        if projections_df is not None and not projections_df.empty:
            proj_row = _match_stats_row(projections_df, norm_query, canonical_name)
            if proj_row:
                projections_payload = proj_row

        stats_payload: Dict[str, Dict[str, Any]] = {}
        if stats_data:
            for dataset_name, dataset_df in stats_data.items():
                if dataset_df is None or dataset_df.empty:
                    continue
                stat_row = _match_stats_row(dataset_df, norm_query, canonical_name)
                if stat_row:
                    stats_payload[dataset_name] = stat_row

        aliases: List[str] = _gather_aliases(alias_map, canonical_name) if payload.include_aliases else []

        espn_candidates = espn_map.get(norm_query, [])
        espn_entry = espn_candidates[0] if espn_candidates else None
        notes: Dict[str, Any] = {}
        if ownership.team:
            notes["ownershipTeam"] = ownership.team
        if ownership.roster_slot:
            notes["ownershipSlot"] = ownership.roster_slot
        player_snapshot: Optional[ESPNPlayerSnapshot] = None
        player_id: Optional[int] = None
        if espn_entry:
            entry = espn_entry["entry"]
            team = espn_entry["team"]
            player_payload = espn_entry["player"]
            player_id = entry.player_id
            notes["espnTeamSlug"] = team.slug
            notes["espnTeamName"] = team.name
            if payload.include_espn:
                weekly_outlooks = {}
                outlooks_container = player_payload.get("outlooks") if isinstance(player_payload.get("outlooks"), dict) else {}
                if not outlooks_container and "outlooksByWeek" in player_payload:
                    weekly_outlooks = player_payload.get("outlooksByWeek") or {}
                else:
                    weekly_outlooks = outlooks_container.get("outlooksByWeek") or {}
                player_snapshot = ESPNPlayerSnapshot(
                    playerId=entry.player_id,
                    fullName=entry.full_name,
                    defaultPosition=entry.position or POSITION_ID_MAP.get(entry.default_position_id or -1),
                    lineupSlot=entry.lineup_slot,
                    injuryStatus=entry.injury_status,
                    isInjured=entry.is_injured,
                    isOnIR=entry.on_ir,
                    lastNewsDate=player_payload.get("lastNewsDate"),
                    seasonOutlook=player_payload.get("seasonOutlook"),
                    weeklyOutlooks=weekly_outlooks,
                    raw=player_payload,
                )

        info = PlayerInfo(
            query=query,
            canonical=canonical_name,
            position=row.get("Position"),
            team=row.get("Team"),
            rank=int(rank_val) if isinstance(rank_val, (int, float)) and rank_val is not None else None,
            pos_rank=int(pos_rank_val) if isinstance(pos_rank_val, (int, float)) and pos_rank_val is not None else None,
            proj_points=proj_points_val,
            proj_z=proj_z_val,
            ownership=ownership,
            rankings=_series_to_dict(row),
            projections=projections_payload,
            stats=stats_payload,
            aliases=aliases,
            espn=player_snapshot,
            notes=notes,
        )

        items.append(info)

        if payload.include_news and player_id is not None:
            news_requests.setdefault(player_id, []).append(info)

    if payload.include_news and news_requests:
        for player_id, targets in news_requests.items():
            try:
                detail_payload = fetch_player_detail(player_id, view=["news"])
            except Exception:
                continue
            if not detail_payload:
                continue
            news_items_raw = detail_payload.get("newsItems") or []
            news_items: List[PlayerNewsItem] = []
            for item in news_items_raw:
                links = item.get("links") or {}
                link_href = None
                if isinstance(links, dict):
                    web_link = links.get("web")
                    if isinstance(web_link, dict):
                        link_href = web_link.get("href")
                    elif isinstance(web_link, str):
                        link_href = web_link
                news_items.append(
                    PlayerNewsItem(
                        headline=item.get("headline"),
                        story=item.get("story"),
                        source=item.get("source"),
                        link=link_href,
                        published=_coerce_timestamp(item.get("published")),
                    )
                )

            player_payload = detail_payload.get("player") or {}
            weekly_outlooks = {}
            outlooks_container = player_payload.get("outlooks") if isinstance(player_payload.get("outlooks"), dict) else {}
            if not outlooks_container and "outlooksByWeek" in player_payload:
                weekly_outlooks = player_payload.get("outlooksByWeek") or {}
            else:
                weekly_outlooks = outlooks_container.get("outlooksByWeek") or {}

            for info in targets:
                if info.espn and player_payload:
                    info.espn.last_news_date = player_payload.get("lastNewsDate", info.espn.last_news_date)
                    info.espn.season_outlook = player_payload.get("seasonOutlook", info.espn.season_outlook)
                    if weekly_outlooks:
                        info.espn.weekly_outlooks = weekly_outlooks
                    info.espn.raw = player_payload
                info.news = news_items

    return PlayerInfoResponse(items=items, unresolved=unresolved)
