from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    LeagueActivity,
    LeagueActivityMessage,
    LeagueActivityResponse,
    LeagueMetadataResponse,
    LeagueReloadRequest,
    LeagueHistoryRequest,
    LeagueHistoryResponse,
    LeagueHistoryWindow,
)
from context import ContextManager, LeagueDataContext
from espn_client import fetch_recent_activity, fetch_player_detail
from api.history_service import collect_week_history


def _coerce_timestamp(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return str(value)

router = APIRouter(prefix="/league", tags=["league"], dependencies=[Depends(require_api_key)])


def _resolve_history_weeks(payload: LeagueHistoryRequest, ctx: LeagueDataContext) -> List[int]:
    league = ctx.espn_league
    if league is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ESPN league data unavailable.")

    status_payload = league.status or {}
    latest = status_payload.get("latestScoringPeriod") or league.scoring_period_id or status_payload.get("currentMatchupPeriod")
    first = status_payload.get("firstScoringPeriod") or 1
    final = status_payload.get("finalScoringPeriod") or latest or status_payload.get("transactionScoringPeriod")

    if latest is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Unable to determine latest scoring period.")

    if payload.window == LeagueHistoryWindow.EXPLICIT:
        weeks = sorted(set(int(week) for week in payload.weeks or [] if week and week > 0))
    elif payload.window == LeagueHistoryWindow.RANGE:
        if payload.week_range is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="weekRange is required for range window.")
        start = max(1, payload.week_range.start)
        end = max(start, payload.week_range.end)
        weeks = list(range(start, end + 1))
    elif payload.window == LeagueHistoryWindow.FULL_SEASON:
        max_week = final if final and final >= first else latest
        weeks = list(range(max(1, first), int(max_week) + 1))
    else:  # TO_DATE or default
        weeks = list(range(max(1, first), int(latest) + 1))

    if not weeks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid weeks resolved for request.")
    return weeks


@router.get("", response_model=LeagueMetadataResponse, summary="Get league metadata")
async def get_league_metadata(
    manager: ContextManager = Depends(get_context_manager),
) -> LeagueMetadataResponse:
    metadata = manager.metadata()
    return LeagueMetadataResponse(**metadata)


@router.post("/reload", response_model=LeagueMetadataResponse, summary="Reload league data from disk")
async def reload_league(
    payload: LeagueReloadRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> LeagueMetadataResponse:
    rankings_path = None
    if payload.rankings_path:
        path = Path(payload.rankings_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Rankings path not found")
        rankings_path = str(path)

    projections_path = None
    if payload.projections_path:
        path = Path(payload.projections_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Projections path not found")
        projections_path = str(path)

    supplemental_path = None
    if payload.supplemental_path:
        path = Path(payload.supplemental_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Supplemental path not found")
        supplemental_path = str(path)

    try:
        fresh = manager.reload(
            rankings_path=rankings_path,
            projections_path=projections_path,
            supplemental_path=supplemental_path,
            projection_scale_beta=payload.projection_scale_beta,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return LeagueMetadataResponse(
        team_count=len(fresh.rosters),
        player_count=int(fresh.dataframe.shape[0]),
        last_reload=fresh.created_at,
        rankings_path=str(fresh.rankings_path),
        projections_path=str(fresh.projections_path) if fresh.projections_path else None,
        supplemental_path=str(fresh.supplemental_path) if fresh.supplemental_path else None,
        settings=fresh.settings_snapshot,
        schedule_source=fresh.schedule_source,
    )


@router.get("/activity", response_model=LeagueActivityResponse, summary="Recent league activity")
async def recent_activity(
    limit: int = Query(25, ge=1, le=100, description="How many recent topics to pull from ESPN."),
    manager: ContextManager = Depends(get_context_manager),
) -> LeagueActivityResponse:
    ctx = manager.get()
    topics = fetch_recent_activity(limit=limit)
    if not topics:
        return LeagueActivityResponse(activities=[])

    team_lookup: Dict[int, Dict[str, str]] = {}
    if ctx.espn_league:
        for team in ctx.espn_league.teams.values():
            owners = ", ".join(team.managers) if team.managers else None
            team_lookup[team.team_id] = {
                "name": team.name,
                "slug": team.slug,
                "owners": owners,
            }

    player_cache: Dict[int, Dict[str, Any]] = {}
    player_lookup: Dict[int, str] = {}

    if ctx.espn_league:
        for team in ctx.espn_league.teams.values():
            for entry in team.roster:
                if entry.player_id is not None:
                    player_lookup[entry.player_id] = entry.full_name

    def _coerce_team_id(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def resolve_player(pid: Optional[int]) -> Dict[str, Any]:
        if pid is None:
            return {}
        if pid in player_cache:
            return player_cache[pid]
        name = player_lookup.get(pid)
        if name is None:
            detail = fetch_player_detail(pid, view=["playercard"])
            if detail:
                player_info = detail.get("player") or {}
                name = player_info.get("fullName") or player_info.get("firstName")
        player_cache[pid] = {"name": name}
        return player_cache[pid]

    activities: List[LeagueActivity] = []

    for topic in topics[:limit]:
        topic_type = topic.get("type") or ""
        if not topic_type.startswith("ACTIVITY"):
            continue

        messages = topic.get("messages") or []
        items: List[LeagueActivityMessage] = []
        summary_parts: List[str] = []

        for message in messages:
            player_id = message.get("targetId")
            source_team_id = _coerce_team_id(message.get("from"))
            dest_team_id = _coerce_team_id(message.get("to"))

            if source_team_id == dest_team_id:
                action = "update"
            elif source_team_id is None and dest_team_id is not None:
                action = "add"
            elif dest_team_id is None and source_team_id is not None:
                action = "drop"
            elif source_team_id is not None and dest_team_id is not None:
                action = "trade"
            else:
                action = "transaction"

            player_info = resolve_player(player_id)
            player_name = player_info.get("name") or (str(player_id) if player_id is not None else None)

            from_team_name = team_lookup.get(source_team_id, {}).get("name") if source_team_id is not None else None
            to_team_name = team_lookup.get(dest_team_id, {}).get("name") if dest_team_id is not None else None

            items.append(
                LeagueActivityMessage(
                    action=action,
                    playerId=player_id,
                    playerName=player_name,
                    fromTeam=from_team_name,
                    toTeam=to_team_name,
                    raw=message,
                )
            )

            verb = {
                "add": "added",
                "drop": "dropped",
                "trade": "traded",
                "update": "updated",
            }.get(action, "moved")

            if action == "trade" and from_team_name and to_team_name:
                summary_parts.append(f"{player_name} from {from_team_name} to {to_team_name}")
            elif action == "add" and to_team_name:
                summary_parts.append(f"{to_team_name} added {player_name}")
            elif action == "drop" and from_team_name:
                summary_parts.append(f"{from_team_name} dropped {player_name}")
            else:
                summary_parts.append(f"{verb} {player_name}")

        team_id = _coerce_team_id(topic.get("targetId"))
        team_name = team_lookup.get(team_id, {}).get("name") if team_id is not None else None

        activities.append(
            LeagueActivity(
                id=str(topic.get("id")),
                type=topic_type,
                timestamp=_coerce_timestamp(topic.get("date")),
                teamId=team_id,
                team=team_name,
                summary="; ".join(summary_parts) if summary_parts else None,
                items=items,
            )
        )

    return LeagueActivityResponse(activities=activities)


@router.post("/history", response_model=LeagueHistoryResponse, summary="Week-by-week league history")
async def league_history(
    payload: LeagueHistoryRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> LeagueHistoryResponse:
    ctx = manager.get()
    if ctx.espn_league is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ESPN league data unavailable.")

    weeks = _resolve_history_weeks(payload, ctx)
    request_summary = {
        "window": payload.window,
        "weeks": weeks,
        "teams": list(payload.teams or []),
        "includeRosters": payload.include_rosters,
        "includeMatchups": payload.include_matchups,
        "includePlayerStats": payload.include_player_stats,
        "includePowerRankings": payload.include_power_rankings,
    }

    try:
        history_payload = collect_week_history(
            ctx,
            weeks=weeks,
            team_filter=payload.teams,
            include_rosters=payload.include_rosters,
            include_matchups=payload.include_matchups,
            include_player_stats=payload.include_player_stats,
            include_power_rankings=payload.include_power_rankings,
            request_summary=request_summary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return LeagueHistoryResponse.model_validate(history_payload)
