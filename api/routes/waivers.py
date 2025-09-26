from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    WaiverListResponse,
    WaiverRecommendRequest,
    WaiverRecommendResponse,
    WaiverTeamResult,
)
from api.utils import build_leaderboards_map, build_team_details, build_waiver_candidates, coerce_float
from context import ContextManager
from data import normalize_name
from main import evaluate_league

router = APIRouter(prefix="/waivers", tags=["waivers"], dependencies=[Depends(require_api_key)])


@router.get("/candidates", response_model=WaiverListResponse, summary="List free-agent waiver candidates")
async def waiver_candidates(
    manager: ContextManager = Depends(get_context_manager),
    team: str | None = Query(None, description="Filter to candidates relevant for this team"),
    position: str | None = Query(None, description="Filter by position (QB/RB/WR/TE)"),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> WaiverListResponse:
    ctx = manager.get()

    league = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
    )

    df = league["df"].copy()
    rosters = league["rosters"]
    if team and team not in rosters:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown team")

    roster_name_norms = {
        normalize_name(name.replace(" (IR)", ""))
        for groups in rosters.values()
        for names in groups.values()
        for name in names
    }
    available = df[~df["name_norm"].isin(roster_name_norms)]

    if position:
        available = available[available["Position"].str.upper() == position.upper()]

    replacement_points = league.get("replacement_points", {})

    def compute_metrics(row):
        pos = row.get("Position")
        proj = row.get("ProjPoints")
        if proj is None or (isinstance(proj, float) and math.isnan(proj)):
            proj = None
        repl = replacement_points.get(pos, 0.0)
        vor = proj - repl if proj is not None else None
        ovar = None
        bench_score = None
        if vor is not None:
            vor = round(vor, 2)
            bench_score = round(max(vor, 0.0), 2)
        return vor, bench_score, ovar

    rows = []
    for _, row in available.iterrows():
        vor, bench_score, ovar = compute_metrics(row)
        rows.append({**row.to_dict(), "vor": vor, "BenchScore": bench_score, "oVAR": ovar})

    if team:
        # Simple positional need boost: prioritize positions with lower Starter VOR
        team_vor = league["starters_totals"].get(team)
        # Currently unused but placeholder for future weighting
        _ = team_vor

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            coerce_float(r.get("vor")),
            coerce_float(r.get("ProjPoints")),
        ),
        reverse=True,
    )

    total = len(rows_sorted)
    window = rows_sorted[offset : offset + limit]
    candidates = build_waiver_candidates(window)

    return WaiverListResponse(
        items=candidates,
        total=total,
        limit=limit,
        offset=offset,
        positionFilter=position.upper() if position else None,
        teamFilter=team,
    )


@router.post("/recommend", response_model=WaiverRecommendResponse, summary="Evaluate add/drop scenarios")
async def waiver_recommend(
    payload: WaiverRecommendRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> WaiverRecommendResponse:
    if not payload.changes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No changes supplied")

    ctx = manager.get()
    from main import evaluate_league  # avoid circular import at module load

    baseline = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
        custom_rosters=ctx.rosters,
    )

    df = baseline["df"].copy()
    rosters = baseline["rosters"]
    mutated = {team: {grp: list(names) for grp, names in groups.items()} for team, groups in rosters.items()}

    def find_position(player_name: str) -> str:
        row = df[df["Name"].str.lower() == player_name.lower()]
        if row.empty:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown player '{player_name}'")
        return str(row.iloc[0]["Position"])

    def remove_player(team_name: str, player_name: str) -> None:
        team_roster = mutated.get(team_name)
        if team_roster is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown team '{team_name}'")
        for group, names in team_roster.items():
            if player_name in names:
                names.remove(player_name)
                return
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{team_name} does not roster {player_name}")

    def ensure_free_agent(player_name: str) -> None:
        for team_roster in mutated.values():
            for names in team_roster.values():
                if player_name in names:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{player_name} already rostered")

    affected_teams = set()
    for change in payload.changes:
        team_name = change.team
        if team_name not in mutated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown team '{team_name}'")
        affected_teams.add(team_name)
        for drop in change.drops:
            remove_player(team_name, drop)
        for add in change.adds:
            ensure_free_agent(add)
            position = find_position(add)
            mutated[team_name].setdefault(position, [])
            mutated[team_name][position].append(add)

    new_eval = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
        custom_rosters=mutated,
    )

    baseline_combined = baseline["combined_scores"]
    new_combined = new_eval["combined_scores"]

    teams_payload = []
    for team_name in sorted(affected_teams):
        base = coerce_float(baseline_combined.get(team_name))
        post = coerce_float(new_combined.get(team_name))
        teams_payload.append(
            WaiverTeamResult(
                team=team_name,
                baseline=base,
                postChange=post,
                delta=post - base,
            )
        )

    details_payload = None
    if payload.include_details:
        details_payload = build_team_details(
            sorted(affected_teams),
            results=new_eval.get("results", {}),
            bench_tables=new_eval.get("bench_tables", {}),
            bench_limit=payload.bench_limit,
        )

    evaluated_at = datetime.now(timezone.utc)
    leaderboards_payload = build_leaderboards_map(new_eval.get("leaderboards", {}))

    return WaiverRecommendResponse(
        evaluated_at=evaluated_at,
        teams=teams_payload,
        combined_scores={team: coerce_float(val) for team, val in new_combined.items()},
        leaderboards=leaderboards_payload,
        details=details_payload,
    )
