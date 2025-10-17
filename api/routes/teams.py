from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_context_manager, require_api_key
from api.models import TeamDetail, TeamLeverageResponse, ZeroSumConcentrationRisk, ZeroSumEntry
from api.utils import build_team_details, build_zero_sum_payload
from context import ContextManager

router = APIRouter(prefix="/teams", tags=["teams"], dependencies=[Depends(require_api_key)])


@router.get("", summary="List team names")
async def list_teams(manager: ContextManager = Depends(get_context_manager)) -> list[str]:
    ctx = manager.get()
    return sorted(ctx.rosters.keys())


@router.get("/{team}", response_model=TeamDetail, summary="Get roster details for a team")
async def get_team(
    team: str,
    manager: ContextManager = Depends(get_context_manager),
) -> TeamDetail:
    ctx = manager.get()
    rosters = ctx.rosters
    if team not in rosters:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown team")

    # Build an evaluation snapshot and extract this team's detail
    from main import evaluate_league  # local import to avoid circular at module import

    league = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
        custom_rosters=ctx.rosters,
    )

    results = league.get("results", {})
    bench_tables = league.get("bench_tables", {})

    detail_list = build_team_details([team], results=results, bench_tables=bench_tables)
    if not detail_list:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to build team detail")
    return detail_list[0]


@router.get("/{team}/leverage", response_model=TeamLeverageResponse, summary="Zero-sum leverage insights for a team")
async def get_team_leverage(
    team: str,
    manager: ContextManager = Depends(get_context_manager),
) -> TeamLeverageResponse:
    ctx = manager.get()
    rosters = ctx.rosters
    if team not in rosters:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown team")

    from main import evaluate_league  # local import to avoid circular dependency at module import time

    league = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
        custom_rosters=ctx.rosters,
    )

    zero_sum_raw = league.get("zero_sum", {})
    zero_sum = build_zero_sum_payload(zero_sum_raw)

    def _lookup(group_entries):
        for entry in group_entries:
            if entry.team == team:
                return entry
        return ZeroSumEntry(team=team, value=0.0, share=0.0, surplus=0.0)

    combined_entry = _lookup(zero_sum.combined.entries)
    starters_entry = _lookup(zero_sum.starters.entries)
    bench_entry = _lookup(zero_sum.bench.entries)

    positions_map = {pos: _lookup(group.entries) for pos, group in zero_sum.positions.items()}
    bench_positions_map = {pos: _lookup(group.entries) for pos, group in zero_sum.bench_positions.items()}
    slots_map = {slot: _lookup(group.entries) for slot, group in zero_sum.slots.items()}

    team_analytics = zero_sum.analytics.teams.get(team)
    scarcity_pressure = team_analytics.scarcity_pressure if team_analytics else {}
    concentration_risk = team_analytics.concentration_risk if team_analytics else ZeroSumConcentrationRisk()

    leverage_positions = [
        pos
        for pos, entry in sorted(positions_map.items(), key=lambda item: item[1].surplus, reverse=True)
        if entry.surplus > 0
    ][:3]
    need_positions = [
        pos
        for pos, metric in sorted(scarcity_pressure.items(), key=lambda item: item[1].deficit, reverse=True)
        if metric.deficit > 0
    ][:3]

    return TeamLeverageResponse(
        team=team,
        combined=combined_entry,
        starters=starters_entry,
        bench=bench_entry,
        positions=positions_map,
        benchPositions=bench_positions_map,
        slots=slots_map,
        scarcityPressure=scarcity_pressure,
        concentrationRisk=concentration_risk,
        leveragePositions=leverage_positions,
        needPositions=need_positions,
    )
