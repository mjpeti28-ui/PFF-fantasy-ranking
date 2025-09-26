from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_context_manager, require_api_key
from api.models import TeamDetail
from api.utils import build_team_details
from context import ContextManager

router = APIRouter(prefix="/teams", tags=["teams"], dependencies=[Depends(require_api_key)])


@router.get("/", summary="List team names")
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
