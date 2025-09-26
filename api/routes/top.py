from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_context_manager, require_api_key
from context import ContextManager
from api.routes.players import _query_players_response

router = APIRouter(prefix="/top", tags=["top"], dependencies=[Depends(require_api_key)])


@router.get("/players", summary="Convenience wrapper for top players by metric")
async def top_players(
    manager: ContextManager = Depends(get_context_manager),
    pos: str | None = Query(None, description="Position filter (QB/RB/WR/TE)"),
    metric: str = Query("rank", description="Metric to sort by: rank, posrank, proj, projz"),
    limit: int = Query(10, ge=1, le=100),
) -> dict:
    response = _query_players_response(
        manager,
        pos=pos,
        team=None,
        contains=None,
        metric=metric,
        limit=limit,
        offset=0,
    )
    return response.model_dump()
