from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_context_manager, require_api_key
from api.models import PlayerListResponse
from api.routes.players import _query_players_response
from context import ContextManager

router = APIRouter(prefix="/rankings", tags=["rankings"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=PlayerListResponse, summary="Retrieve player rankings")
async def get_rankings(
    manager: ContextManager = Depends(get_context_manager),
    pos: Optional[str] = Query(None, description="Filter by fantasy position"),
    metric: str = Query("rank", description="Metric to sort by (rank, posrank, proj, projz)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PlayerListResponse:
    return _query_players_response(
        manager,
        pos=pos,
        team=None,
        contains=None,
        metric=metric,
        limit=limit,
        offset=offset,
    )
