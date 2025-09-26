from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_context_manager, require_api_key
from api.models import LeagueMetadataResponse, LeagueReloadRequest
from context import ContextManager

router = APIRouter(prefix="/league", tags=["league"], dependencies=[Depends(require_api_key)])


@router.get("/", response_model=LeagueMetadataResponse, summary="Get league metadata")
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
    fresh = manager.reload(
        rankings_path=payload.rankings_path,
        projections_path=payload.projections_path,
        supplemental_path=payload.supplemental_path,
        projection_scale_beta=payload.projection_scale_beta,
    )
    return LeagueMetadataResponse(
        team_count=len(fresh.rosters),
        player_count=int(fresh.dataframe.shape[0]),
        last_reload=fresh.created_at,
        rankings_path=str(fresh.rankings_path),
        projections_path=str(fresh.projections_path) if fresh.projections_path else None,
        supplemental_path=str(fresh.supplemental_path) if fresh.supplemental_path else None,
        settings=fresh.settings_snapshot,
    )
