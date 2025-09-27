from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from pathlib import Path

from api.dependencies import get_context_manager, require_api_key
from api.models import LeagueMetadataResponse, LeagueReloadRequest
from context import ContextManager

router = APIRouter(prefix="/league", tags=["league"], dependencies=[Depends(require_api_key)])


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
    )
