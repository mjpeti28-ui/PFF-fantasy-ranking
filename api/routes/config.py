from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import require_api_key
from api.models import ConfigResponse, ConfigUpdateRequest
from config import SETTINGS_HELP, settings

router = APIRouter(prefix="/config", tags=["config"], dependencies=[Depends(require_api_key)])


@router.get("/", response_model=ConfigResponse, summary="List current scoring knobs")
async def get_config() -> ConfigResponse:
    return ConfigResponse(knobs=settings.snapshot())


@router.patch("/", response_model=ConfigResponse, summary="Update one or more scoring knobs")
async def patch_config(payload: ConfigUpdateRequest) -> ConfigResponse:
    if not payload.updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")
    snapshot = settings.snapshot()
    for name, value in payload.updates.items():
        if name not in snapshot:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown knob '{name}'")
        settings.set(name, value)
    return ConfigResponse(knobs=settings.snapshot())


@router.get("/help", summary="Describe available configuration knobs")
async def config_help() -> dict[str, str]:
    return SETTINGS_HELP.copy()
