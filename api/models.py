"""Pydantic schemas used by the API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    knobs: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)


class LeagueMetadataResponse(BaseModel):
    num_teams: int = Field(..., alias="team_count")
    player_count: int
    last_reload: datetime
    rankings_path: str
    projections_path: Optional[str] = None
    supplemental_path: Optional[str] = None
    settings: Dict[str, Any]


class LeagueReloadRequest(BaseModel):
    rankings_path: Optional[str] = None
    projections_path: Optional[str] = None
    supplemental_path: Optional[str] = None
    projection_scale_beta: Optional[float] = None


class MessageResponse(BaseModel):
    message: str
