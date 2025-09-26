"""Shared FastAPI dependencies (auth, context access, etc.)."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from context import ContextManager, context_manager


class APISettings:
    """Runtime settings for the API layer."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("PFF_API_KEY")


def get_api_settings() -> APISettings:
    return APISettings()


async def require_api_key(
    settings: Annotated[APISettings, Depends(get_api_settings)],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate the ``X-API-Key`` header if an API key is configured."""

    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def get_context_manager() -> ContextManager:
    return context_manager
