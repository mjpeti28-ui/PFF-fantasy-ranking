from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.cache import build_query_key, query_cache
from api.dependencies import get_context_manager, require_api_key
from api.models import DataTableResponse
from api.utils import apply_dataframe_query, dataframe_to_records
from context import ContextManager

router = APIRouter(prefix="/sources", tags=["sources"], dependencies=[Depends(require_api_key)])


@router.get("/projections", response_model=DataTableResponse, summary="Query raw projections feed")
async def get_projections(
    manager: ContextManager = Depends(get_context_manager),
    filters: List[str] | None = Query(None),
    sort: str | None = Query(None),
    columns: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> DataTableResponse:
    ctx = manager.get()
    df = ctx.projections_df
    if df is None or df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No projections data available")
    selected_columns = [col.strip() for col in columns.split(",") if col.strip()] if columns else None
    try:
        cache_key = build_query_key(
            "sources:projections",
            context_timestamp=ctx.created_at.isoformat(),
            filters=tuple(filters or []),
            sort=sort,
            columns=tuple(selected_columns or ()),
            limit=limit,
            offset=offset,
        )
        cached = query_cache.get(cache_key)
        if cached is not None:
            return DataTableResponse(**cached)
        window, total = apply_dataframe_query(
            df,
            filters=filters or [],
            sort=sort,
            columns=selected_columns,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    payload = DataTableResponse(
        items=dataframe_to_records(window),
        total=total,
        limit=limit,
        offset=offset,
    )
    query_cache.set(cache_key, payload.model_dump())
    return payload


@router.get("/rankings", response_model=DataTableResponse, summary="Query merged rankings dataset")
async def get_rankings(
    manager: ContextManager = Depends(get_context_manager),
    filters: List[str] | None = Query(None),
    sort: str | None = Query(None),
    columns: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> DataTableResponse:
    ctx = manager.get()
    df = ctx.dataframe
    if df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No rankings data available")
    selected_columns = [col.strip() for col in columns.split(",") if col.strip()] if columns else None
    try:
        cache_key = build_query_key(
            "sources:rankings",
            context_timestamp=ctx.created_at.isoformat(),
            filters=tuple(filters or []),
            sort=sort,
            columns=tuple(selected_columns or ()),
            limit=limit,
            offset=offset,
        )
        cached = query_cache.get(cache_key)
        if cached is not None:
            return DataTableResponse(**cached)
        window, total = apply_dataframe_query(
            df,
            filters=filters or [],
            sort=sort,
            columns=selected_columns,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    payload = DataTableResponse(
        items=dataframe_to_records(window),
        total=total,
        limit=limit,
        offset=offset,
    )
    query_cache.set(cache_key, payload.model_dump())
    return payload
