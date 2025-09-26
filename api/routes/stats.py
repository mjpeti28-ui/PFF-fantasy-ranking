from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.cache import build_query_key, query_cache
from api.dependencies import get_context_manager, require_api_key
from api.models import DataTableResponse
from api.utils import apply_dataframe_query, dataframe_to_records
from context import ContextManager

router = APIRouter(prefix="/stats", tags=["stats"], dependencies=[Depends(require_api_key)])


@router.get("/{dataset}", response_model=DataTableResponse, summary="Query statistical datasets")
async def get_stats_dataset(
    dataset: str,
    manager: ContextManager = Depends(get_context_manager),
    filters: List[str] | None = Query(None, description="Filter expressions column:op:value"),
    sort: str | None = Query(None, description="Comma-separated sort fields, prefix with - for descending"),
    columns: str | None = Query(None, description="Comma-separated list of columns to include"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DataTableResponse:
    ctx = manager.get()
    dataset_key = dataset.lower()
    stats_df = ctx.stats_data.get(dataset_key)
    if stats_df is None or stats_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown dataset '{dataset}'")

    try:
        selected_columns = [col.strip() for col in columns.split(",") if col.strip()] if columns else None
        cache_key = build_query_key(
            f"stats:{dataset_key}",
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
            stats_df,
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
