from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import require_api_key
from api.models import PowerSnapshotRecord, PowerSnapshotResponse
from power_archetypes import resolve_archetype_filters
from power_snapshots import iter_snapshot_records

router = APIRouter(prefix="/history", tags=["history"], dependencies=[Depends(require_api_key)])


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _filter_records(
    records: Iterable[Dict[str, Any]],
    *,
    sources: List[str],
    tags: List[str],
    weeks: List[int],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> List[Dict[str, Any]]:
    sources_set = {src.lower() for src in sources if src}
    tags_set = {tag.lower() for tag in tags if tag}
    weeks_set = {int(week) for week in weeks if week}

    filtered: List[Dict[str, Any]] = []
    for record in records:
        source_value = (record.get("source") or "").lower()
        if sources_set and source_value not in sources_set:
            continue

        record_week = record.get("week")
        if weeks_set and record_week not in weeks_set:
            continue

        record_tags = [str(tag).lower() for tag in record.get("tags", []) if tag]
        if tags_set and not tags_set.issubset(record_tags):
            continue

        created_at = _parse_datetime(record.get("createdAt"))
        if start_date and (created_at is None or created_at < start_date):
            continue
        if end_date and (created_at is None or created_at > end_date):
            continue

        filtered.append(record)
    return filtered


def _convert_record(record: Dict[str, Any], include_teams: bool) -> PowerSnapshotRecord:
    created_at = _parse_datetime(record.get("createdAt"))
    payload: Dict[str, Any] = {
        "snapshotId": record.get("snapshotId"),
        "createdAt": created_at,
        "source": record.get("source"),
        "week": record.get("week"),
        "tags": record.get("tags") or [],
        "rankingsPath": record.get("rankingsPath"),
        "projectionsPath": record.get("projectionsPath"),
        "supplementalPath": record.get("supplementalPath"),
        "file": record.get("file"),
        "path": record.get("path"),
        "settings": record.get("settings") or {},
        "meta": record.get("meta") or {},
    }
    if include_teams:
        payload["teams"] = record.get("teams") or []
    return PowerSnapshotRecord(**payload)


@router.get(
    "/power-rankings",
    response_model=PowerSnapshotResponse,
    summary="List saved power ranking snapshots",
)
async def list_power_ranking_snapshots(
    sources: List[str] = Query(default=[], description="Filter by source label."),
    tags: List[str] = Query(default=[], description="Require these tags (all must be present)."),
    weeks: List[int] = Query(default=[], description="Limit to specific weeks."),
    start_date: Optional[datetime] = Query(default=None, alias="startDate", description="Earliest snapshot timestamp (inclusive)."),
    end_date: Optional[datetime] = Query(default=None, alias="endDate", description="Latest snapshot timestamp (inclusive)."),
    include_teams: bool = Query(default=False, alias="includeTeams", description="Include team-level snapshot data."),
    archetypes: List[str] = Query(
        default=[],
        description="One or more archetype keys to expand into matching sources/tags.",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PowerSnapshotResponse:
    try:
        records = iter_snapshot_records(include_teams=include_teams)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    archetype_sources, archetype_tags = resolve_archetype_filters(archetypes)
    if archetype_sources:
        sources = list(dict.fromkeys([*archetype_sources, *sources]))
    if archetype_tags:
        tags = list(dict.fromkeys([*archetype_tags, *tags]))

    filtered = _filter_records(
        records,
        sources=sources,
        tags=tags,
        weeks=weeks,
        start_date=start_date,
        end_date=end_date,
    )

    total = len(filtered)
    sliced = filtered[offset : offset + limit]

    payload = [
        _convert_record(record, include_teams=include_teams)
        for record in sliced
    ]
    return PowerSnapshotResponse(
        total=total,
        limit=limit,
        offset=offset,
        snapshots=payload,
    )
