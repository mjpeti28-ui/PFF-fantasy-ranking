from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from context import LeagueDataContext

POWER_HISTORY_DIR = Path("history") / "power_rankings"


def build_snapshot_metadata(
    source: str,
    ctx: "LeagueDataContext" | None = None,
    **extras: Any,
) -> Dict[str, Any]:
    """Assemble a metadata dictionary for power ranking snapshots."""

    metadata: Dict[str, Any] = {"source": source}
    if ctx is not None:
        metadata["contextCreatedAt"] = ctx.created_at.isoformat()
        if ctx.espn_league:
            league = ctx.espn_league
            metadata.setdefault("week", league.scoring_period_id)
            metadata["leagueId"] = league.league_id
            metadata["season"] = league.season
    for key, value in extras.items():
        if value is not None:
            metadata[key] = value
    return metadata


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _slugify(parts: Iterable[Any]) -> str:
    tokens: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part)
        if not text:
            continue
        clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)
        clean = clean.strip("-_")
        if clean:
            tokens.append(clean)
    return "_".join(tokens) or "snapshot"


def _normalise_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def save_power_ranking_snapshot(
    *,
    leaderboard: Iterable[tuple[str, Any]] | None,
    starters_totals: Dict[str, Any],
    bench_totals: Dict[str, Any],
    starter_projections: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    rankings_path: Optional[str] = None,
    projections_path: Optional[str] = None,
    supplemental_path: Optional[str] = None,
) -> Optional[Path]:
    """Persist a power ranking snapshot with contextual metadata."""

    snapshot_meta = dict(metadata or {})
    if snapshot_meta.pop("disableSnapshot", False):
        return None

    timestamp = _normalise_timestamp(snapshot_meta.pop("timestamp", datetime.now(timezone.utc)))
    timestamp_iso = timestamp.isoformat()
    week = snapshot_meta.pop("week", None)
    source = snapshot_meta.pop("source", "unknown")

    tags: List[str] = []
    initial_tags = snapshot_meta.pop("tags", None)
    if isinstance(initial_tags, list):
        tags.extend(str(tag) for tag in initial_tags if tag)

    def add_tag(tag: str | None) -> None:
        if tag and tag not in tags:
            tags.append(tag)

    add_tag(f"source:{source}")
    if week is not None:
        add_tag(f"week:{week}")
    if rankings_path:
        add_tag(f"rankings:{Path(rankings_path).name}")
    if projections_path:
        add_tag(f"projections:{Path(projections_path).name}")
    if supplemental_path:
        add_tag(f"supplemental:{Path(supplemental_path).name}")

    settings_payload = settings or {}
    for key, value in sorted(settings_payload.items()):
        simple_value = value
        if isinstance(value, float):
            simple_value = f"{value:.4f}".rstrip("0").rstrip(".")
        add_tag(f"{key}:{simple_value}")

    entries: List[Dict[str, Any]] = []
    leaderboard = list(leaderboard or [])
    for idx, (team, combined_score) in enumerate(leaderboard, start=1):
        entries.append(
            {
                "rank": idx,
                "team": team,
                "combinedScore": _safe_float(combined_score),
                "starterVOR": _safe_float(starters_totals.get(team)),
                "benchScore": _safe_float(bench_totals.get(team)),
                "starterProjection": _safe_float(starter_projections.get(team)),
            }
        )

    snapshot: Dict[str, Any] = {
        "createdAt": timestamp_iso,
        "source": source,
        "week": week,
        "tags": tags,
        "rankingsPath": str(rankings_path) if rankings_path else None,
        "projectionsPath": str(projections_path) if projections_path else None,
        "supplementalPath": str(supplemental_path) if supplemental_path else None,
        "settings": settings_payload,
        "teams": entries,
    }
    if snapshot_meta:
        snapshot["meta"] = snapshot_meta

    POWER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify([week if week is not None else None, source])
    timestamp_label = timestamp.strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp_label}_{slug}.json"
    target_path = POWER_HISTORY_DIR / filename
    counter = 1
    while target_path.exists():
        target_path = POWER_HISTORY_DIR / f"{timestamp_label}_{slug}_{counter}.json"
        counter += 1

    payload = json.dumps(snapshot, indent=2)
    target_path.write_text(payload)
    return target_path


def iter_snapshot_metadata() -> Iterable[Dict[str, Any]]:
    """Yield metadata dictionaries for each saved snapshot without loading team entries.

    Returns shallow dicts containing path, createdAt, week, source, tags, and settings/meta markers.
    """

    if not POWER_HISTORY_DIR.exists():
        return []

    def _iterator() -> Iterable[Dict[str, Any]]:
        for path in sorted(POWER_HISTORY_DIR.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
            except Exception:
                continue
            tags_raw = raw.get("tags")
            tags: List[str]
            if isinstance(tags_raw, list):
                tags = [str(tag) for tag in tags_raw if tag]
            elif isinstance(tags_raw, str):
                tags = [tags_raw]
            else:
                tags = []
            metadata: Dict[str, Any] = {
                "snapshotId": path.stem,
                "path": str(path),
                "file": path.name,
                "createdAt": raw.get("createdAt"),
                "source": raw.get("source"),
                "week": raw.get("week"),
                "tags": tags,
                "rankingsPath": raw.get("rankingsPath"),
                "projectionsPath": raw.get("projectionsPath"),
                "supplementalPath": raw.get("supplementalPath"),
            }
            settings = raw.get("settings") or {}
            for key, value in settings.items():
                metadata[f"setting::{key}"] = value
            extra = raw.get("meta") or {}
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[f"meta::{key}"] = value
            yield metadata

    return list(_iterator())


def iter_snapshot_records(*, include_teams: bool = False) -> List[Dict[str, Any]]:
    if not POWER_HISTORY_DIR.exists():
        return []

    records: List[Dict[str, Any]] = []
    for path in sorted(POWER_HISTORY_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue

        tags_raw = raw.get("tags")
        if isinstance(tags_raw, list):
            tags = [str(tag) for tag in tags_raw if tag]
        elif isinstance(tags_raw, str):
            tags = [tags_raw]
        else:
            tags = []

        record: Dict[str, Any] = {
            "snapshotId": path.stem,
            "path": str(path),
            "file": path.name,
            "createdAt": raw.get("createdAt"),
            "source": raw.get("source"),
            "week": raw.get("week"),
            "tags": tags,
            "rankingsPath": raw.get("rankingsPath"),
            "projectionsPath": raw.get("projectionsPath"),
            "supplementalPath": raw.get("supplementalPath"),
            "settings": raw.get("settings") or {},
            "meta": raw.get("meta") or {},
        }
        if include_teams:
            teams = raw.get("teams")
            if isinstance(teams, list):
                record["teams"] = teams
            else:
                record["teams"] = []
        records.append(record)
    return records
