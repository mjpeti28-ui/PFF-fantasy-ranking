from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from api.models import (
    BenchDetail,
    LeaderboardEntry,
    StarterDetail,
    TeamDetail,
    WaiverCandidate,
)


def coerce_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_team_details(
    team_names: List[str],
    *,
    results: Dict[str, Dict],
    bench_tables: Dict[str, List[Dict]],
    bench_limit: Optional[int] = None,
) -> List[TeamDetail]:
    bench_limit = bench_limit if bench_limit and bench_limit > 0 else None
    payload: List[TeamDetail] = []
    for team_name in team_names:
        res = results.get(team_name, {})
        starters_raw = res.get("starters", [])
        starters_payload: List[StarterDetail] = []
        for starter in starters_raw:
            starters_payload.append(
                StarterDetail(
                    name=str(starter.get("name")),
                    position=starter.get("pos"),
                    csv_name=starter.get("csv"),
                    rank=(int(starter.get("rank")) if starter.get("rank") is not None else None),
                    pos_rank=(int(starter.get("posrank")) if starter.get("posrank") is not None else None),
                    projection=starter.get("proj"),
                    vor=starter.get("vor"),
                )
            )

        bench_rows = bench_tables.get(team_name, [])
        if bench_limit is not None:
            bench_rows = bench_rows[:bench_limit]
        bench_payload: List[BenchDetail] = []
        for bench in bench_rows:
            bench_payload.append(
                BenchDetail(
                    name=str(bench.get("name")),
                    position=bench.get("pos"),
                    rank=(int(bench.get("rank")) if bench.get("rank") is not None else None),
                    pos_rank=(int(bench.get("posrank")) if bench.get("posrank") is not None else None),
                    projection=bench.get("proj"),
                    vor=bench.get("vor"),
                    ovar=bench.get("oVAR"),
                    bench_score=bench.get("BenchScore"),
                )
            )

        payload.append(
            TeamDetail(
                team=team_name,
                starters=starters_payload,
                bench=bench_payload,
                bench_limit=bench_limit,
            )
        )
    return payload


def build_leaderboards_map(raw_lb: Dict[str, List]) -> Dict[str, List[LeaderboardEntry]]:
    leaderboards: Dict[str, List[LeaderboardEntry]] = {}
    for name, entries in raw_lb.items():
        leaderboards[name] = [LeaderboardEntry(team=str(team), value=coerce_float(val)) for team, val in entries]
    return leaderboards


def build_waiver_candidates(
    players: Iterable[dict],
) -> List[WaiverCandidate]:
    candidates: List[WaiverCandidate] = []
    for row in players:
        candidates.append(
            WaiverCandidate(
                name=str(row.get("Name")),
                position=row.get("Position"),
                team=row.get("Team"),
                rank=int(row.get("Rank")) if row.get("Rank") is not None else None,
                pos_rank=int(row.get("PosRank")) if row.get("PosRank") is not None else None,
                proj_points=row.get("ProjPoints"),
                vor=row.get("vor"),
                ovar=row.get("oVAR"),
                bench_score=row.get("BenchScore"),
                need_factor=row.get("needFactor"),
            )
        )
    return candidates


# DataFrame query helpers -------------------------------------------------

VALID_FILTER_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}


def _coerce_value(value: str):
    try:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return float(value)
    except ValueError:
        return value


def apply_dataframe_query(
    df: pd.DataFrame,
    *,
    filters: List[str] | None = None,
    sort: str | None = None,
    columns: List[str] | None = None,
    limit: int,
    offset: int,
) -> tuple[pd.DataFrame, int]:
    working = df.copy()

    if filters:
        for item in filters:
            try:
                column, op, value = item.split(":", 2)
            except ValueError:
                raise ValueError(f"Invalid filter '{item}'. Expected format column:op:value")
            column = column.strip()
            op = op.strip().lower()
            if op not in VALID_FILTER_OPS:
                raise ValueError(f"Unsupported operator '{op}'")
            if column not in working.columns:
                raise ValueError(f"Unknown column '{column}'")
            series = working[column]
            comp_value = _coerce_value(value.strip())

            if op == "contains":
                mask = series.astype(str).str.contains(str(comp_value), case=False, na=False)
            else:
                if pd.api.types.is_numeric_dtype(series):
                    comp_value = float(comp_value)
                if op == "eq":
                    mask = series == comp_value
                elif op == "ne":
                    mask = series != comp_value
                elif op == "gt":
                    mask = series > comp_value
                elif op == "gte":
                    mask = series >= comp_value
                elif op == "lt":
                    mask = series < comp_value
                elif op == "lte":
                    mask = series <= comp_value
                else:
                    raise ValueError(f"Unsupported operator '{op}'")
            working = working[mask]

    total = len(working)

    if sort:
        sort_fields = []
        ascending = []
        for field in sort.split(","):
            field = field.strip()
            if not field:
                continue
            direction = True
            if field.startswith("-"):
                direction = False
                field = field[1:]
            if field not in working.columns:
                raise ValueError(f"Unknown sort column '{field}'")
            sort_fields.append(field)
            ascending.append(direction)
        if sort_fields:
            working = working.sort_values(sort_fields, ascending=ascending, na_position="last")

    if columns:
        cols_present = [col for col in columns if col in working.columns]
        if cols_present:
            working = working[cols_present]

    window = working.iloc[offset : offset + limit]
    window = window.replace({pd.NA: None}).where(pd.notnull(window), None)
    return window, total


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")
