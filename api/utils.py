from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from api.models import (
    BenchDetail,
    LeaderboardEntry,
    StarterDetail,
    TeamDetail,
    WaiverCandidate,
    ZeroSumCombinedGroup,
    ZeroSumEntry,
    ZeroSumGroup,
    ZeroSumHerfindahl,
    ZeroSumResponse,
    ZeroSumAnalytics,
    ZeroSumTeamAnalytics,
    ZeroSumScarcityMetric,
    ZeroSumConcentrationRisk,
    ZeroSumAnalyticsLeague,
    ZeroSumAnalyticsLeagueEntry,
    ZeroSumShift,
    ZeroSumShiftMetrics,
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


def build_zero_sum_payload(raw: Dict[str, Any]) -> ZeroSumResponse:
    def _build_entries(items: List[Dict[str, Any]]) -> List[ZeroSumEntry]:
        entries: List[ZeroSumEntry] = []
        for item in items:
            entries.append(
                ZeroSumEntry(
                    team=str(item.get("team", "")),
                    value=coerce_float(item.get("value")),
                    share=coerce_float(item.get("share")),
                    surplus=coerce_float(item.get("surplus")),
                )
            )
        return entries

    def _build_group(payload: Dict[str, Any], *, combined: bool = False):
        entries = _build_entries(payload.get("entries", []))
        kwargs = {
            "total": coerce_float(payload.get("total")),
            "baseline": coerce_float(payload.get("baseline")),
            "shareSum": coerce_float(payload.get("shareSum")),
            "surplusSum": coerce_float(payload.get("surplusSum")),
            "entries": entries,
        }
        if combined:
            weights_raw = payload.get("weights", {})
            weights = {str(k): coerce_float(v) for k, v in weights_raw.items()}
            kwargs["weights"] = weights
            return ZeroSumCombinedGroup(**kwargs)
        return ZeroSumGroup(**kwargs)

    def _build_group_map(payload: Dict[str, Any]) -> Dict[str, ZeroSumGroup]:
        groups: Dict[str, ZeroSumGroup] = {}
        for key, value in payload.items():
            groups[str(key)] = _build_group(value)
        return groups

    starters_group = _build_group(raw.get("starters", {}))
    bench_group = _build_group(raw.get("bench", {}))
    combined_group = _build_group(raw.get("combined", {}), combined=True)
    positions_payload = _build_group_map(raw.get("positions", {}))
    bench_positions_payload = _build_group_map(raw.get("benchPositions", {}))
    slots_payload = _build_group_map(raw.get("slots", {}))
    flex_raw = raw.get("flex")
    flex_group = _build_group(flex_raw) if flex_raw else None

    analytics_raw = raw.get("analytics", {})
    team_analytics: Dict[str, ZeroSumTeamAnalytics] = {}
    for team, data in analytics_raw.get("teams", {}).items():
        scarcity_raw = data.get("scarcityPressure", {})
        scarcity_payload = {
            str(pos): ZeroSumScarcityMetric(
                deficit=coerce_float(metric.get("deficit")),
                pressure=coerce_float(metric.get("pressure")),
            )
            for pos, metric in scarcity_raw.items()
        }
        risk_raw = data.get("concentrationRisk", {})
        risk = ZeroSumConcentrationRisk(
            starterPositions={str(k): coerce_float(v) for k, v in risk_raw.get("starterPositions", {}).items()},
            benchPositions={str(k): coerce_float(v) for k, v in risk_raw.get("benchPositions", {}).items()},
            slotShares={str(k): coerce_float(v) for k, v in risk_raw.get("slotShares", {}).items()},
            flexShare=coerce_float(risk_raw.get("flexShare")),
            herfindahl=ZeroSumHerfindahl(
                starters=coerce_float(risk_raw.get("herfindahl", {}).get("starters", 0.0)),
                bench=coerce_float(risk_raw.get("herfindahl", {}).get("bench", 0.0)),
                slots=coerce_float(risk_raw.get("herfindahl", {}).get("slots", 0.0)),
            ),
        )
        team_analytics[str(team)] = ZeroSumTeamAnalytics(
            scarcityPressure=scarcity_payload,
            concentrationRisk=risk,
        )

    league_raw = analytics_raw.get("league", {})
    league_analytics = ZeroSumAnalyticsLeague(
        highPressurePositions=[
            ZeroSumAnalyticsLeagueEntry(
                position=str(entry.get("position")),
                aggregateDeficit=coerce_float(entry.get("aggregateDeficit")),
            )
            for entry in league_raw.get("highPressurePositions", [])
        ]
    )

    analytics_payload = ZeroSumAnalytics(
        teams=team_analytics,
        league=league_analytics,
    )

    return ZeroSumResponse(
        teamCount=int(raw.get("teamCount", 0)),
        starters=starters_group,
        bench=bench_group,
        combined=combined_group,
        positions=positions_payload,
        benchPositions=bench_positions_payload,
        slots=slots_payload,
        flex=flex_group,
        analytics=analytics_payload,
    )


def build_zero_sum_shift(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    focus_teams: Optional[List[str]] = None,
) -> ZeroSumShift:
    def _entries_map(group: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        entries = group.get("entries", []) if isinstance(group, dict) else []
        return {str(entry.get("team")): entry for entry in entries}

    def _compute_group(
        before_group: Dict[str, Any],
        after_group: Dict[str, Any],
        teams: Optional[List[str]] = None,
    ) -> tuple[Dict[str, ZeroSumShiftMetrics], Dict[str, Dict[str, float]]]:
        before_map = _entries_map(before_group)
        after_map = _entries_map(after_group)
        participants = teams or sorted(set(before_map) | set(after_map))
        raw_shift: Dict[str, Dict[str, float]] = {}
        metrics: Dict[str, ZeroSumShiftMetrics] = {}
        for team in participants:
            before_entry = before_map.get(team, {})
            after_entry = after_map.get(team, {})
            value_delta = coerce_float(after_entry.get("value")) - coerce_float(before_entry.get("value"))
            share_delta = coerce_float(after_entry.get("share")) - coerce_float(before_entry.get("share"))
            surplus_delta = coerce_float(after_entry.get("surplus")) - coerce_float(before_entry.get("surplus"))
            raw_shift[team] = {
                "valueDelta": value_delta,
                "shareDelta": share_delta,
                "surplusDelta": surplus_delta,
            }
            metrics[team] = ZeroSumShiftMetrics(
                value_delta=value_delta,
                share_delta=share_delta,
                surplus_delta=surplus_delta,
            )
        return metrics, raw_shift

    def _compute_group_map(
        before_groups: Dict[str, Any],
        after_groups: Dict[str, Any],
        teams: Optional[List[str]] = None,
    ) -> tuple[Dict[str, Dict[str, ZeroSumShiftMetrics]], Dict[str, Dict[str, Dict[str, float]]]]:
        metrics_map: Dict[str, Dict[str, ZeroSumShiftMetrics]] = {}
        raw_map: Dict[str, Dict[str, Dict[str, float]]] = {}
        keys = set(before_groups.keys()) | set(after_groups.keys())
        for key in keys:
            before_group = before_groups.get(key, {})
            after_group = after_groups.get(key, {})
            metrics, raw = _compute_group(before_group, after_group, teams=teams)
            metrics_map[str(key)] = metrics
            raw_map[str(key)] = raw
        return metrics_map, raw_map

    focus = focus_teams
    combined_metrics, combined_raw = _compute_group(
        before.get("combined", {}),
        after.get("combined", {}),
        teams=focus,
    )
    starters_metrics, starters_raw = _compute_group(
        before.get("starters", {}),
        after.get("starters", {}),
        teams=focus,
    )
    bench_metrics, bench_raw = _compute_group(
        before.get("bench", {}),
        after.get("bench", {}),
        teams=focus,
    )
    positions_metrics, positions_raw = _compute_group_map(
        before.get("positions", {}),
        after.get("positions", {}),
        teams=focus,
    )
    bench_positions_metrics, bench_positions_raw = _compute_group_map(
        before.get("benchPositions", {}),
        after.get("benchPositions", {}),
        teams=focus,
    )
    slots_metrics, slots_raw = _compute_group_map(
        before.get("slots", {}),
        after.get("slots", {}),
        teams=focus,
    )
    flex_metrics, flex_raw = _compute_group(
        before.get("flex", {}),
        after.get("flex", {}),
        teams=focus,
    )

    def _summaries(raw_data: Dict[str, Dict[str, float]], *, top_n: int = 3) -> Dict[str, List[Dict[str, float]]]:
        winners = sorted(
            (
                {
                    "team": team,
                    "valueDelta": vals["valueDelta"],
                    "shareDelta": vals["shareDelta"],
                    "surplusDelta": vals["surplusDelta"],
                }
                for team, vals in raw_data.items()
            ),
            key=lambda item: item["surplusDelta"],
            reverse=True,
        )
        losers = [entry for entry in winners if entry["surplusDelta"] < 0]
        winners = [entry for entry in winners if entry["surplusDelta"] > 0]
        return {
            "winners": winners[:top_n],
            "losers": losers[:top_n],
        }

    summary: Dict[str, Any] = {
        "combined": _summaries(combined_raw),
        "starters": _summaries(starters_raw),
        "bench": _summaries(bench_raw),
        "positions": {
            pos: _summaries(raw)
            for pos, raw in positions_raw.items()
        },
    }

    return ZeroSumShift(
        combined=combined_metrics,
        starters=starters_metrics,
        bench=bench_metrics,
        positions=positions_metrics,
        benchPositions=bench_positions_metrics,
        slots=slots_metrics,
        flex=flex_metrics,
        summary=summary,
    )


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
