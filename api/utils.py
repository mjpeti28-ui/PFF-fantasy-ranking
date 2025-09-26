from __future__ import annotations

from typing import Dict, List, Optional

from api.models import BenchDetail, LeaderboardEntry, StarterDetail, TeamDetail


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
