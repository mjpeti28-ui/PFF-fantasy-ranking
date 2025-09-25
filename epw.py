from __future__ import annotations

import math
import statistics as stats
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import SCORABLE_POS, SLOT_DEFS
from optimizer import dfs_pick
from data import normalize_name, load_rosters

STATS_DIR = Path("Stats")
SOS_DIR = Path("StrengthOfSchedule")
DEFAULT_LAST_WEEK = 17


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_stats() -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    if STATS_DIR.exists():
        for path in STATS_DIR.glob("*.csv"):
            df = _read_csv(path)
            df.columns = df.columns.astype(str)
            lower_cols = [c.lower() for c in df.columns]
            if "fantasypts" in lower_cols or "fantasypts" in ''.join(lower_cols):
                pass
            for _, row in df.iterrows():
                player = str(row.get("player"))
                if not player or player == "nan":
                    continue
                n = normalize_name(player)
                games = float(row.get("games", 0)) or 0.0
                fantasy_pts = float(row.get("fantasyPts", 0)) if "fantasyPts" in row else float(row.get("fantasypts", 0))
                ppg = fantasy_pts / games if games > 0 else None
                mapping.setdefault(n, {})
                mapping[n]["games"] = max(mapping[n].get("games", 0.0), games)
                if ppg:
                    mapping[n]["ppg"] = ppg
                if "ptsPerDb" in row and not math.isnan(row["ptsPerDb"]):
                    mapping[n]["volatility"] = row["ptsPerDb"]
    return mapping


def _extract_weeks(columns: List[str]) -> List[int]:
    weeks = []
    for col in columns:
        try:
            w = int(col)
            weeks.append(w)
        except ValueError:
            continue
    return weeks


def _strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())


def _name_variants(name: str) -> List[str]:
    if not name:
        return []
    base = name.replace(" (IR)", "")
    stripped = _strip_punctuation(base)
    variants = {
        name,
        base,
        stripped,
        normalize_name(name),
        normalize_name(base),
        normalize_name(stripped),
    }
    return [v for v in variants if v]


@lru_cache(maxsize=1)
def load_sos() -> Dict[str, Dict[str, Dict[int, float]]]:
    sos: Dict[str, Dict[str, Dict[int, float]]] = {pos: {} for pos in SCORABLE_POS}
    if not SOS_DIR.exists():
        return sos
    for pos in SCORABLE_POS:
        path = SOS_DIR / f"{pos.lower()}-fantasy-sos.csv"
        if not path.exists():
            continue
        df = _read_csv(path)
        df.columns = df.columns.astype(str)
        weeks = _extract_weeks(df.columns.tolist())
        for _, row in df.iterrows():
            team = str(row.get("Offense") or row.get("offense"))
            if not team or team == "nan":
                continue
            team_dict = sos.setdefault(pos, {}).setdefault(team, {})
            values = [float(row[str(w)]) for w in weeks if str(w) in row and row[str(w)] != ""]
        # compute z-scores per week
        for w in weeks:
            col_name = str(w)
            values = [float(r.get(col_name)) for _, r in df.iterrows() if r.get(col_name) not in ("", None)]
            if not values:
                continue
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
            for _, row in df.iterrows():
                team = str(row.get("Offense") or row.get("offense"))
                if not team or row.get(col_name) in ("", None):
                    continue
                raw = float(row[col_name])
                sos[pos].setdefault(team, {})[w] = (raw - mean) / std
    return sos


def _current_week(stats: Dict[str, dict]) -> int:
    if not stats:
        return 1
    return int(max(info.get("games", 0) for info in stats.values())) + 1


def _player_expected_points(name: str, info: dict, stats_map: Dict[str, dict], remaining_weeks: int) -> float:
    n = normalize_name(name)
    stats = stats_map.get(n, {})
    ppg = stats.get("ppg")
    proj = info.get("ProjPoints")
    if proj is not None and not math.isnan(proj):
        proj = proj / max(1, remaining_weeks)
    else:
        proj_csv = info.get("ProjPointsCsv")
        if proj_csv is not None and not math.isnan(proj_csv):
            proj = proj_csv / max(1, remaining_weeks)
        else:
            proj = None
    if proj is None:
        proj = ppg or 0.0
    if ppg is None:
        return proj
    return 0.7 * proj + 0.3 * ppg


def _team_from_info(info: dict) -> str:
    return info.get("Team") or info.get("Team Abbreviation") or ""


def compute_weekly_team_points(
    team: str,
    roster: Dict[str, List[str]],
    player_info: Dict[str, dict],
    week: int,
    stats_map: Dict[str, dict],
    sos_map: Dict[str, Dict[str, Dict[int, float]]],
    alpha: float,
    remaining_weeks: int,
) -> Tuple[float, float, List[dict], List[dict]]:
    players = []
    for group, names in roster.items():
        for name in names:
            base = name.replace(" (IR)", "")
            info = (
                player_info.get(name)
                or player_info.get(base)
                or player_info.get(normalize_name(name))
                or player_info.get(normalize_name(base))
            )
            if info is None:
                continue
            pos = info.get("Position")
            if pos not in SCORABLE_POS:
                continue
            label = info.get("Name") or base or name
            expected = _player_expected_points(label, info, stats_map, remaining_weeks)
            team_code = _team_from_info(info)
            sos_val = sos_map.get(pos, {}).get(team_code, {}).get(week)
            multiplier = 1.0 + alpha * sos_val if sos_val is not None else 1.0
            adj_points = max(0.0, expected * multiplier)
            players.append((pos, label, adj_points))

    if not players:
        return 0.0, 0.0, [], []

    roster_points = [
        {"name": name, "pos": pos, "rank": -pts, "points": pts}
        for pos, name, pts in players
    ]

    slot_defs = [(slot, eligible) for slot, eligible in SLOT_DEFS]
    eligible = []
    for _, elig in slot_defs:
        cand = [i for i, p in enumerate(roster_points) if p["pos"] in elig]
        eligible.append(cand)
    best = {"sum": math.inf, "assign": None}
    used = set()
    assignment = [-1] * len(slot_defs)

    def dfs(idx, curr_sum):
        nonlocal best
        if curr_sum >= best["sum"]:
            return
        if idx == len(slot_defs):
            best = {"sum": curr_sum, "assign": assignment.copy()}
            return
        s = idx
        options = eligible[s]
        if not options:
            return
        for i in sorted(options, key=lambda j: roster_points[j]["rank"]):
            if i in used:
                continue
            used.add(i)
            assignment[s] = i
            dfs(idx + 1, curr_sum + roster_points[i]["rank"])
            assignment[s] = -1
            used.remove(i)

    dfs(0, 0.0)
    starters = [roster_points[i] for i in best["assign"] if i >= 0]
    bench = [p for idx, p in enumerate(roster_points) if idx not in set(best["assign"]) and p["pos"] in SCORABLE_POS]
    starter_points = sum(p["points"] for p in starters)
    bench_points = sum(p["points"] for p in bench)
    starter_details = [
        {"name": p["name"], "pos": p["pos"], "points": float(p["points"])}
        for p in starters
    ]
    bench_details = [
        {"name": p["name"], "pos": p["pos"], "points": float(p["points"])}
        for p in bench
    ]
    return starter_points, bench_points, starter_details, bench_details


def compute_league_epw(league_data: dict, alpha: float = 0.1) -> Dict[str, Dict]:
    """Return baseline expected weekly starter totals for every team in the league."""
    stats_map = load_stats()
    sos_map = load_sos()
    current_week = _current_week(stats_map)
    remaining_weeks = max(1, DEFAULT_LAST_WEEK - current_week + 1)

    baseline_rosters = league_data.get("rosters", load_rosters())
    player_info: Dict[str, dict] = {}
    for _, row in league_data["df"].iterrows():
        data = row.to_dict()
        name = data.get("Name")
        if not name:
            continue
        for variant in _name_variants(name):
            player_info.setdefault(variant, data)

    weeks = list(range(current_week, DEFAULT_LAST_WEEK + 1))
    results: Dict[str, Dict[str, List[float]]] = {}

    for team, roster in baseline_rosters.items():
        starter_weekly: List[float] = []
        bench_weekly: List[float] = []
        player_totals: Dict[str, float] = {}
        player_weeks: Dict[str, int] = {}
        for week in weeks:
            starter_pts, bench_pts, starter_details, bench_details = compute_weekly_team_points(
                team,
                roster,
                player_info,
                week,
                stats_map,
                sos_map,
                alpha,
                remaining_weeks,
            )
            starter_weekly.append(float(starter_pts))
            bench_weekly.append(float(bench_pts))
            for detail in starter_details:
                nm = detail.get("name")
                if not nm:
                    continue
                player_totals[nm] = player_totals.get(nm, 0.0) + float(detail.get("points", 0.0))
                player_weeks[nm] = player_weeks.get(nm, 0) + 1
            for detail in bench_details:
                nm = detail.get("name")
                if not nm:
                    continue
                player_totals[nm] = player_totals.get(nm, 0.0) + float(detail.get("points", 0.0))
                player_weeks[nm] = player_weeks.get(nm, 0) + 1
        if starter_weekly:
            mean_val = float(stats.mean(starter_weekly))
            total_val = float(sum(starter_weekly))
        else:
            mean_val = 0.0
            total_val = 0.0
        roster_names = set()
        for names in roster.values():
            for raw in names:
                base = raw.replace(" (IR)", "")
                if base:
                    roster_names.add(base)
        player_avg: Dict[str, float] = {}
        for nm in roster_names.union(player_totals.keys()):
            total = player_totals.get(nm, 0.0)
            weeks_cnt = player_weeks.get(nm, 0)
            avg = float(total / weeks_cnt) if weeks_cnt else 0.0
            variants = set(_name_variants(nm))
            for var in variants:
                if not var:
                    continue
                if weeks_cnt > 0 or var not in player_avg:
                    player_avg[var] = avg
        results[team] = {
            "weekly": starter_weekly,
            "bench_weekly": bench_weekly,
            "mean": mean_val,
            "total": total_val,
            "player_totals": player_totals,
            "player_weeks": player_weeks,
            "player_avg": player_avg,
        }

    return {
        "weeks": weeks,
        "teams": results,
        "current_week": current_week,
    }


def evaluate_trade_epw(league_data: dict, team_a: str, team_b: str, send_a: List[str], send_b: List[str], alpha: float = 0.1):
    stats_map = load_stats()
    sos_map = load_sos()
    current_week = _current_week(stats_map)
    remaining_weeks = max(1, DEFAULT_LAST_WEEK - current_week + 1)

    baseline_rosters = league_data.get("rosters", load_rosters())
    player_info = {}
    for _, row in league_data["df"].iterrows():
        data = row.to_dict()
        name = data.get("Name")
        if name:
            for variant in _name_variants(name):
                player_info.setdefault(variant, data)

    weeks = list(range(current_week, DEFAULT_LAST_WEEK + 1))
    baseline = {team_a: [], team_b: []}
    adjusted = {team_a: [], team_b: []}

    for week in weeks:
        for team in (team_a, team_b):
            starter, _, _, _ = compute_weekly_team_points(
                team,
                baseline_rosters[team],
                player_info,
                week,
                stats_map,
                sos_map,
                alpha,
                remaining_weeks,
            )
            baseline[team].append(starter)

    mutated_rosters = {team: {grp: list(names) for grp, names in roster.items()} for team, roster in baseline_rosters.items()}
    for player in send_a:
        for group, names in mutated_rosters[team_a].items():
            if player in names:
                names.remove(player)
                break
    for player in send_b:
        for group, names in mutated_rosters[team_b].items():
            if player in names:
                names.remove(player)
                break
    for player in send_b:
        info = player_info.get(player) or player_info.get(normalize_name(player))
        if info:
            pos = info.get("Position")
            mutated_rosters[team_a].setdefault(pos, []).append(player)
    for player in send_a:
        info = player_info.get(player) or player_info.get(normalize_name(player))
        if info:
            pos = info.get("Position")
            mutated_rosters[team_b].setdefault(pos, []).append(player)

    for week in weeks:
        for team in (team_a, team_b):
            starter, _, _, _ = compute_weekly_team_points(
                team,
                mutated_rosters[team],
                player_info,
                week,
                stats_map,
                sos_map,
                alpha,
                remaining_weeks,
            )
            adjusted[team].append(starter)

    delta = {
        team_a: [a - b for a, b in zip(adjusted[team_a], baseline[team_a])],
        team_b: [a - b for a, b in zip(adjusted[team_b], baseline[team_b])],
    }

    return {
        "weeks": weeks,
        "baseline": baseline,
        "adjusted": adjusted,
        "delta": delta,
    }
