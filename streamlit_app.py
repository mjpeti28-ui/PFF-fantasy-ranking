import copy
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from alias import build_alias_map
from api.utils import build_team_metadata_map
from data import load_rosters, normalize_name
from espn_client import get_last_league_state
import main as main_module
from config import SCORABLE_POS, settings
from trading import (
    TradeFinder,
    TRADE_CONFIG,
    TEAM_A,
    TEAM_B,
    RANKINGS_PATH,
    PROJECTIONS_PATH,
)
from history_tracker import archive_if_changed
from power_snapshots import POWER_HISTORY_DIR
from optimizer import flatten_league_names
from simulation import SimulationStore, generate_random_configs
from epw import evaluate_trade_epw, compute_league_epw
from playoffs import (
    DEFAULT_PLAYOFF_TEAMS,
    compute_playoff_predictions,
)
from context import context_manager


SIM_STORE = SimulationStore()

ARCHIVE_ENTRIES = []
ARCHIVE_ENTRIES.append(("rankings", Path("ROS_week_2_PFF_rankings.csv")))
ARCHIVE_ENTRIES.append(("rankings", Path("PFF_rankings.csv")))
ARCHIVE_ENTRIES.append(("projections", Path(PROJECTIONS_PATH)))

SCHEDULE_DIR = Path("StrengthOfSchedule")
if SCHEDULE_DIR.exists():
    for file in SCHEDULE_DIR.glob("*.csv"):
        ARCHIVE_ENTRIES.append(("sos", file))

STATS_DIR = Path("Stats")
if STATS_DIR.exists():
    for file in STATS_DIR.glob("*.csv"):
        ARCHIVE_ENTRIES.append(("stats", file))

archive_if_changed(ARCHIVE_ENTRIES)


def _get_evaluate_league():
    return main_module.evaluate_league


def evaluate_league_safe(*args, **kwargs):
    evaluate_fn = _get_evaluate_league()
    if "snapshot_metadata" not in kwargs:
        kwargs["snapshot_metadata"] = {
            "source": "streamlit.app",
            "tags": ["streamlit"],
        }
    try:
        return evaluate_fn(*args, **kwargs)
    except TypeError:
        # Reload main module in case cached version lacks newer parameters
        importlib.reload(main_module)
        evaluate_fn = _get_evaluate_league()
        return evaluate_fn(*args, **kwargs)


@st.cache_data(show_spinner=False)
def load_power_history_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not POWER_HISTORY_DIR.exists():
        return pd.DataFrame(), pd.DataFrame()

    snapshot_rows: List[Dict[str, Any]] = []
    team_rows: List[Dict[str, Any]] = []

    for path in sorted(POWER_HISTORY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        snapshot_id = path.stem
        created_raw = data.get("createdAt")
        created_at = pd.to_datetime(created_raw, utc=True, errors="coerce")
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        base_row: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "file": path.name,
            "path": str(path),
            "createdAt": created_at,
            "source": data.get("source"),
            "week": data.get("week"),
            "rankingsPath": data.get("rankingsPath"),
            "projectionsPath": data.get("projectionsPath"),
            "supplementalPath": data.get("supplementalPath"),
            "tags": tuple(tags),
            "tags_str": ", ".join(tags),
        }

        settings = data.get("settings") or {}
        for key, value in settings.items():
            col = f"setting::{key}"
            base_row[col] = value

        meta = data.get("meta") or {}
        if isinstance(meta, dict):
            for key, value in meta.items():
                if key in base_row:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    base_row[f"meta::{key}"] = value

        snapshot_rows.append(base_row)

        teams_payload = data.get("teams") or []
        for team_entry in teams_payload:
            team_row = {
                "snapshot_id": snapshot_id,
                "team": team_entry.get("team"),
                "rank": team_entry.get("rank"),
                "combinedScore": team_entry.get("combinedScore"),
                "starterVOR": team_entry.get("starterVOR"),
                "benchScore": team_entry.get("benchScore"),
                "starterProjection": team_entry.get("starterProjection"),
            }
            team_row.update({
                "createdAt": created_at,
                "source": base_row["source"],
                "week": base_row["week"],
                "tags": base_row["tags"],
                "tags_str": base_row["tags_str"],
                "rankingsPath": base_row.get("rankingsPath"),
                "projectionsPath": base_row.get("projectionsPath"),
                "supplementalPath": base_row.get("supplementalPath"),
            })
            for key, value in base_row.items():
                if key.startswith("setting::") or key.startswith("meta::"):
                    team_row[key] = value
            team_rows.append(team_row)

    snapshots_df = pd.DataFrame(snapshot_rows)
    teams_df = pd.DataFrame(team_rows)
    if not snapshots_df.empty:
        snapshots_df = snapshots_df.sort_values("createdAt", na_position="first").reset_index(drop=True)
    if not teams_df.empty:
        teams_df = teams_df.sort_values(["team", "createdAt"], na_position="first").reset_index(drop=True)
    return snapshots_df, teams_df


def format_player_list(players: List[tuple[str, str]]) -> str:
    if not players:
        return "—"
    return ", ".join(f"{name} ({pos})" for pos, name in players)


def _strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())


def _lookup_epw(player_epw: Optional[Dict[str, float]], name: Optional[str]) -> Optional[float]:
    if not player_epw or not name:
        return None
    variants = [
        name,
        name.replace(" (IR)", ""),
        _strip_punctuation(name),
        _strip_punctuation(name.replace(" (IR)", "")),
        normalize_name(name),
        normalize_name(name.replace(" (IR)", "")),
        normalize_name(_strip_punctuation(name)),
    ]
    seen: set[str] = set()
    for key in variants:
        if not key or key in seen:
            continue
        seen.add(key)
        if key in player_epw:
            return player_epw[key]
    return None


def starters_dataframe(trade: Dict, team: str) -> pd.DataFrame:
    entries = trade["evaluation"]["results"][team]["starters"]
    rows = [
        {
            "Pos": entry.get("pos"),
            "Name": entry.get("name"),
            "Proj": round(entry.get("proj", 0.0), 1) if entry.get("proj") is not None else None,
            "VOR": round(entry.get("vor", 0.0), 1) if entry.get("vor") is not None else None,
            "Rank": entry.get("rank"),
            "PosRank": entry.get("posrank"),
        }
        for entry in entries
    ]
    return pd.DataFrame(rows)


def bench_dataframe(trade: Dict, team: str) -> pd.DataFrame:
    entries = trade["evaluation"]["bench_tables"].get(team, [])
    rows = [
        {
            "Pos": entry.get("pos"),
            "Name": entry.get("name"),
            "Proj": round(entry.get("proj", 0.0), 1) if entry.get("proj") is not None else None,
            "VOR": round(entry.get("vor", 0.0), 2),
            "BenchScore": round(entry.get("BenchScore", 0.0), 2),
        }
        for entry in entries
    ]
    return pd.DataFrame(rows)


def combined_leaderboard_df(trade: Dict) -> pd.DataFrame:
    board = trade["evaluation"]["combined_board"]
    return pd.DataFrame(board, columns=["Team", "Combined Score"])


def leaderboard_df(board: List, value_label: str) -> pd.DataFrame:
    return pd.DataFrame(board, columns=["Team", value_label])


def starters_table(
    results: Dict[str, Dict],
    team: str,
    replacement_points: Dict[str, float],
    player_epw: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    entries = results[team]["starters"]
    rows = []
    for entry in entries:
        pos = entry.get("pos")
        proj = entry.get("proj")
        repl = replacement_points.get(pos, 0.0)
        proj = repl if proj is None else proj
        vor = entry.get("vor")
        if vor is None and proj is not None:
            vor = proj - repl
        name = entry.get("name")
        epw_val = _lookup_epw(player_epw, name)
        row = {
            "Pos": pos,
            "Name": name,
            "Proj": round(proj, 1) if isinstance(proj, (int, float)) else None,
            "VOR": round(vor, 2) if isinstance(vor, (int, float)) else None,
            "Rank": entry.get("rank"),
            "PosRank": entry.get("posrank"),
        }
        if isinstance(epw_val, (int, float)):
            row["EPW"] = round(epw_val, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def bench_table(
    bench_tables: Dict[str, List[Dict]],
    team: str,
    limit: Optional[int] = None,
    player_epw: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    entries = bench_tables.get(team, [])
    if limit is not None and limit > 0:
        entries = entries[:limit]
    rows = []
    for entry in entries:
        proj = entry.get("proj")
        name = entry.get("name")
        epw_val = _lookup_epw(player_epw, name)
        row = {
            "Name": name,
            "Pos": entry.get("pos"),
            "Proj": round(proj, 1) if isinstance(proj, (int, float)) else None,
            "VOR": round(entry.get("vor", 0.0), 2) if entry.get("vor") is not None else None,
            "oVAR": entry.get("oVAR"),
            "BenchScore": entry.get("BenchScore"),
            "Rank": entry.get("rank"),
            "PosRank": entry.get("posrank"),
        }
        if isinstance(epw_val, (int, float)):
            row["EPW"] = round(epw_val, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("BenchScore", ascending=False)
    return df


def zero_sum_group_to_dataframe(group: Dict[str, Any]) -> pd.DataFrame:
    entries = group.get("entries", []) if isinstance(group, dict) else []
    rows = []
    for item in entries:
        share = float(item.get("share", 0.0))
        rows.append(
            {
                "Team": item.get("team"),
                "Value": round(float(item.get("value", 0.0)), 2),
                "Share (%)": round(share * 100.0, 2),
                "Surplus": round(float(item.get("surplus", 0.0)), 2),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Surplus", ascending=False).reset_index(drop=True)
    return df


def zero_sum_team_snapshot(zero_sum: Dict[str, Any], team: str) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for key in ("combined", "starters", "bench"):
        group = zero_sum.get(key, {})
        if not isinstance(group, dict):
            continue
        baseline = float(group.get("baseline", 0.0))
        entries = group.get("entries", [])
        entry = next((item for item in entries if item.get("team") == team), None)
        if entry is None:
            continue
        snapshot[key] = {
            "value": float(entry.get("value", 0.0)),
            "share": float(entry.get("share", 0.0)),
            "surplus": float(entry.get("surplus", 0.0)),
            "baseline": baseline,
        }
    return snapshot


def zero_sum_team_positions(zero_sum: Dict[str, Any], team: str) -> pd.DataFrame:
    positions = zero_sum.get("positions", {}) if isinstance(zero_sum, dict) else {}
    rows = []
    for pos, group in positions.items():
        entries = group.get("entries", []) if isinstance(group, dict) else []
        entry = next((item for item in entries if item.get("team") == team), None)
        if entry is None:
            continue
        share = float(entry.get("share", 0.0))
        rows.append(
            {
                "Position": pos,
                "Value": round(float(entry.get("value", 0.0)), 2),
                "Share (%)": round(share * 100.0, 2),
                "Surplus": round(float(entry.get("surplus", 0.0)), 2),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Surplus", ascending=False).reset_index(drop=True)
    return df


def zero_sum_team_bench_positions(zero_sum: Dict[str, Any], team: str) -> pd.DataFrame:
    bench_positions = zero_sum.get("benchPositions", {}) if isinstance(zero_sum, dict) else {}
    rows = []
    for pos, group in bench_positions.items():
        entries = group.get("entries", []) if isinstance(group, dict) else []
        entry = next((item for item in entries if item.get("team") == team), None)
        if entry is None:
            continue
        share = float(entry.get("share", 0.0))
        rows.append(
            {
                "Position": pos,
                "Bench Score": round(float(entry.get("value", 0.0)), 2),
                "Share (%)": round(share * 100.0, 2),
                "Surplus": round(float(entry.get("surplus", 0.0)), 2),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Surplus", ascending=False).reset_index(drop=True)
    return df


def zero_sum_team_slots(zero_sum: Dict[str, Any], team: str) -> pd.DataFrame:
    slots = zero_sum.get("slots", {}) if isinstance(zero_sum, dict) else {}
    rows = []
    for slot, group in slots.items():
        entries = group.get("entries", []) if isinstance(group, dict) else []
        entry = next((item for item in entries if item.get("team") == team), None)
        if entry is None:
            continue
        share = float(entry.get("share", 0.0))
        rows.append(
            {
                "Slot": slot,
                "Value": round(float(entry.get("value", 0.0)), 2),
                "Share (%)": round(share * 100.0, 2),
                "Surplus": round(float(entry.get("surplus", 0.0)), 2),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Surplus", ascending=False).reset_index(drop=True)
    return df


def summary_table(
    starters_totals: Dict[str, float],
    starter_projections: Dict[str, float],
    bench_totals: Dict[str, float],
    combined_scores: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for team, combined in combined_scores.items():
        rows.append(
            {
                "Team": team,
                "Starter VOR": round(starters_totals.get(team, 0.0), 2),
                "Starter Proj": round(starter_projections.get(team, 0.0), 1),
                "Bench Score": round(bench_totals.get(team, 0.0), 2),
                "Combined Score": round(combined, 3),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Combined Score", ascending=False)
    return df


def _prepare_playoff_dataframe(
    predictions: Dict[str, Any],
    playoff_spots: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    teams_df = pd.DataFrame(predictions.get("teams", []))
    if teams_df.empty:
        empty_series = pd.Series(dtype=float)
        return teams_df, pd.DataFrame(), empty_series, empty_series, empty_series, empty_series

    teams_df = teams_df.fillna(
        {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "games_played": 0,
            "games_remaining": 0,
            "points_for": 0.0,
            "points_against": 0.0,
            "win_pct": 0.0,
            "playoff_probability": 0.0,
            "average_seed": np.nan,
            "median_seed": np.nan,
            "rating": np.nan,
            "rating_z": np.nan,
            "sos_remaining": 0.0,
            "bench_adjust": 1.0,
        }
    )

    for column in ["wins", "losses", "ties", "games_played", "games_remaining"]:
        teams_df[column] = pd.to_numeric(teams_df[column], errors="coerce").fillna(0).astype(int)
    for column in ["points_for", "points_against", "win_pct"]:
        teams_df[column] = pd.to_numeric(teams_df[column], errors="coerce").fillna(0.0)

    playoff_pct = (pd.to_numeric(teams_df["playoff_probability"], errors="coerce").fillna(0.0) * 100.0).round(1)
    avg_seed = pd.to_numeric(teams_df["average_seed"], errors="coerce").round(2)
    median_seed = pd.to_numeric(teams_df["median_seed"], errors="coerce")
    rating = pd.to_numeric(teams_df.get("rating"), errors="coerce")
    rating_z = pd.to_numeric(teams_df.get("rating_z"), errors="coerce").fillna(0.0)
    sos_z = pd.to_numeric(teams_df.get("sos_remaining"), errors="coerce").fillna(0.0)
    bench_adjust = pd.to_numeric(teams_df.get("bench_adjust"), errors="coerce").fillna(1.0)

    display_df = pd.DataFrame(
        {
            "Team": teams_df["team"],
            "Managers": teams_df["managers"],
            "W": teams_df["wins"],
            "L": teams_df["losses"],
            "T": teams_df["ties"],
            "Win %": teams_df["win_pct"].round(3),
            "PF": teams_df["points_for"].round(1),
            "PA": teams_df["points_against"].round(1),
            "Playoff %": playoff_pct,
            "Avg Seed": avg_seed,
            "Median Seed": median_seed,
            "Best Seed": teams_df["best_seed"],
            "Worst Seed": teams_df["worst_seed"],
            "Games Left": teams_df["games_remaining"],
            "Rating": rating.round(2),
            "Rating Z": rating_z.round(2),
            "Remaining SOS (z)": sos_z.round(2),
            "Bench Volatility": bench_adjust.round(2),
        }
    )

    display_df = display_df.sort_values("Playoff %", ascending=False).reset_index(drop=True)
    display_df["Playoff Range"] = display_df.index.map(lambda idx: "Inside" if idx < playoff_spots else "Bubble")

    return teams_df, display_df, playoff_pct, sos_z, rating_z, bench_adjust


def _build_trade_options(
    rosters: Dict[str, Dict[str, List[str]]],
    team: str,
) -> Dict[str, Tuple[str, str]]:
    options: Dict[str, Tuple[str, str]] = {}
    roster = rosters.get(team, {})
    for group, names in roster.items():
        for name in names:
            label = f"{group} — {name}"
            options[label] = (group, name)
    return dict(sorted(options.items()))


def _validate_trade_players(
    rosters: Dict[str, Dict[str, List[str]]],
    team: str,
    pieces: List[Tuple[str, str]],
) -> None:
    roster = rosters.get(team)
    if roster is None:
        raise ValueError(f"Unknown team '{team}'.")
    missing: List[str] = []
    for group, name in pieces:
        if name not in roster.get(group, []):
            missing.append(f"{name} ({group})")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Team '{team}' does not roster: {joined}")


def _mutate_rosters_for_trade(
    rosters: Dict[str, Dict[str, List[str]]],
    team_a: str,
    team_b: str,
    send_a: List[Tuple[str, str]],
    send_b: List[Tuple[str, str]],
) -> Dict[str, Dict[str, List[str]]]:
    finder = TradeFinder(str(RANKINGS_PATH), str(PROJECTIONS_PATH), build_baseline=False)
    finder.rosters = copy.deepcopy(rosters)
    finder.team_targets = {
        team: finder._team_player_count(roster)  # noqa: SLF001
        for team, roster in finder.rosters.items()
    }
    finder.team_groups = {
        team: {grp for grp in roster if grp in SCORABLE_POS}
        for team, roster in finder.rosters.items()
    }
    alias_names = [
        name
        for name in flatten_league_names(finder.rosters)
        if not name.startswith("Replacement ")
    ]
    finder.alias_map = build_alias_map(alias_names, finder.df)
    league_state = get_last_league_state()
    finder.team_metadata = build_team_metadata_map(finder.rosters, league_state) if league_state else {}
    return finder._apply_trade(  # noqa: SLF001
        finder.rosters,
        team_a,
        team_b,
        send_a,
        send_b,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff_entry(base: Dict[str, Any], scenario: Dict[str, Any], key: str) -> Optional[float]:
    base_val = _safe_float(base.get(key))
    scenario_val = _safe_float(scenario.get(key))
    if base_val is None or scenario_val is None:
        return None
    return scenario_val - base_val


def _compute_playoff_delta_df(
    baseline_predictions: Dict[str, Any],
    scenario_predictions: Dict[str, Any],
) -> pd.DataFrame:
    baseline_map = {
        entry.get("team_id"): entry
        for entry in baseline_predictions.get("teams", [])
        if entry.get("team_id")
    }
    scenario_map = {
        entry.get("team_id"): entry
        for entry in scenario_predictions.get("teams", [])
        if entry.get("team_id")
    }

    rows: List[Dict[str, Any]] = []
    for team_id, base_entry in baseline_map.items():
        scenario_entry = scenario_map.get(team_id)
        if scenario_entry is None:
            continue
        team_name = scenario_entry.get("team") or base_entry.get("team") or team_id
        base_prob = _safe_float(base_entry.get("playoff_probability")) or 0.0
        scenario_prob = _safe_float(scenario_entry.get("playoff_probability")) or 0.0

        rows.append(
            {
                "Team": team_name,
                "Δ Playoff %": (scenario_prob - base_prob) * 100.0,
                "Δ Avg Seed": _diff_entry(base_entry, scenario_entry, "average_seed"),
                "Δ Median Seed": _diff_entry(base_entry, scenario_entry, "median_seed"),
                "Δ Best Seed": _diff_entry(base_entry, scenario_entry, "best_seed"),
                "Δ Worst Seed": _diff_entry(base_entry, scenario_entry, "worst_seed"),
                "Δ Rating": _diff_entry(base_entry, scenario_entry, "rating"),
                "Δ Rating Z": _diff_entry(base_entry, scenario_entry, "rating_z"),
                "Δ SOS (z)": _diff_entry(base_entry, scenario_entry, "sos_remaining"),
                "Δ Mean Score": _diff_entry(base_entry, scenario_entry, "mean_score"),
                "Δ Std Dev": _diff_entry(base_entry, scenario_entry, "std_dev"),
                "Δ Bench Volatility": _diff_entry(base_entry, scenario_entry, "bench_adjust"),
            }
        )

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return delta_df

    rounding_map = {
        "Δ Playoff %": 2,
        "Δ Avg Seed": 2,
        "Δ Median Seed": 2,
        "Δ Best Seed": 2,
        "Δ Worst Seed": 2,
        "Δ Rating": 3,
        "Δ Rating Z": 3,
        "Δ SOS (z)": 3,
        "Δ Mean Score": 2,
        "Δ Std Dev": 2,
        "Δ Bench Volatility": 3,
    }
    for column, precision in rounding_map.items():
        if column in delta_df.columns:
            delta_df[column] = pd.to_numeric(delta_df[column], errors="coerce").round(precision)

    delta_df = delta_df.fillna("--")
    delta_df = delta_df.sort_values("Δ Playoff %", ascending=False).reset_index(drop=True)
    return delta_df


def build_available_players(df: pd.DataFrame, rosters: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    existing_norm = set()
    for team_groups in rosters.values():
        for names in team_groups.values():
            for name in names:
                cleaned = name.replace(" (IR)", "")
                existing_norm.add(normalize_name(cleaned))

    available = df[~df["name_norm"].isin(existing_norm)].copy()
    available["Display"] = (
        available.apply(
            lambda row: f"{row['Name']} ({row['Position']} #{row['PosRank']})", axis=1
        )
    )
    return available


def apply_add_drop(
    rosters: Dict[str, Dict[str, List[str]]],
    team: str,
    adds: List[str],
    drops: List[str],
    position_map: Dict[str, str],
) -> Dict[str, Dict[str, List[str]]]:
    mutated = copy.deepcopy(rosters)
    team_groups = mutated.get(team)
    if team_groups is None:
        raise ValueError(f"Team {team} not found in rosters")

    # Remove dropped players
    for drop in drops:
        removed = False
        for group, names in team_groups.items():
            if drop in names:
                names.remove(drop)
                removed = True
                break
        if not removed:
            raise ValueError(f"Player {drop} not found on team {team}")

    # Add new players
    for add in adds:
        pos = position_map.get(add)
        if pos is None:
            continue
        group = pos if pos in SCORABLE_POS else pos
        team_groups.setdefault(group, [])
        if add not in team_groups[group]:
            team_groups[group].append(add)

    return mutated


def run_trade_analysis(
    team_a: str,
    team_b: str,
    config: Dict,
    *,
    must_send_from_a: Optional[List[str]] = None,
    must_receive_from_b: Optional[List[str]] = None,
) -> tuple[TradeFinder, Dict[str, List[Dict]], List[Dict], List[str]]:
    finder = TradeFinder(RANKINGS_PATH, PROJECTIONS_PATH)
    opponents = [team_b] if team_b.lower() != "all" else [team for team in finder.rosters.keys() if team != team_a]

    per_opponent: Dict[str, List[Dict]] = {}
    aggregate: List[Dict] = []
    no_trades: List[str] = []

    for opp in opponents:
        include_send = must_send_from_a or []
        include_receive = must_receive_from_b if team_b.lower() != "all" else []
        results = finder.find_trades(
            team_a,
            opp,
            max_players=config["max_players"],
            player_pool=config["player_pool"],
            top_results=config["top_results"],
            top_bench=config["top_bench"],
            min_gain_a=config["min_gain_a"],
            max_loss_b=config["max_loss_b"],
            prune_margin=config["prune_margin"],
            min_upper_bound=config["min_upper_bound"],
            fairness_mode=config["fairness_mode"],
            fairness_self_bias=config["fairness_self_bias"],
            fairness_penalty_weight=config["fairness_penalty_weight"],
            consolidation_bonus=config["consolidation_bonus"],
            drop_tax_factor=config["drop_tax_factor"],
            acceptance_fairness_weight=config["acceptance_fairness_weight"],
            acceptance_need_weight=config["acceptance_need_weight"],
            acceptance_star_weight=config["acceptance_star_weight"],
            acceptance_need_scale=config["acceptance_need_scale"],
            star_vor_scale=config["star_vor_scale"],
            drop_tax_acceptance_weight=config["drop_tax_acceptance_weight"],
            narrative_on=config["narrative_on"],
            min_acceptance=config["min_acceptance"],
            verbose=False,
            show_progress=False,
            must_send_from_a=include_send,
            must_receive_from_b=include_receive,
        )
        per_opponent[opp] = results or []
        if results:
            aggregate.extend(results)
        else:
            no_trades.append(opp)

    aggregate.sort(
        key=lambda d: (d.get("score", float("-inf")), d.get("acceptance", 0.0)),
        reverse=True,
    )

    return finder, per_opponent, aggregate, no_trades


def render_trade_finder(teams: List[str]) -> None:
    knob_help = {
        "max_players": "Maximum number of players each side can include in a trade (higher = more combinations).",
        "player_pool": "How many of each team's top projected players to consider as trade chips.",
        "top_results": "Number of best trades to show in the summary table and detail view.",
        "fairness_mode": "Scoring objective: 'sum' maximizes total gain, 'weighted' targets a preferred split, 'nash' requires both sides to win.",
        "min_gain_a": "Require Team A to gain at least this much combined score after the trade.",
        "max_loss_b": "Allow Team B to lose at most this much combined score (0 means they cannot lose).",
        "prune_margin": "Slack used in the quick-prune heuristic; larger values keep more borderline trades for evaluation.",
        "min_upper_bound": "Trades whose optimistic total gain is below this number are skipped early.",
        "fairness_self_bias": "In weighted mode, preferred share of the surplus that Team A should keep.",
        "fairness_penalty_weight": "How strongly to penalize surplus splits that deviate from the preferred share.",
        "consolidation_bonus": "Reward when one side trades multiple pieces for a concentrated upgrade (2-for-1, etc.).",
        "drop_tax_factor": "Penalty applied when a team must cut bench players with positive VOR to complete the trade.",
        "min_acceptance": "Filter out trades whose acceptance probability falls below this value.",
        "acceptance_fairness_weight": "Contribution of fairness to the acceptance probability score.",
        "acceptance_need_weight": "Contribution of net team improvement to the acceptance probability score.",
        "acceptance_star_weight": "Weight placed on adding star-level VOR in the acceptance probability score.",
        "acceptance_need_scale": "Scaler for translating combined score deltas into the 'need' component (lower = stricter).",
        "star_vor_scale": "Scaler for how much incoming star VOR contributes to the acceptance score.",
        "drop_tax_acceptance_weight": "Acceptance penalty per point of positive bench VOR a team would need to drop.",
        "narrative_on": "If enabled, display short pitches describing how each team benefits from the trade.",
    }

    default_team_a = TEAM_A if TEAM_A in teams else teams[0]
    team_a = st.sidebar.selectbox(
        "Team A",
        teams,
        index=teams.index(default_team_a),
        key="trade_team_a",
    )

    opponent_options = ["All"] + [team for team in teams if team != team_a]
    default_team_b = TEAM_B if TEAM_B in opponent_options else "All"
    team_b = st.sidebar.selectbox(
        "Opponent",
        opponent_options,
        index=opponent_options.index(default_team_b),
        key="trade_team_b",
    )

    finder = TradeFinder(RANKINGS_PATH, PROJECTIONS_PATH)
    roster_a = finder.rosters.get(team_a, {})
    roster_b = finder.rosters.get(team_b, {})

    roster_choices_a = [name for names in roster_a.values() for name in names]
    roster_choices_b = [name for names in roster_b.values() for name in names]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Scope")
    max_players = st.sidebar.slider(
        "Max players per side", 1, 5, TRADE_CONFIG["max_players"], help=knob_help["max_players"], key="trade_max_players"
    )
    player_pool = st.sidebar.slider(
        "Top players considered", 5, 20, TRADE_CONFIG["player_pool"], help=knob_help["player_pool"], key="trade_player_pool"
    )
    top_results = st.sidebar.slider(
        "Trades to display", 1, 20, TRADE_CONFIG["top_results"], help=knob_help["top_results"], key="trade_top_results"
    )

    must_include_a = st.sidebar.multiselect(
        "Players you must send",
        options=sorted(roster_choices_a),
        help="Selected players are forced to be part of Team A's outgoing package.",
        key="trade_must_send",
    )
    must_receive_b = st.sidebar.multiselect(
        "Players you must receive",
        options=sorted(roster_choices_b),
        help="Trade suggestions will include at least these opponents.",
        key="trade_must_receive",
    )

    if team_b.lower() == "all" and must_receive_b:
        st.sidebar.warning("Must-receive selections are ignored when scanning all opponents.")

    fairness_mode = st.sidebar.selectbox(
        "Fairness mode",
        ["sum", "weighted", "nash"],
        index=["sum", "weighted", "nash"].index(TRADE_CONFIG["fairness_mode"]),
        help=knob_help["fairness_mode"],
        key="trade_fairness_mode",
    )

    st.sidebar.subheader("Fairness Targets")
    min_gain_a = st.sidebar.number_input(
        "Min gain for Team A",
        value=TRADE_CONFIG["min_gain_a"],
        step=0.01,
        format="%.2f",
        help=knob_help["min_gain_a"],
        key="trade_min_gain_a",
    )
    max_loss_b = st.sidebar.number_input(
        "Max loss for Team B",
        value=TRADE_CONFIG["max_loss_b"],
        step=0.01,
        format="%.2f",
        help=knob_help["max_loss_b"],
        key="trade_max_loss_b",
    )
    prune_margin = st.sidebar.number_input(
        "Prune margin",
        value=TRADE_CONFIG["prune_margin"],
        step=0.05,
        format="%.2f",
        help=knob_help["prune_margin"],
        key="trade_prune_margin",
    )
    min_upper_bound = st.sidebar.number_input(
        "Min optimistic delta",
        value=TRADE_CONFIG["min_upper_bound"],
        step=1.0,
        format="%.0f",
        help=knob_help["min_upper_bound"],
        key="trade_min_upper_bound",
    )
    fairness_self_bias = st.sidebar.slider(
        "Target surplus split",
        0.0,
        1.0,
        TRADE_CONFIG["fairness_self_bias"],
        step=0.05,
        help=knob_help["fairness_self_bias"],
        key="trade_self_bias",
    )
    fairness_penalty_weight = st.sidebar.slider(
        "Fairness penalty weight",
        0.0,
        2.0,
        TRADE_CONFIG["fairness_penalty_weight"],
        step=0.05,
        help=knob_help["fairness_penalty_weight"],
        key="trade_penalty_weight",
    )

    st.sidebar.subheader("Depth & Acceptance")
    consolidation_bonus = st.sidebar.slider(
        "Consolidation bonus",
        0.0,
        1.0,
        TRADE_CONFIG["consolidation_bonus"],
        step=0.05,
        help=knob_help["consolidation_bonus"],
        key="trade_consolidation_bonus",
    )
    drop_tax_factor = st.sidebar.slider(
        "Drop tax factor",
        0.0,
        1.0,
        TRADE_CONFIG["drop_tax_factor"],
        step=0.05,
        help=knob_help["drop_tax_factor"],
        key="trade_drop_tax_factor",
    )
    min_acceptance = st.sidebar.slider(
        "Min acceptance",
        0.0,
        1.0,
        TRADE_CONFIG["min_acceptance"],
        step=0.05,
        help=knob_help["min_acceptance"],
        key="trade_min_acceptance",
    )

    acceptance_fairness_weight = st.sidebar.slider(
        "Acceptance weight: fairness",
        0.0,
        1.0,
        TRADE_CONFIG["acceptance_fairness_weight"],
        step=0.05,
        help=knob_help["acceptance_fairness_weight"],
        key="trade_accept_fairness",
    )
    acceptance_need_weight = st.sidebar.slider(
        "Acceptance weight: need",
        0.0,
        1.0,
        TRADE_CONFIG["acceptance_need_weight"],
        step=0.05,
        help=knob_help["acceptance_need_weight"],
        key="trade_accept_need",
    )
    acceptance_star_weight = st.sidebar.slider(
        "Acceptance weight: star power",
        0.0,
        1.0,
        TRADE_CONFIG["acceptance_star_weight"],
        step=0.05,
        help=knob_help["acceptance_star_weight"],
        key="trade_accept_star",
    )
    acceptance_need_scale = st.sidebar.number_input(
        "Need scale",
        value=TRADE_CONFIG["acceptance_need_scale"],
        step=0.1,
        format="%.1f",
        help=knob_help["acceptance_need_scale"],
        key="trade_need_scale",
    )
    star_vor_scale = st.sidebar.number_input(
        "Star VOR scale",
        value=TRADE_CONFIG["star_vor_scale"],
        step=5.0,
        format="%.0f",
        help=knob_help["star_vor_scale"],
        key="trade_star_scale",
    )
    drop_tax_acceptance_weight = st.sidebar.slider(
        "Drop tax acceptance weight",
        0.0,
        0.1,
        TRADE_CONFIG["drop_tax_acceptance_weight"],
        step=0.01,
        help=knob_help["drop_tax_acceptance_weight"],
        key="trade_drop_tax_acceptance",
    )

    st.sidebar.subheader("Display")
    narrative_on = st.sidebar.checkbox(
        "Show trade narratives",
        TRADE_CONFIG["narrative_on"],
        help=knob_help["narrative_on"],
        key="trade_narrative_on",
    )

    run_button = st.sidebar.button("Run trade finder", key="trade_run")

    if not run_button:
        st.info("Adjust the knobs and click **Run trade finder** to evaluate trades.")
        return

    config = {
        "max_players": max_players,
        "player_pool": player_pool,
        "top_results": top_results,
        "top_bench": TRADE_CONFIG["top_bench"],
        "min_gain_a": min_gain_a,
        "max_loss_b": max_loss_b,
        "prune_margin": prune_margin,
        "min_upper_bound": min_upper_bound,
        "fairness_mode": fairness_mode,
        "fairness_self_bias": fairness_self_bias,
        "fairness_penalty_weight": fairness_penalty_weight,
        "consolidation_bonus": consolidation_bonus,
        "drop_tax_factor": drop_tax_factor,
        "acceptance_fairness_weight": acceptance_fairness_weight,
        "acceptance_need_weight": acceptance_need_weight,
        "acceptance_star_weight": acceptance_star_weight,
        "acceptance_need_scale": acceptance_need_scale,
        "star_vor_scale": star_vor_scale,
        "drop_tax_acceptance_weight": drop_tax_acceptance_weight,
        "narrative_on": narrative_on,
        "min_acceptance": min_acceptance,
    }

    with st.spinner("Evaluating trades..."):
        finder, per_opponent, aggregate, no_trades = run_trade_analysis(
            team_a,
            team_b,
            config,
            must_send_from_a=must_include_a,
            must_receive_from_b=must_receive_b,
        )

    if team_b.lower() == "all" and no_trades:
        st.warning(f"No qualifying win-win trades against: {', '.join(no_trades)}")

    if not aggregate:
        st.warning("No trades met the requested thresholds.")
        return

    top_n = min(top_results, len(aggregate))
    rows = []
    for trade in aggregate[:top_n]:
        opp = trade["team_b"]
        delta_a = trade["delta"][team_a]
        delta_b = trade["delta"][opp]
        rows.append(
            {
                "Opponent": opp,
                "Team A Δ": round(delta_a, 3),
                "Opp Δ": round(delta_b, 3),
                "Score": round(trade.get("score", 0.0), 3),
                "Acceptance": round(trade.get("acceptance", 0.0), 2),
                "Split": round(trade.get("fairness_split", 0.0), 2) if trade.get("fairness_split") is not None else "--",
                "Team A gets": format_player_list(trade.get("receive_a", [])),
                "Team A sends": format_player_list(trade.get("send_a", [])),
                "Narrative": trade.get("narrative", {}).get(team_a, ""),
            }
        )

    st.subheader("Top trade opportunities")
    st.dataframe(pd.DataFrame(rows))

    baseline = finder.baseline["combined_scores"] if finder.baseline else {}
    baseline_df = pd.DataFrame(
        sorted(baseline.items(), key=lambda x: x[1], reverse=True),
        columns=["Team", "Combined Score"],
    )
    st.subheader("Current league combined scores")
    st.dataframe(baseline_df)

    st.subheader("Trade details")
    for idx, trade in enumerate(aggregate[:top_n], start=1):
        opp = trade["team_b"]
        delta_a = trade["delta"][team_a]
        delta_b = trade["delta"][opp]
        with st.expander(f"Trade #{idx}: {team_a} ↔ {opp}"):
            st.markdown(
                f"**Score:** {trade.get('score', 0.0):.3f} | **Acceptance:** {trade.get('acceptance', 0.0):.2f} | "
                f"**Split:** {trade.get('fairness_split', '--') if trade.get('fairness_split') is not None else '--'}"
            )
            st.markdown(
                f"**{team_a} Combined:** {baseline.get(team_a, 0.0):.3f} → {trade['combined'][team_a]:.3f} "
                f"({delta_a:+.3f})"
            )
            st.markdown(
                f"**{opp} Combined:** {baseline.get(opp, 0.0):.3f} → {trade['combined'][opp]:.3f} "
                f"({delta_b:+.3f})"
            )
            st.markdown(
                f"**{team_a} receives:** {format_player_list(trade.get('receive_a', []))}<br>"
                f"**{team_a} sends:** {format_player_list(trade.get('send_a', []))}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Pitch — {team_a}:** {trade.get('narrative', {}).get(team_a, '—')}<br>"
                f"**Pitch — {opp}:** {trade.get('narrative', {}).get(opp, '—')}",
                unsafe_allow_html=True,
            )

            drop_tax = trade.get("drop_tax", {})
            if drop_tax:
                st.markdown(
                    f"Drop tax (bench VOR lost): {team_a} {drop_tax.get(team_a, 0.0):.2f} | {opp} {drop_tax.get(opp, 0.0):.2f}"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{team_a} starters**")
                st.dataframe(starters_dataframe(trade, team_a))
                st.markdown(f"**{team_a} bench**")
                st.dataframe(bench_dataframe(trade, team_a))
            with col2:
                st.markdown(f"**{opp} starters**")
                st.dataframe(starters_dataframe(trade, opp))
                st.markdown(f"**{opp} bench**")
                st.dataframe(bench_dataframe(trade, opp))

            st.markdown("**Combined leaderboard (post-trade)**")
            st.dataframe(combined_leaderboard_df(trade))


def render_power_rankings(teams: List[str]) -> None:
    knob_help = {
        "projection_scale_beta": "Boosts separation for players with above-average projections (higher = more projection influence).",
        "combined_weights": "Weight given to starter strength vs. bench depth when building the combined score.",
        "bench_ovar_beta": "How much overall rank advantage (oVAR) matters on the bench relative to projection surplus.",
        "bench_z_threshold": "If bench score variance drops below this value, percentile-based z-scores are used instead of standard z-scores.",
        "bench_percentile_clamp": "Trims extreme percentile z-scores so outliers do not dominate the bench metric.",
        "replacement_skip_pct": "Skip this share of top projected players before picking replacement-level baselines (helps reduce elite skew).",
        "replacement_window": "Average projections over this many players when setting replacement-level points for each position.",
    }

    knob_defaults = settings.snapshot()

    st.sidebar.subheader("Projection scaling")
    projection_scale_beta = st.sidebar.slider(
        "Projection scale beta",
        0.0,
        1.5,
        knob_defaults["projection_scale_beta"],
        step=0.05,
        help=knob_help["projection_scale_beta"],
        key="rank_projection_beta",
    )

    st.sidebar.subheader("Combined weights")
    starters_weight = st.sidebar.slider(
        "Starter weight",
        0.0,
        1.0,
        knob_defaults["combined_starters_weight"],
        step=0.05,
        help=knob_help["combined_weights"],
        key="rank_starter_weight",
    )
    bench_weight = 1.0 - starters_weight
    st.sidebar.caption(f"Bench weight automatically set to {bench_weight:.2f}.")

    st.sidebar.subheader("Bench scoring")
    bench_ovar_beta = st.sidebar.slider(
        "Bench oVAR beta",
        0.0,
        1.0,
        knob_defaults["bench_ovar_beta"],
        step=0.05,
        help=knob_help["bench_ovar_beta"],
        key="rank_bench_beta",
    )
    bench_z_threshold = st.sidebar.slider(
        "Bench z fallback threshold",
        0.0,
        5.0,
        knob_defaults["bench_z_fallback_threshold"],
        step=0.1,
        help=knob_help["bench_z_threshold"],
        key="rank_bench_threshold",
    )
    bench_percentile_clamp = st.sidebar.slider(
        "Bench percentile clamp",
        0.01,
        0.25,
        knob_defaults["bench_percentile_clamp"],
        step=0.01,
        help=knob_help["bench_percentile_clamp"],
        key="rank_bench_clamp",
    )

    st.sidebar.subheader("Replacement levels")
    replacement_skip_pct = st.sidebar.slider(
        "Skip top percent",
        0.0,
        0.3,
        0.1,
        step=0.01,
        help=knob_help["replacement_skip_pct"],
        key="rank_skip_pct",
    )
    replacement_window = st.sidebar.slider(
        "Replacement window",
        1,
        6,
        3,
        help=knob_help["replacement_window"],
        key="rank_replacement_window",
    )
    scarcity_sample_step = st.sidebar.slider(
        "Scarcity sample step",
        0.1,
        2.0,
        0.5,
        step=0.1,
        help="Granularity for sampling positional scarcity curves (lower = more points).",
        key="rank_scarcity_step",
    )

    st.sidebar.subheader("Expected points")
    use_epw = st.sidebar.checkbox(
        "Use EPW ranking",
        value=False,
        help="Rank teams by average expected weekly points (EPW) using the same engine as the trade analyzer.",
        key="rank_use_epw",
    )
    epw_alpha = st.sidebar.slider(
        "EPW SOS adjustment",
        0.0,
        0.5,
        0.1,
        step=0.01,
        help="How much strength-of-schedule adjusts expected points (higher = more schedule impact).",
        key="rank_epw_alpha",
        disabled=not use_epw,
    )
    if not use_epw:
        epw_alpha = 0.1

    run_button = st.sidebar.button("Run power rankings", key="rank_run")

    if not run_button:
        st.info("Adjust the knobs and click **Run power rankings** to refresh the board.")
        return

    with st.spinner("Evaluating power rankings..."):
        league = evaluate_league_safe(
            RANKINGS_PATH,
            projections_path=PROJECTIONS_PATH,
            projection_scale_beta=projection_scale_beta,
            replacement_skip_pct=replacement_skip_pct,
            replacement_window=replacement_window,
            bench_ovar_beta=bench_ovar_beta,
            combined_starters_weight=starters_weight,
            combined_bench_weight=bench_weight,
            bench_z_fallback_threshold=bench_z_threshold,
            bench_percentile_clamp=bench_percentile_clamp,
            scarcity_sample_step=scarcity_sample_step,
        )
        epw_summary = compute_league_epw(league, alpha=epw_alpha) if use_epw else None

    settings_used = league["settings"]
    leaderboards = league["leaderboards"]

    combined_df = leaderboard_df(leaderboards["combined"], "Combined Score")

    if use_epw and epw_summary is not None:
        st.subheader("Expected points leaderboard")
        epw_rows = []
        for team, info in epw_summary["teams"].items():
            epw_rows.append(
                {
                    "Team": team,
                    "Avg Weekly EPW": round(info["mean"], 2),
                    "Total EPW": round(info["total"], 1),
                }
            )
        epw_df = pd.DataFrame(epw_rows).sort_values("Avg Weekly EPW", ascending=False).reset_index(drop=True)
        st.dataframe(epw_df)

        epw_chart_rows = []
        weeks = epw_summary["weeks"]
        for team, info in epw_summary["teams"].items():
            for wk, value in zip(weeks, info["weekly"]):
                epw_chart_rows.append({"Team": team, "Week": wk, "Expected Points": value})
        if epw_chart_rows:
            epw_chart_df = pd.DataFrame(epw_chart_rows)
            epw_chart = (
                alt.Chart(epw_chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Week:Q", axis=alt.Axis(format=".0f")),
                    y=alt.Y("Expected Points:Q"),
                    color="Team:N",
                )
            )
            st.altair_chart(epw_chart, use_container_width=True)

        st.markdown("---")
        st.caption("Combined score leaderboard shown below for comparison with the baseline model.")
        st.subheader("Combined leaderboard (baseline model)")
    else:
        st.subheader("Combined leaderboard")

    st.dataframe(combined_df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Starter leaderboard**")
        st.dataframe(leaderboard_df(leaderboards["starters"], "Starter VOR"))
    with col2:
        st.markdown("**Bench leaderboard**")
        st.dataframe(leaderboard_df(leaderboards["bench"], "Bench Score"))

    st.subheader("Team summary")
    summary_df = summary_table(
        league["starters_totals"],
        league["starter_projections"],
        league["bench_totals"],
        league["combined_scores"],
    )
    st.dataframe(summary_df)

    zero_sum = league.get("zero_sum", {})
    combined_group = zero_sum.get("combined") if isinstance(zero_sum, dict) else None
    if isinstance(combined_group, dict) and combined_group.get("entries"):
        st.subheader("Zero-sum ledger insights")
        weights = combined_group.get("weights", {})
        baseline = combined_group.get("baseline", 0.0)
        starter_w = float(weights.get("starters", 0.0))
        bench_w = float(weights.get("bench", 0.0))
        st.caption(
            "Each ledger shows how much league-wide value a team captures relative to an equal share. "
            "Surplus values are measured against the baseline per-team value "
            f"(combined baseline ≈ {baseline:.2f}). "
            f"Weights: starters={starter_w:.2f}, bench={bench_w:.2f}."
        )

        tab_combined, tab_starters, tab_bench, tab_positions, tab_bench_positions, tab_slots, tab_team = st.tabs(
            [
                "Combined (weighted ledger)",
                "Starters",
                "Bench",
                "Positional lenses",
                "Bench by position",
                "Slot usage",
                "Team drilldown",
            ]
        )

        combined_df = zero_sum_group_to_dataframe(combined_group)
        with tab_combined:
            if combined_df.empty:
                st.info("No zero-sum combined data available.")
            else:
                top_team = combined_df.iloc[0]
                bottom_team = combined_df.iloc[-1]
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(
                    "Highest combined surplus",
                    f"{top_team['Team']}",
                    f"{top_team['Surplus']:+.2f}",
                )
                col_b.metric(
                    "Largest combined deficit",
                    f"{bottom_team['Team']}",
                    f"{bottom_team['Surplus']:+.2f}",
                )
                share_delta = top_team["Share (%)"] - bottom_team["Share (%)"]
                col_c.metric(
                    "Share spread",
                    f"{share_delta:.2f} pts",
                    help="Difference between largest and smallest share of league value.",
                )

                chart_df = combined_df.copy()
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Surplus:Q", title="Surplus vs baseline"),
                        y=alt.Y("Team:N", sort="-x"),
                        color=alt.condition(
                            alt.datum.Surplus >= 0,
                            alt.value("#0E824E"),
                            alt.value("#C72C41"),
                        ),
                        tooltip=["Team", "Value", "Share (%)", "Surplus"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(combined_df)

        starters_group = zero_sum.get("starters", {}) if isinstance(zero_sum, dict) else {}
        with tab_starters:
            starters_df = zero_sum_group_to_dataframe(starters_group)
            if starters_df.empty:
                st.info("No starter ledger data available.")
            else:
                st.markdown("**Starter ledger** — surplus equals projected starter VOR captured beyond the league baseline.")
                st.dataframe(starters_df)

        bench_group = zero_sum.get("bench", {}) if isinstance(zero_sum, dict) else {}
        with tab_bench:
            bench_df = zero_sum_group_to_dataframe(bench_group)
            if bench_df.empty:
                st.info("No bench ledger data available.")
            else:
                st.markdown("**Bench ledger** — highlights how depth hoarding shifts the bench talent pool.")
                st.dataframe(bench_df)

        with tab_positions:
            positions = zero_sum.get("positions", {}) if isinstance(zero_sum, dict) else {}
            if not positions:
                st.info("No positional ledger data available.")
            else:
                pos_choice = st.selectbox(
                    "Position",
                    sorted(positions.keys()),
                    key="zero_sum_position",
                )
                pos_df = zero_sum_group_to_dataframe(positions.get(pos_choice, {}))
                if pos_df.empty:
                    st.info("No ledger entries for the selected position.")
                else:
                    st.markdown(
                        f"**{pos_choice} ledger** — shows which teams are consuming the scarce {pos_choice} starter value."
                    )
                    st.dataframe(pos_df)

        with tab_bench_positions:
            bench_positions = zero_sum.get("benchPositions", {}) if isinstance(zero_sum, dict) else {}
            if not bench_positions:
                st.info("No bench-position ledger data available.")
            else:
                bench_pos_choice = st.selectbox(
                    "Bench position",
                    sorted(bench_positions.keys()),
                    key="zero_sum_bench_position",
                )
                bench_pos_df = zero_sum_group_to_dataframe(bench_positions.get(bench_pos_choice, {}))
                if bench_pos_df.empty:
                    st.info("No bench ledger entries for the selected position.")
                else:
                    st.markdown(f"**Bench {bench_pos_choice} depth** — who controls bench value at this position.")
                    st.dataframe(bench_pos_df)

        with tab_slots:
            slots = zero_sum.get("slots", {}) if isinstance(zero_sum, dict) else {}
            if not slots:
                st.info("No slot-level ledger data available.")
            else:
                slot_choice = st.selectbox(
                    "Slot",
                    sorted(slots.keys()),
                    key="zero_sum_slot_choice",
                )
                slot_df = zero_sum_group_to_dataframe(slots.get(slot_choice, {}))
                if slot_df.empty:
                    st.info("No ledger entries for the selected slot.")
                else:
                    st.markdown(f"**{slot_choice} slot ledger** — value controlled in this lineup slot.")
                    st.dataframe(slot_df)

        with tab_team:
            team_options = combined_df["Team"].tolist() if not combined_df.empty else list(league["results"].keys())
            team_choice = st.selectbox(
                "Team",
                team_options,
                key="zero_sum_team_choice",
            )
            snapshot = zero_sum_team_snapshot(zero_sum, team_choice)
            if not snapshot:
                st.info("No zero-sum snapshot available for the selected team.")
            else:
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                for idx, (label, info) in enumerate(snapshot.items()):
                    col = cols[idx] if idx < len(cols) else cols[-1]
                    share_pct = info["share"] * 100.0
                    col.metric(
                        f"{label.title()} surplus",
                        f"{info['value']:.2f}",
                        f"{info['surplus']:+.2f}",
                        help=f"Share: {share_pct:.2f}% | Baseline: {info['baseline']:.2f}",
                    )
                team_pos_df = zero_sum_team_positions(zero_sum, team_choice)
                if team_pos_df.empty:
                    st.caption("No positional breakdown available for this team.")
                else:
                    st.markdown("**Positional share** — which positions drive this team's leverage.")
                    st.dataframe(team_pos_df)
                    hottest = team_pos_df.iloc[0]
                    st.caption(
                        f"{team_choice} currently holds {hottest['Share (%)']:.2f}% of league {hottest['Position']} surplus "
                        "— a strong bargaining chip in trades."
                    )
                bench_pos_df = zero_sum_team_bench_positions(zero_sum, team_choice)
                if not bench_pos_df.empty:
                    st.markdown("**Bench positional share** — bench depth concentration by position.")
                    st.dataframe(bench_pos_df)
                slot_df = zero_sum_team_slots(zero_sum, team_choice)
                if not slot_df.empty:
                    st.markdown("**Slot dominance** — starter value allocated by lineup slot.")
                    st.dataframe(slot_df)
                analytics_info = zero_sum.get("analytics", {}).get("teams", {}).get(team_choice, {}) if isinstance(zero_sum, dict) else {}
                scarcity_metrics = analytics_info.get("scarcityPressure", {})
                if scarcity_metrics:
                    scarcity_df = pd.DataFrame(
                        [
                            {
                                "Position": pos,
                                "Deficit": round(metric.get("deficit", 0.0), 2),
                                "Pressure": round(metric.get("pressure", 0.0), 3),
                            }
                            for pos, metric in scarcity_metrics.items()
                        ]
                    ).sort_values("Deficit", ascending=False)
                    st.markdown("**Scarcity pressure** — deficits relative to league baseline.")
                    st.dataframe(scarcity_df)
                risk_metrics = analytics_info.get("concentrationRisk", {})
                if risk_metrics:
                    st.markdown("**Concentration risk**")
                    risk_cols = st.columns(2)
                    starter_risk = risk_metrics.get("starterPositions", {})
                    bench_risk = risk_metrics.get("benchPositions", {})
                    if starter_risk:
                        risk_cols[0].metric("Starter Herfindahl", f"{risk_metrics.get('herfindahl', {}).get('starters', 0.0):.3f}")
                    if bench_risk:
                        risk_cols[1].metric("Bench Herfindahl", f"{risk_metrics.get('herfindahl', {}).get('bench', 0.0):.3f}")
                    risk_tables = []
                    if starter_risk:
                        risk_tables.append(("Starter shares", starter_risk))
                    if bench_risk:
                        risk_tables.append(("Bench shares", bench_risk))
                    slot_shares = risk_metrics.get("slotShares", {})
                    if slot_shares:
                        risk_tables.append(("Slot shares", slot_shares))
                    for title, mapping in risk_tables:
                        df = pd.DataFrame(
                            [
                                {"Key": key, "Share (%)": round(val * 100.0, 2)}
                                for key, val in mapping.items()
                            ]
                        ).sort_values("Share (%)", ascending=False)
                        st.markdown(f"**{title}**")
                        st.dataframe(df)
                    flex_share = risk_metrics.get("flexShare")
                    if isinstance(flex_share, (int, float)):
                        st.caption(f"Flex share: {flex_share * 100:.2f}% of league flex surplus.")

    st.subheader("Replacement baselines")
    replacement_df = pd.DataFrame(
        [
            {
                "Position": pos,
                "Baseline Slot": round(league["replacement_targets"].get(pos, 0.0), 2),
                "Replacement Points": round(league["replacement_points"].get(pos, 0.0), 2),
            }
            for pos in sorted(league["replacement_points"].keys())
        ]
    )
    st.dataframe(replacement_df)

    st.subheader("Top team breakdowns")
    combined_lookup = dict(leaderboards["combined"])
    top_teams = [team for team, _ in leaderboards["combined"][:5]]
    for team in top_teams:
        starter_total = league["starters_totals"].get(team, 0.0)
        bench_total = league["bench_totals"].get(team, 0.0)
        starter_proj = league["starter_projections"].get(team, 0.0)
        team_epw_map: Optional[Dict[str, float]] = None
        if use_epw and epw_summary is not None:
            team_epw_map = epw_summary["teams"].get(team, {}).get("player_avg", {})
        with st.expander(f"{team} — {combined_lookup.get(team, 0.0):.3f} combined"):
            st.markdown(
                f"Starter VOR: {starter_total:.2f} | Bench score: {bench_total:.2f} | Starter projection: {starter_proj:.1f}"
            )
            st.markdown("**Starters**")
            st.dataframe(
                starters_table(
                    league["results"],
                    team,
                    league["replacement_points"],
                    player_epw=team_epw_map,
                )
            )
            st.markdown("**Bench (top 5)**")
            st.dataframe(
                bench_table(
                    league["bench_tables"],
                    team,
                    limit=5,
                    player_epw=team_epw_map,
                )
            )

    st.subheader("Positional scarcity curves")
    if league["scarcity_samples"]:
        pos_options = sorted(league["scarcity_samples"].keys())
        selected_pos = st.selectbox(
            "Position",
            pos_options,
            key="scarcity_curve_position",
        )
        curve_df = pd.DataFrame(league["scarcity_samples"].get(selected_pos, []))
        baseline_slot = league["replacement_targets"].get(selected_pos)
        if baseline_slot is not None:
            st.caption(f"Replacement baseline slot ≈ {baseline_slot:.2f}")
        if not curve_df.empty:
            st.dataframe(curve_df.rename(columns={"slot": "Slot", "projection": "Proj Points"}))
        else:
            st.info("No projection data available for the selected position.")
    else:
        st.info("No projection data available to build scarcity curves.")

    with st.expander("Settings used"):
        st.json(settings_used)


def render_history_explorer() -> None:
    snapshots_df, teams_df = load_power_history_tables()
    if st.button("Refresh snapshot cache", type="secondary", help="Reload history files from disk, clearing the cached tables."):
        load_power_history_tables.clear()
        snapshots_df, teams_df = load_power_history_tables()
    if snapshots_df.empty:
        st.info("No power-ranking snapshots have been saved yet. Run a league evaluation to populate history.")
        return

    st.caption("Explore saved power ranking snapshots, filter by metadata, and visualise trends over time.")

    available_sources = sorted([src for src in snapshots_df["source"].dropna().unique()])
    tag_pool = sorted({tag for tags in snapshots_df["tags"] if isinstance(tags, (list, tuple)) for tag in tags})

    preset_options = {
        "All snapshots": {"sources": None, "tags": None},
        "Historical (archived rankings/projections)": {"sources": ["backfill.historical"], "tags": ["backfill", "archived"]},
        "Historical (current rankings/projections)": {"sources": ["backfill.historical-current"], "tags": ["backfill", "current"]},
    }

    def _default_selection(values: Optional[Sequence[str]], available: Sequence[str]) -> List[str]:
        if not available:
            return []
        if not values:
            return list(available)
        filtered = [val for val in values if val in available]
        return filtered if filtered else list(available)

    if "history_applied_preset" not in st.session_state:
        st.session_state["history_applied_preset"] = None
        st.session_state["history_source_selection"] = list(available_sources)
        st.session_state["history_tag_selection"] = []

    preset_label = st.selectbox("Preset view", list(preset_options.keys()), key="history_preset_view")
    preset_config = preset_options[preset_label]
    desired_sources = _default_selection(preset_config["sources"], available_sources)
    desired_tags = _default_selection(preset_config["tags"], tag_pool)

    if st.session_state.get("history_applied_preset") != preset_label:
        st.session_state["history_source_selection"] = desired_sources
        st.session_state["history_tag_selection"] = desired_tags
        st.session_state["history_applied_preset"] = preset_label

    filters = st.columns(3)
    with filters[0]:
        source_selection = st.multiselect(
            "Sources",
            available_sources,
            default=st.session_state.get("history_source_selection", desired_sources),
            key="history_source_selection",
        )

    available_weeks = sorted([int(week) for week in snapshots_df["week"].dropna().unique()])
    with filters[1]:
        week_selection = st.multiselect("Weeks", available_weeks, default=available_weeks) if available_weeks else []

    with filters[2]:
        tag_selection = st.multiselect(
            "Tags",
            tag_pool,
            default=st.session_state.get("history_tag_selection", desired_tags),
            key="history_tag_selection",
        )

    date_range = None
    if not snapshots_df["createdAt"].isna().all():
        min_date = snapshots_df["createdAt"].min()
        max_date = snapshots_df["createdAt"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input("Created date range", (min_date.date(), max_date.date()))

    filtered_snapshots = snapshots_df.copy()
    if source_selection:
        filtered_snapshots = filtered_snapshots[filtered_snapshots["source"].isin(source_selection)]
    if week_selection:
        filtered_snapshots = filtered_snapshots[filtered_snapshots["week"].isin(week_selection)]
    if tag_selection:
        tag_set = set(tag_selection)
        filtered_snapshots = filtered_snapshots[
            filtered_snapshots["tags"].apply(lambda tags: bool(tag_set.intersection(tags)))
        ]
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
        filtered_snapshots = filtered_snapshots[
            (filtered_snapshots["createdAt"] >= start_ts) & (filtered_snapshots["createdAt"] < end_ts)
        ]

    if filtered_snapshots.empty:
        st.warning("No snapshots match the current filter selection.")
        return

    teams_filtered = teams_df[teams_df["snapshot_id"].isin(filtered_snapshots["snapshot_id"])]
    available_teams = sorted([team for team in teams_filtered["team"].dropna().unique()])
    default_teams = available_teams[: min(6, len(available_teams))]
    selected_teams = st.multiselect("Teams", available_teams, default=default_teams)
    if selected_teams:
        teams_filtered = teams_filtered[teams_filtered["team"].isin(selected_teams)]

    setting_cols = [col for col in filtered_snapshots.columns if col.startswith("setting::")]
    setting_name_map = {col.split("::", 1)[1]: col for col in setting_cols}
    color_choice = st.selectbox("Colour series by knob", ["(none)"] + sorted(setting_name_map.keys()))
    color_column = setting_name_map.get(color_choice)

    chart_mode = "Created Timestamp"
    if not teams_filtered["week"].isna().all():
        chart_mode = st.radio("X-axis", ["Created Timestamp", "Week"], horizontal=True)

    summary_cols = [
        "createdAt",
        "source",
        "week",
        "tags_str",
        "rankingsPath",
        "projectionsPath",
        "supplementalPath",
    ]
    summary_table = filtered_snapshots[summary_cols].copy()
    summary_table["createdAt"] = summary_table["createdAt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    metrics = st.columns(3)
    metrics[0].metric("Snapshots", len(filtered_snapshots))
    metrics[1].metric("Teams tracked", len(selected_teams) or len(available_teams))
    metrics[2].metric("Data points", len(teams_filtered))

    st.dataframe(summary_table, use_container_width=True)

    if teams_filtered.empty:
        st.warning("No team-level records after filtering.")
        return

    chart_df = teams_filtered.copy()
    if color_column:
        chart_df["series_colour"] = chart_df[color_column]
    else:
        chart_df["series_colour"] = chart_df["team"]

    x_encoding = (
        alt.X("createdAt:T", title="Snapshot timestamp")
        if chart_mode == "Created Timestamp"
        else alt.X("week:Q", title="Week")
    )
    color_title = color_choice if color_column else "Team"

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=x_encoding,
            y=alt.Y("combinedScore:Q", title="Combined Score"),
            color=alt.Color("series_colour:N", title=color_title),
            detail="team:N",
            tooltip=[
                "team",
                "combinedScore",
                "rank",
                "createdAt:T",
                "week",
                alt.Tooltip("series_colour:N", title=color_title),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    if color_column:
        pivot = (
            chart_df.groupby(["series_colour", "team"])["combinedScore"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "series_colour": color_choice,
                    "mean": "Avg Combined",
                    "std": "Std Dev",
                    "count": "Observations",
                }
            )
        )
        st.dataframe(pivot, use_container_width=True)


def render_playoff_predictor(teams: List[str]) -> None:
    ctx = context_manager.get()
    schedule_df = ctx.schedule if not ctx.schedule.empty else None
    schedule_label = ctx.schedule_source or "schedule.csv"
    if schedule_df is None:
        st.warning("Schedule data not available from ESPN; falling back to local CSV if accessible.")
    st.sidebar.subheader("Playoff simulation settings")

    max_spots = max(2, len(teams))
    playoff_spots = st.sidebar.number_input(
        "Playoff spots",
        min_value=2,
        max_value=max_spots,
        value=min(DEFAULT_PLAYOFF_TEAMS, max_spots),
        step=1,
        key="playoff_spots",
    )

    rosters = load_rosters()
    espn_league = get_last_league_state()

    simulations = st.sidebar.slider(
        "Simulation runs",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500,
        help="More simulations reduce variance but take longer to compute.",
        key="playoff_simulations",
    )

    seed_text = st.sidebar.text_input(
        "Random seed (optional)",
        value="42",
        help="Set a seed for reproducible simulations; leave blank for random.",
        key="playoff_seed",
    ).strip()

    seed: Optional[int]
    if seed_text:
        try:
            seed = int(seed_text)
        except ValueError:
            st.sidebar.warning(f"Invalid seed '{seed_text}'. Using a random seed instead.")
            seed = None
    else:
        seed = None

    refresh = st.sidebar.button("Run playoff simulation", key="playoff_run")

    stored_settings = st.session_state.get("playoff_settings")
    current_settings = {
        "playoff_spots": int(playoff_spots),
        "simulations": int(simulations),
        "seed": seed,
        "espn_scoring_period": espn_league.scoring_period_id if espn_league else None,
        "schedule_label": schedule_label,
    }

    should_run = refresh or stored_settings != current_settings or "playoff_result" not in st.session_state

    if should_run:
        with st.spinner("Simulating playoff odds..."):
            try:
                league_snapshot = evaluate_league_safe(
                    str(RANKINGS_PATH),
                    projections_path=str(PROJECTIONS_PATH),
                )
                predictions = compute_playoff_predictions(
                    schedule_df=schedule_df,
                    schedule_label=schedule_label,
                    rankings_path=str(RANKINGS_PATH),
                    projections_path=str(PROJECTIONS_PATH),
                    num_simulations=int(simulations),
                    playoff_teams=int(playoff_spots),
                    seed=seed,
                    league_snapshot=league_snapshot,
                    espn_league=espn_league,
                )
            except FileNotFoundError as exc:
                st.error(f"Could not load schedule: {exc}")
                return
            except Exception as exc:
                st.error(f"Failed to compute playoff predictions: {exc}")
                return
        st.session_state["playoff_result"] = predictions
        st.session_state["playoff_settings"] = current_settings

    predictions = st.session_state.get("playoff_result")
    if not predictions:
        st.info("Run the playoff simulation to view odds and projections.")
        return

    teams_df, display_df, playoff_pct, sos_z, rating_z, bench_adjust = _prepare_playoff_dataframe(
        predictions,
        playoff_spots,
    )
    if teams_df.empty or display_df.empty:
        st.warning("No team projections were returned from the playoff simulator.")
        return

    st.subheader("Playoff odds snapshot")
    st.dataframe(display_df, use_container_width=True)
    st.caption("Positive SOS z-scores indicate tougher remaining schedules; bench volatility reflects how much depth stabilizes weekly variance (lower is steadier).")

    if not display_df.empty:
        toughest = display_df.sort_values("Remaining SOS (z)", ascending=False).iloc[0]
        easiest = display_df.sort_values("Remaining SOS (z)").iloc[0]
        col_tough, col_easy = st.columns(2)
        col_tough.metric("Toughest remaining slate", toughest["Team"], f"{toughest['Remaining SOS (z)']:+.2f}")
        col_easy.metric("Easiest remaining slate", easiest["Team"], f"{easiest['Remaining SOS (z)']:+.2f}")

    chart_df = display_df[["Team", "Playoff %", "Playoff Range"]]
    sorted_teams = chart_df.sort_values("Playoff %")["Team"].tolist()
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Playoff %:Q", scale=alt.Scale(domain=(0, 100)), title="Playoff probability (%)"),
            y=alt.Y("Team:N", sort=sorted_teams),
            color=alt.Color("Playoff Range:N", scale=alt.Scale(domain=["Inside", "Bubble"], range=["#1f77b4", "#ff7f0e"])),
            tooltip=[
                alt.Tooltip("Team:N"),
                alt.Tooltip("Playoff %:Q", format=".1f"),
                alt.Tooltip("Playoff Range:N"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    sos_chart_df = pd.DataFrame(
        {
            "Team": teams_df["team"],
            "Playoff %": playoff_pct,
            "Remaining SOS (z)": sos_z,
            "Rating Z": rating_z,
        }
    )
    if not sos_chart_df.empty:
        sos_chart = (
            alt.Chart(sos_chart_df)
            .mark_circle(size=120, opacity=0.85)
            .encode(
                x=alt.X(
                    "Remaining SOS (z):Q",
                    title="Remaining schedule difficulty (z-score)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y("Playoff %:Q", title="Playoff probability (%)", scale=alt.Scale(domain=(0, 100))),
                color=alt.Color(
                    "Rating Z:Q",
                    title="Power rating (z)",
                    scale=alt.Scale(scheme="blues"),
                ),
                tooltip=[
                    alt.Tooltip("Team:N"),
                    alt.Tooltip("Playoff %:Q", format=".1f"),
                    alt.Tooltip("Remaining SOS (z):Q", format=".2f"),
                    alt.Tooltip("Rating Z:Q", format=".2f"),
                ],
            )
        )
        st.subheader("Playoff outlook vs. remaining schedule")
        st.altair_chart(sos_chart, use_container_width=True)

    with st.expander("Trade impact simulation"):
        if len(teams) < 2:
            st.info("Need at least two teams to simulate a trade.")
        else:
            trade_col1, trade_col2 = st.columns(2)
            team_a = trade_col1.selectbox(
                "Team A",
                teams,
                key="playoff_trade_team_a",
            )
            opponents = [team for team in teams if team != team_a]
            if not opponents:
                st.warning("Select a different league configuration with more teams.")
            else:
                team_b = trade_col2.selectbox(
                    "Team B",
                    opponents,
                    key="playoff_trade_team_b",
                )
                options_a = _build_trade_options(rosters, team_a)
                options_b = _build_trade_options(rosters, team_b)

                send_a_labels = st.multiselect(
                    "Team A sends",
                    list(options_a.keys()),
                    key="playoff_trade_send_a",
                )
                send_b_labels = st.multiselect(
                    "Team B sends",
                    list(options_b.keys()),
                    key="playoff_trade_send_b",
                )

                simulate_trade = st.button("Simulate trade playoff odds", key="playoff_trade_simulate")
                if simulate_trade:
                    if team_a == team_b:
                        st.error("Select two different teams.")
                    else:
                        send_a = [options_a[label] for label in send_a_labels]
                        send_b = [options_b[label] for label in send_b_labels]
                        try:
                            _validate_trade_players(rosters, team_a, send_a)
                            _validate_trade_players(rosters, team_b, send_b)
                            mutated_rosters = _mutate_rosters_for_trade(rosters, team_a, team_b, send_a, send_b)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            with st.spinner("Simulating trade impact..."):
                                scenario_snapshot = evaluate_league_safe(
                                    str(RANKINGS_PATH),
                                    projections_path=str(PROJECTIONS_PATH),
                                    custom_rosters=mutated_rosters,
                                )
                                scenario_predictions = compute_playoff_predictions(
                                    schedule_df=schedule_df,
                                    schedule_label=schedule_label,
                                    rankings_path=str(RANKINGS_PATH),
                                    projections_path=str(PROJECTIONS_PATH),
                                    num_simulations=int(simulations),
                                    playoff_teams=int(playoff_spots),
                                    seed=seed,
                                    league_snapshot=scenario_snapshot,
                                    espn_league=espn_league,
                                )

                            (
                                scenario_teams_df,
                                scenario_display_df,
                                scenario_playoff_pct,
                                scenario_sos_z,
                                scenario_rating_z,
                                scenario_bench,
                            ) = _prepare_playoff_dataframe(scenario_predictions, playoff_spots)

                            if scenario_display_df.empty:
                                st.warning("Unable to compute playoff odds for the adjusted rosters.")
                            else:
                                st.markdown("**Scenario playoff odds**")
                                st.dataframe(scenario_display_df, use_container_width=True)

                                delta_df = _compute_playoff_delta_df(predictions, scenario_predictions)
                                if delta_df.empty:
                                    st.info("No differences detected between baseline and post-trade projections.")
                                else:
                                    st.markdown("**Delta vs. baseline**")
                                    st.dataframe(delta_df, use_container_width=True)

                                scenario_chart_df = scenario_display_df[["Team", "Playoff %", "Playoff Range"]]
                                sorted_scenario = scenario_chart_df.sort_values("Playoff %")["Team"].tolist()
                                scenario_chart = (
                                    alt.Chart(scenario_chart_df)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("Playoff %:Q", scale=alt.Scale(domain=(0, 100)), title="Playoff probability (%)"),
                                        y=alt.Y("Team:N", sort=sorted_scenario),
                                        color=alt.Color("Playoff Range:N", scale=alt.Scale(domain=["Inside", "Bubble"], range=["#1f77b4", "#ff7f0e"])),
                                        tooltip=[
                                            alt.Tooltip("Team:N"),
                                            alt.Tooltip("Playoff %:Q", format=".1f"),
                                            alt.Tooltip("Playoff Range:N"),
                                        ],
                                    )
                                )
                                st.altair_chart(scenario_chart, use_container_width=True)

                                scenario_sos_chart_df = pd.DataFrame(
                                    {
                                        "Team": scenario_teams_df["team"],
                                        "Playoff %": scenario_playoff_pct,
                                        "Remaining SOS (z)": scenario_sos_z,
                                        "Rating Z": scenario_rating_z,
                                    }
                                )
                                if not scenario_sos_chart_df.empty:
                                    scenario_scatter = (
                                        alt.Chart(scenario_sos_chart_df)
                                        .mark_circle(size=120, opacity=0.85)
                                        .encode(
                                            x=alt.X(
                                                "Remaining SOS (z):Q",
                                                title="Remaining schedule difficulty (z-score)",
                                                scale=alt.Scale(zero=False),
                                            ),
                                            y=alt.Y("Playoff %:Q", title="Playoff probability (%)", scale=alt.Scale(domain=(0, 100))),
                                            color=alt.Color(
                                                "Rating Z:Q",
                                                title="Power rating (z)",
                                                scale=alt.Scale(scheme="blues"),
                                            ),
                                            tooltip=[
                                                alt.Tooltip("Team:N"),
                                                alt.Tooltip("Playoff %:Q", format=".1f"),
                                                alt.Tooltip("Remaining SOS (z):Q", format=".2f"),
                                                alt.Tooltip("Rating Z:Q", format=".2f"),
                                            ],
                                        )
                                    )
                                    st.altair_chart(scenario_scatter, use_container_width=True)

                                focus_delta = delta_df[delta_df["Team"].isin([team_a, team_b])]
                                if not focus_delta.empty:
                                    st.markdown("**Trade teams delta**")
                                    st.dataframe(focus_delta.reset_index(drop=True), use_container_width=True)

    standings = pd.DataFrame(predictions.get("standings", []))
    if not standings.empty:
        st.subheader("Current standings")
        standings_display = standings[
            ["Rank", "Team", "Managers", "Wins", "Losses", "Ties", "WinPct", "PointsFor", "PointsAgainst", "GamesRemaining"]
        ].copy()
        standings_display["WinPct"] = standings_display["WinPct"].round(3)
        standings_display["PointsFor"] = standings_display["PointsFor"].round(1)
        standings_display["PointsAgainst"] = standings_display["PointsAgainst"].round(1)
        st.dataframe(standings_display, use_container_width=True)

    if schedule_df is not None and not schedule_df.empty:
        future_mask = schedule_df["AwayScore"].isna() | schedule_df["HomeScore"].isna()
        future_games = schedule_df.loc[future_mask].copy()
        if not future_games.empty:
            st.subheader("Upcoming schedule")
            future_games = future_games[["Week", "AwayTeam", "HomeTeam", "AwayManagers", "HomeManagers"]]
            future_games = future_games.sort_values(["Week", "AwayTeam"])
            future_games = future_games.rename(
                columns={
                    "AwayTeam": "Away",
                    "HomeTeam": "Home",
                    "AwayManagers": "Away Managers",
                    "HomeManagers": "Home Managers",
                }
            )
            st.dataframe(future_games.reset_index(drop=True), use_container_width=True)

    sim_meta = predictions.get("simulation", {})
    pending_weeks = sim_meta.get("pending_weeks", [])
    st.caption(
        f"Simulations: {sim_meta.get('runs', simulations):,} | Playoff spots: {sim_meta.get('playoff_spots', playoff_spots)} "
        f"| Pending weeks: {', '.join(str(int(week)) for week in pending_weeks) if pending_weeks else 'None'}"
    )


def render_simulation_playground(teams: List[str]) -> None:
    store = SIM_STORE

    current_knobs = settings.snapshot()
    defaults = {
        "projection_scale_beta": current_knobs["projection_scale_beta"],
        "combined_starters_weight": current_knobs["combined_starters_weight"],
        "bench_ovar_beta": current_knobs["bench_ovar_beta"],
        "bench_z_fallback_threshold": current_knobs["bench_z_fallback_threshold"],
        "bench_percentile_clamp": current_knobs["bench_percentile_clamp"],
        "replacement_skip_pct": 0.1,
        "replacement_window": 3,
        "scarcity_sample_step": 0.5,
    }

    st.sidebar.subheader("Simulation controls")
    strategy = st.sidebar.selectbox(
        "Sampling strategy",
        ["Random", "Variance-driven refinement"],
        key="sim_strategy",
    )
    with st.sidebar.form("sim_runner"):
        num_samples = st.number_input(
            "Initial configurations",
            min_value=1,
            max_value=5000,
            value=200,
            step=10,
            help="Number of base configurations to evaluate before any refinement.",
        )
        include_baseline = st.checkbox("Include baseline configuration", value=True)
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1,
            help="Set to 0 for stochastic seed based on time.",
        )

        refinement_rounds = 0
        top_k_std = 3
        if strategy == "Variance-driven refinement":
            refinement_rounds = st.number_input(
                "Refinement rounds",
                min_value=1,
                max_value=2000,
                value=200,
                step=10,
                help="Number of additional configs sampled around the most volatile baselines.",
            )
            top_k_std = st.number_input(
                "Refine top K configs",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="At each refinement step, sample around this many highest-variance configs.",
            )

        ranges: Dict[str, tuple[float, float]] = {}
        base_config: Dict[str, Any] = {}

        def range_control(
            key: str,
            label: str,
            min_val: float,
            max_val: float,
            *,
            step: float,
            is_int: bool = False,
        ) -> None:
            baseline_value = defaults[key]
            base_config.setdefault(key, baseline_value)
            vary = st.checkbox(f"Vary {label}", value=False, key=f"sim_vary_{key}")
            if vary:
                span = max(step, (max_val - min_val) * 0.15)
                lo_default = baseline_value - span
                hi_default = baseline_value + span
                if is_int:
                    lo_default = max(min_val, int(round(lo_default)))
                    hi_default = min(max_val, int(round(hi_default)))
                    if hi_default <= lo_default:
                        hi_default = min(int(max_val), lo_default + max(1, int(step)))
                    lo, hi = st.slider(
                        f"{label} range",
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=(int(lo_default), int(hi_default)),
                        step=int(max(1, int(step))),
                        key=f"sim_range_{key}",
                    )
                    ranges[key] = (float(lo), float(hi))
                else:
                    lo_default = max(min_val, lo_default)
                    hi_default = min(max_val, hi_default)
                    if hi_default <= lo_default:
                        hi_default = min(max_val, lo_default + step)
                    lo, hi = st.slider(
                        f"{label} range",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(lo_default), float(hi_default)),
                        step=step,
                        key=f"sim_range_{key}",
                    )
                    ranges[key] = (float(lo), float(hi))
            else:
                if is_int:
                    value = st.slider(
                        label,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(baseline_value),
                        step=int(max(1, int(step))),
                        key=f"sim_fixed_{key}",
                    )
                    base_config[key] = int(value)
                else:
                    value = st.slider(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(baseline_value),
                        step=step,
                        key=f"sim_fixed_{key}",
                    )
                    base_config[key] = float(value)

        range_control(
            "projection_scale_beta",
            "Projection scale beta",
            min_val=0.0,
            max_val=1.5,
            step=0.05,
        )
        range_control(
            "combined_starters_weight",
            "Starter weight",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        )
        range_control(
            "bench_ovar_beta",
            "Bench oVAR beta",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        )
        range_control(
            "bench_z_fallback_threshold",
            "Bench z fallback threshold",
            min_val=0.0,
            max_val=5.0,
            step=0.1,
        )
        range_control(
            "bench_percentile_clamp",
            "Bench percentile clamp",
            min_val=0.01,
            max_val=0.25,
            step=0.01,
        )
        range_control(
            "replacement_skip_pct",
            "Replacement skip percent",
            min_val=0.0,
            max_val=0.3,
            step=0.01,
        )
        range_control(
            "replacement_window",
            "Replacement window",
            min_val=1,
            max_val=8,
            step=1,
            is_int=True,
        )
        range_control(
            "scarcity_sample_step",
            "Scarcity sample step",
            min_val=0.1,
            max_val=2.0,
            step=0.1,
        )

        tags_input = st.text_input(
            "Tags",
            help="Comma-separated labels to attach to this run (e.g., random, week3).",
        )
        notes = st.text_area("Notes")

        submit_run = st.form_submit_button("Run simulations")

    if submit_run:
        integer_params = {"replacement_window"}
        rng_seed = int(random_seed) if random_seed != 0 else None
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()

        def make_progress_callback(total_expected: int):
            total_expected = max(1, total_expected)

            def _callback(completed: int, total: int = total_expected) -> None:
                completed = min(completed, total)
                fraction = completed / total
                progress_bar.progress(fraction)
                status_placeholder.write(f"Evaluated {completed}/{total} configurations…")

            return _callback

        if strategy == "Random":
            sampled_configs = generate_random_configs(
                num_samples=num_samples,
                base_config=base_config,
                ranges=ranges,
                integer_params=integer_params,
                seed=rng_seed,
            )
            if include_baseline:
                baseline_cfg = base_config.copy()
                sampled_configs.insert(0, baseline_cfg)

            total_cfgs = len(sampled_configs)
            progress_cb = make_progress_callback(total_cfgs)

            record = store.run_batch(
                sampled_configs,
                rankings_path=str(RANKINGS_PATH),
                projections_path=str(PROJECTIONS_PATH),
                tags=tags,
                notes=notes or None,
                extra_metadata={
                    "strategy": "random",
                    "num_samples": num_samples,
                    "varying_params": sorted(ranges.keys()),
                    "seed": rng_seed,
                },
                progress_callback=progress_cb,
            )
        else:
            base_random = generate_random_configs(
                num_samples=num_samples,
                base_config=base_config,
                ranges=ranges,
                integer_params=integer_params,
                seed=rng_seed,
            )
            if include_baseline:
                base_random.insert(0, base_config.copy())

            total_evals = len(base_random) + max(0, refinement_rounds)
            progress_cb = make_progress_callback(total_evals)

            record = store.variance_refine(
                base_configs=base_random,
                ranges=ranges,
                num_refinements=refinement_rounds,
                top_k_by_std=top_k_std,
                integer_params=integer_params,
                rankings_path=str(RANKINGS_PATH),
                projections_path=str(PROJECTIONS_PATH),
                seed=rng_seed,
                run_notes=notes or None,
                tags=tags,
                progress_callback=progress_cb,
            )

        progress_bar.empty()
        status_placeholder.empty()
        st.success(f"Simulation run {record.run_id} completed with {record.num_configs} configs.")
        st.session_state["last_sim_run"] = record.run_id

    st.subheader("Simulation runs")
    runs = store.list_runs()
    if not runs:
        st.info("No simulations executed yet. Configure a run in the sidebar to get started.")
        return

    run_records = [
        {
            "run_id": r.run_id,
            "created_at": r.created_at,
            "num_configs": r.num_configs,
            "tags": ", ".join(r.metadata.get("tags", [])),
            "notes": r.metadata.get("notes"),
        }
        for r in runs
    ]
    run_df = pd.DataFrame(run_records).sort_values("created_at", ascending=False)

    all_tags = sorted({tag.strip() for entry in run_records for tag in (entry["tags"].split(",") if entry["tags"] else []) if tag.strip()})
    tag_filter = st.multiselect("Filter runs by tag", options=all_tags)
    filtered_df = run_df
    if tag_filter:
        mask = filtered_df["tags"].apply(
            lambda cell: any(tag in cell.split(", ") for tag in tag_filter)
            if isinstance(cell, str)
            else False
        )
        filtered_df = filtered_df[mask]

    run_options = filtered_df["run_id"].tolist()
    if not run_options:
        st.warning("No runs match the selected filters.")
        return

    default_run = st.session_state.get("last_sim_run")
    if default_run not in run_options:
        default_run = run_options[0]

    selected_run = st.selectbox(
        "Select a run",
        run_options,
        index=run_options.index(default_run) if default_run in run_options else 0,
        format_func=lambda run_id: f"{run_id} ({filtered_df.loc[filtered_df['run_id'] == run_id, 'created_at'].iloc[0]})",
    )

    run_data = store.load_run(selected_run)
    teams_df = run_data.get("teams")
    configs_df = run_data.get("configs")

    if teams_df is None or teams_df.empty:
        st.warning("Selected run has no team data to display.")
        return

    summary = (
        teams_df.groupby("team")
        .agg(
            combined_mean=("combined_score", "mean"),
            combined_std=("combined_score", "std"),
            combined_rank_mean=("combined_rank", "mean"),
            combined_rank_std=("combined_rank", "std"),
            starter_vor_mean=("starter_vor", "mean"),
            bench_score_mean=("bench_score", "mean"),
        )
        .reset_index()
        .sort_values("combined_rank_mean")
    )
    summary = summary.fillna(0.0)
    st.subheader("Aggregate summary")
    st.dataframe(summary, use_container_width=True)

    default_selection = summary["team"].head(min(6, len(summary))).tolist()
    selected_teams = st.multiselect(
        "Teams to visualize",
        options=summary["team"].tolist(),
        default=default_selection,
    )

    if selected_teams:
        filtered_teams = teams_df[teams_df["team"].isin(selected_teams)]
        st.markdown("**Combined score distribution**")
        box_chart = (
            alt.Chart(filtered_teams)
            .mark_boxplot()
            .encode(
                y=alt.Y("team:N", sort=selected_teams, title="Team"),
                x=alt.X("combined_score:Q", title="Combined Score"),
                color="team:N",
            )
        )
        st.altair_chart(box_chart, use_container_width=True)

    if configs_df is not None and not configs_df.empty:
        param_columns = [col for col in configs_df.columns if col.startswith("input_")]
        if param_columns:
            param_display = {col: col.replace("input_", "") for col in param_columns}
            selected_param = st.selectbox(
                "Parameter to plot",
                param_columns,
                format_func=lambda col: param_display[col],
            )
            merged = teams_df.merge(
                configs_df[["run_id", "config_id", selected_param]],
                on=["run_id", "config_id"],
                how="left",
            )
            if selected_teams:
                merged = merged[merged["team"].isin(selected_teams)]
            scatter_chart = (
                alt.Chart(merged)
                .mark_circle(size=60, opacity=0.6)
                .encode(
                    x=alt.X(f"{selected_param}:Q", title=param_display[selected_param]),
                    y=alt.Y("combined_score:Q", title="Combined Score"),
                    color="team:N",
                    tooltip=["team", "combined_score", "combined_rank", selected_param],
                )
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)

    st.subheader("Download data")
    teams_csv = teams_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download team results",
        teams_csv,
        file_name=f"{selected_run}_teams.csv",
        mime="text/csv",
    )
    if configs_df is not None and not configs_df.empty:
        configs_csv = configs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download config summary",
            configs_csv,
            file_name=f"{selected_run}_configs.csv",
            mime="text/csv",
        )
    replacement_df = run_data.get("replacement_levels")
    if replacement_df is not None and not replacement_df.empty:
        replacement_csv = replacement_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download replacement levels",
            replacement_csv,
            file_name=f"{selected_run}_replacement_levels.csv",
            mime="text/csv",
        )


def render_add_drop_tool(teams: List[str]) -> None:
    refresh_baseline = st.sidebar.button("Recalculate baseline rankings")

    baseline_data = st.session_state.get("add_drop_baseline")
    if baseline_data is None or refresh_baseline:
        with st.spinner("Computing baseline league rankings..."):
            baseline_data = evaluate_league_safe(str(RANKINGS_PATH), projections_path=str(PROJECTIONS_PATH))
        st.session_state["add_drop_baseline"] = baseline_data

    df = baseline_data["df"]
    rosters = load_rosters()
    available_df = build_available_players(df, rosters)

    team = st.selectbox("Team", teams, key="add_drop_team")

    team_groups = rosters.get(team, {})
    drop_choices = {}
    for group, names in team_groups.items():
        for name in names:
            label = f"{group}: {name}"
            drop_choices[label] = name

    drop_selection = st.multiselect(
        "Players to drop",
        options=list(drop_choices.keys()),
        help="Select rostered players to remove (matches current roster entries).",
        key="add_drop_drops",
    )

    position_filter = st.selectbox(
        "Available player position",
        options=["All"] + sorted(SCORABLE_POS),
        key="add_drop_position_filter",
    )
    filtered_available = available_df
    if position_filter != "All":
        filtered_available = filtered_available[filtered_available["Position"] == position_filter]
    filtered_available = filtered_available.sort_values(["Position", "Rank"]).head(200)

    add_choices = {row["Display"]: row["Name"] for _, row in filtered_available.iterrows()}
    add_selection = st.multiselect(
        "Players to add",
        options=list(add_choices.keys()),
        help="Available players from the rankings who are not currently rostered.",
        key="add_drop_adds",
    )

    st.caption("Tip: Select the same number of players to add and drop to keep roster counts balanced.")

    evaluate_button = st.button("Evaluate roster impact", key="add_drop_run")

    if not evaluate_button:
        return

    adds = [add_choices[label] for label in add_selection]
    drops = [drop_choices[label] for label in drop_selection]

    if not adds and not drops:
        st.warning("Select at least one player to add or drop.")
        return

    if len(adds) != len(drops):
        st.warning("For now, please select the same number of players to add and drop to keep roster sizes consistent.")
        return

    position_map = df.set_index("Name")["Position"].to_dict()

    try:
        mutated_rosters = apply_add_drop(rosters, team, adds, drops, position_map)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Re-evaluating league with adjusted roster..."):
        adjusted_league = evaluate_league_safe(
            str(RANKINGS_PATH),
            projections_path=str(PROJECTIONS_PATH),
            custom_rosters=mutated_rosters,
        )

    baseline_combined = baseline_data["combined_scores"]
    adjusted_combined = adjusted_league["combined_scores"]

    comparison_rows = []
    for tm, base_val in baseline_combined.items():
        new_val = adjusted_combined.get(tm, base_val)
        comparison_rows.append(
            {
                "Team": tm,
                "Baseline Combined": round(base_val, 3),
                "New Combined": round(new_val, 3),
                "Delta": round(new_val - base_val, 3),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("Delta", ascending=False)

    st.subheader("League Combined Score Impact")
    st.dataframe(comparison_df, use_container_width=True)

    team_row = comparison_df[comparison_df["Team"] == team]
    if not team_row.empty:
        st.markdown(
            f"**{team} Combined Score:** {team_row.iloc[0]['Baseline Combined']:.3f} → {team_row.iloc[0]['New Combined']:.3f} "
            f"({team_row.iloc[0]['Delta']:+.3f})"
        )

    # Detailed team view
    st.subheader(f"{team} roster details")

    new_results = adjusted_league["results"]
    new_starters = starters_table(new_results, team, adjusted_league["replacement_points"])
    new_bench = bench_table(adjusted_league["bench_tables"], team, limit=None)

    base_starter_vor = baseline_data["starters_totals"].get(team, 0.0)
    new_starter_vor = adjusted_league["starters_totals"].get(team, 0.0)
    base_bench_score = baseline_data["bench_totals"].get(team, 0.0)
    new_bench_score = adjusted_league["bench_totals"].get(team, 0.0)

    st.markdown(
        f"Starter VOR: {base_starter_vor:.2f} → {new_starter_vor:.2f} ({new_starter_vor - base_starter_vor:+.2f})"
    )
    st.markdown(
        f"Bench Score: {base_bench_score:.2f} → {new_bench_score:.2f} ({new_bench_score - base_bench_score:+.2f})"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Adjusted starters**")
        st.dataframe(new_starters)
    with col2:
        st.markdown("**Adjusted bench**")
        st.dataframe(new_bench)

    st.markdown(
        "**Transactions evaluated:** "
        + ", ".join([f"+{player}" for player in adds] + [f"−{player}" for player in drops])
    )


def render_trade_analyzer(teams: List[str]) -> None:
    refresh_baseline = st.sidebar.button("Recalculate baseline", key="trade_analyzer_refresh")

    baseline = st.session_state.get("trade_analyzer_baseline")
    if baseline is None or refresh_baseline:
        with st.spinner("Computing baseline league rankings..."):
            baseline = evaluate_league_safe(
                str(RANKINGS_PATH),
                projections_path=str(PROJECTIONS_PATH),
            )
        st.session_state["trade_analyzer_baseline"] = baseline

    baseline_rosters = baseline.get("rosters", load_rosters())
    player_info = baseline["df"].set_index("Name").to_dict("index")

    col_header = st.columns([1, 0.15, 1])
    default_team_a = TEAM_A if TEAM_A in teams else teams[0]
    default_team_b = TEAM_B if TEAM_B in teams and TEAM_B != default_team_a else next(t for t in teams if t != default_team_a)
    with col_header[0]:
        team_a = st.selectbox("Team A", teams, index=teams.index(default_team_a), key="trade_analyzer_team_a")
    with col_header[2]:
        opponent_options = [team for team in teams if team != team_a]
        team_b = st.selectbox("Team B", opponent_options, index=opponent_options.index(default_team_b) if default_team_b in opponent_options else 0, key="trade_analyzer_team_b")

    def flatten_roster(roster_dict: Dict[str, List[str]]) -> List[tuple[str, str, str]]:
        items = []
        for group, names in roster_dict.items():
            for name in names:
                clean = name.replace(" (IR)", "")
                items.append((group, clean, name))
        return sorted(items, key=lambda x: (x[0], x[1]))

    player_df = baseline["df"].copy()
    info_by_name = player_df.set_index("Name").to_dict("index")
    if "name_norm" in player_df.columns:
        info_by_norm = player_df.set_index("name_norm").to_dict("index")
    else:
        info_by_norm = {}

    league_names: set[str] = set()
    for team_roster in baseline_rosters.values():
        for names in team_roster.values():
            for raw in names:
                clean = raw.replace(" (IR)", "")
                if clean:
                    league_names.add(clean)

    alias_map = build_alias_map(league_names, player_df)

    roster_a = flatten_roster(baseline_rosters.get(team_a, {}))
    roster_b = flatten_roster(baseline_rosters.get(team_b, {}))

    def lookup_info(player: str) -> dict:
        canonical = alias_map.get(player)
        if canonical:
            info = info_by_name.get(canonical)
            if info:
                return info

        info = info_by_name.get(player)
        if info is None:
            norm = normalize_name(player)
            info = info_by_norm.get(norm)
        return info or {}

    def roster_dataframe(roster_list: List[tuple[str, str, str]], checkbox_label: str):
        rows = []
        for grp, name, original in roster_list:
            info = lookup_info(name)
            rows.append(
                {
                    "Select": False,
                    "Player": name,
                    "Pos": grp,
                    "Team": info.get("Team", "-"),
                    "ECR": info.get("PosRank"),
                    "Bye": info.get("ByeWeek"),
                }
            )
        df_display = pd.DataFrame(rows)
        edited = st.data_editor(
            df_display,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(checkbox_label, default=False),
                "Player": st.column_config.TextColumn("Player"),
                "Pos": st.column_config.TextColumn("Pos"),
                "Team": st.column_config.TextColumn("NFL Team"),
                "ECR": st.column_config.NumberColumn("ECR", format="%d"),
                "Bye": st.column_config.NumberColumn("Bye", format="%d"),
            },
            use_container_width=True,
        )
        selected = edited[edited["Select"]][["Player", "Pos"]]
        original_names = []
        for _, row in selected.iterrows():
            player = row["Player"]
            pos = row["Pos"]
            for grp, clean, original in roster_list:
                if grp == pos and clean == player:
                    original_names.append(original)
                    break
        return original_names

    st.markdown("### Proposed trade")
    table_cols = st.columns(2)
    with table_cols[0]:
        st.caption(f"Trade away from {team_a}")
        send_a = roster_dataframe(roster_a, "Trade away")
    with table_cols[1]:
        st.caption(f"Receive from {team_b}")
        send_b = roster_dataframe(roster_b, "Trade for")

    evaluate = st.button("Evaluate trade impact", key="trade_analyzer_eval")
    if not evaluate:
        st.info("Select players from each roster and click **Evaluate trade impact** to see score changes.")
        return

    if not send_a and not send_b:
        st.warning("Select at least one player to exchange.")
        return

    with st.spinner("Simulating trade..."):
        delta = main_module.evaluate_trade_scenario(baseline, team_a, team_b, send_a, send_b)

    combined = baseline["combined_scores"]
    summary_rows = []
    for team in [team_a, team_b]:
        summary_rows.append(
            {
                "Team": team,
                "Baseline": round(combined.get(team, 0.0), 3),
                "Delta": round(delta.get(team, 0.0), 3),
                "New Score": round(combined.get(team, 0.0) + delta.get(team, 0.0), 3),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Combined score impact")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown(
        "**Trade proposal:** "
        + ", ".join([f"{team_a} sends {name}" for name in send_a]
                      + [f"{team_b} sends {name}" for name in send_b])
    )

    metric_cols = st.columns(2)
    for idx, team in enumerate([team_a, team_b]):
        with metric_cols[idx]:
            baseline_val = combined.get(team, 0.0)
            st.metric(team, round(baseline_val + delta.get(team, 0.0), 3), delta.get(team, 0.0))

    epw_data = evaluate_trade_epw(baseline, team_a, team_b, send_a, send_b)
    weeks = epw_data["weeks"]
    chart_rows = []
    for team in [team_a, team_b]:
        for week, base, adj, d in zip(
            weeks,
            epw_data["baseline"][team],
            epw_data["adjusted"][team],
            epw_data["delta"][team],
        ):
            chart_rows.append(
                {
                    "Week": week,
                    "Team": team,
                    "Baseline": base,
                    "Adjusted": adj,
                    "Delta": d,
                }
            )
    epw_df = pd.DataFrame(chart_rows)
    st.subheader("Expected points per week (EPW)")
    delta_summary = (
        epw_df.groupby("Team")["Delta"].agg(["sum", "mean"])
        .rename(columns={"sum": "Total Δ", "mean": "Avg Δ"})
        .reset_index()
    )
    st.dataframe(delta_summary, use_container_width=True)

    epw_chart = (
        alt.Chart(epw_df)
        .transform_fold(["Baseline", "Adjusted"], as_=["Series", "Value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("Week:Q", axis=alt.Axis(format=".0f")),
            y=alt.Y("Value:Q", title="Expected Points"),
            color=alt.Color("Series:N"),
            facet=alt.Facet("Team:N", columns=2),
        )
    )
    st.altair_chart(epw_chart, use_container_width=True)



def main() -> None:
    st.set_page_config(page_title="Fantasy League Explorer", layout="wide")
    st.title("Fantasy League Explorer")
    st.caption("Tune trades or rebuild the power rankings with the same underlying model.")

    rosters = load_rosters()
    teams = sorted(rosters.keys())

    mode_options = [
        "Trade Finder",
        "Power Rankings",
        "History Explorer",
        "Playoff Predictor",
        "Simulation Playground",
        "Add/Drop Impact",
        "Trade Analyzer",
    ]
    mode = st.sidebar.radio("Tool", mode_options, key="mode_select")
    st.sidebar.markdown("---")

    if mode == "Trade Finder":
        st.subheader("Trade Finder")
        render_trade_finder(teams)
    elif mode == "Power Rankings":
        st.subheader("Power Rankings")
        render_power_rankings(teams)
    elif mode == "History Explorer":
        st.subheader("History Explorer")
        render_history_explorer()
    elif mode == "Playoff Predictor":
        st.subheader("Playoff Predictor")
        render_playoff_predictor(teams)
    elif mode == "Simulation Playground":
        st.subheader("Simulation Playground")
        render_simulation_playground(teams)
    elif mode == "Add/Drop Impact":
        st.subheader("Add/Drop Impact")
        render_add_drop_tool(teams)
    else:
        st.subheader("Trade Analyzer")
        render_trade_analyzer(teams)

if __name__ == "__main__":
    main()
