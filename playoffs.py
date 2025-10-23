"""Playoff projection utilities for league simulations.

This module loads the league schedule, derives current standings, estimates
per-team scoring distributions, and runs Monte Carlo simulations to compute
playoff odds. The core entry point, :func:`compute_playoff_predictions`,
returns structured data that can be exposed via an API endpoint or reused in
the Streamlit application.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from main import DEFAULT_PROJECTIONS, DEFAULT_RANKINGS, evaluate_league
from rosters import Fantasy_Rosters
from espn_client import ESPNLeagueData

DEFAULT_SCHEDULE_PATH = Path("schedule.csv")
DEFAULT_PLAYOFF_TEAMS = 8
MIN_STD_DEV = 8.0
STD_FRACTION = 0.15
DEFAULT_MEAN_GUESS = 120.0
POWER_RATING_POINT_SCALE = 8.0
LOGIT_SCALE = 1.35
TIE_PROBABILITY = 0.01
MARGIN_PER_RATING_Z = 10.0
BENCH_VARIANCE_FLOOR = 0.6
BENCH_VARIANCE_CAP = 1.35


@dataclass
class TeamInfo:
    team_id: str
    display_name: str
    managers: Optional[str] = None
    roster_key: Optional[str] = None


@dataclass
class TeamRecord:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    games_played: int = 0

    def clone(self) -> "TeamRecord":
        return TeamRecord(
            wins=self.wins,
            losses=self.losses,
            ties=self.ties,
            points_for=self.points_for,
            points_against=self.points_against,
            games_played=self.games_played,
        )

    @property
    def win_pct(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.games_played


@dataclass
class TeamMetrics:
    mean_score: float
    std_dev: float
    rating: float
    rating_z: float
    sos_remaining: float = 0.0
    bench_adjust: float = 1.0
    source: Dict[str, Any] = field(default_factory=dict)


def load_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> pd.DataFrame:
    """Load the league schedule CSV and normalize dtypes."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [
        "Week",
        "AwayTeam",
        "AwayManagers",
        "AwayScore",
        "HomeScore",
        "HomeManagers",
        "HomeTeam",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Schedule file missing required columns: {missing}")

    # Normalize whitespace and types for string columns we care about.
    str_cols = [
        "AwayTeam",
        "HomeTeam",
        "AwayManagers",
        "HomeManagers",
        "AwayRecord",
        "HomeRecord",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["AwayScore"] = pd.to_numeric(df["AwayScore"], errors="coerce")
    df["HomeScore"] = pd.to_numeric(df["HomeScore"], errors="coerce")

    # Ensure schedule is ordered by week (and fallback lexicographically for stability).
    df = df.sort_values(["Week", "AwayTeam", "HomeTeam"]).reset_index(drop=True)
    return df


def espn_schedule_to_dataframe(league: ESPNLeagueData) -> pd.DataFrame:
    """Convert an ESPN league schedule into the canonical dataframe layout."""

    rows: List[Dict[str, Any]] = []
    for matchup in league.schedule:
        if matchup.home_team_id is None or matchup.away_team_id is None:
            continue
        home_team = league.teams.get(matchup.home_team_id)
        away_team = league.teams.get(matchup.away_team_id)
        if home_team is None or away_team is None:
            continue
        rows.append(
            {
                "Week": matchup.matchup_period,
                "AwayTeam": away_team.name,
                "AwayManagers": ", ".join(away_team.managers) if away_team.managers else "",
                "AwayScore": matchup.away_points,
                "HomeTeam": home_team.name,
                "HomeManagers": ", ".join(home_team.managers) if home_team.managers else "",
                "HomeScore": matchup.home_points,
                "MatchupId": matchup.matchup_id,
                "Winner": matchup.winner,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Week",
                "AwayTeam",
                "AwayManagers",
                "AwayScore",
                "HomeTeam",
                "HomeManagers",
                "HomeScore",
                "MatchupId",
                "Winner",
            ]
        )

    df = pd.DataFrame(rows)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["AwayScore"] = pd.to_numeric(df["AwayScore"], errors="coerce")
    df["HomeScore"] = pd.to_numeric(df["HomeScore"], errors="coerce")
    df = df.sort_values(["Week", "AwayTeam", "HomeTeam"]).reset_index(drop=True)
    return df


def canonical_team_id(name: str) -> str:
    """Normalize team display names to a consistent identifier."""

    if not isinstance(name, str):
        raise TypeError(f"Expected string team name, received {type(name)!r}")
    cleaned = re.sub(r"[^a-z0-9]", "", name.lower())
    if not cleaned:
        raise ValueError(f"Unable to canonicalize team name: {name!r}")
    return cleaned


def build_team_catalog(
    schedule: pd.DataFrame,
    roster_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, TeamInfo]:
    """Derive canonical team metadata (name, managers, roster key)."""

    if roster_lookup is None:
        roster_lookup = {
            canonical_team_id(team.replace("_", " ")): team for team in Fantasy_Rosters.keys()
        }

    catalog: Dict[str, TeamInfo] = {}
    for _, row in schedule.iterrows():
        for side in ("Away", "Home"):
            name = row.get(f"{side}Team")
            if not name:
                continue
            team_id = canonical_team_id(name)
            managers = row.get(f"{side}Managers")
            managers = managers if managers else None
            roster_key = roster_lookup.get(team_id)
            info = catalog.get(team_id)
            if info is None:
                catalog[team_id] = TeamInfo(
                    team_id=team_id,
                    display_name=name,
                    managers=managers,
                    roster_key=roster_key,
                )
            else:
                if info.managers is None and managers:
                    info.managers = managers
                if info.roster_key is None and roster_key:
                    info.roster_key = roster_key
    return catalog


def _initialize_records(team_catalog: Dict[str, TeamInfo]) -> Dict[str, TeamRecord]:
    return {team_id: TeamRecord() for team_id in team_catalog}


def compute_records(schedule: pd.DataFrame, team_catalog: Dict[str, TeamInfo]) -> Dict[str, TeamRecord]:
    """Compute current win/loss records and scoring totals from played games."""

    records = _initialize_records(team_catalog)
    played_mask = schedule["AwayScore"].notna() & schedule["HomeScore"].notna()

    for _, row in schedule.loc[played_mask].iterrows():
        away_team = canonical_team_id(row["AwayTeam"])
        home_team = canonical_team_id(row["HomeTeam"])
        away_score = float(row["AwayScore"])
        home_score = float(row["HomeScore"])

        away_record = records.setdefault(away_team, TeamRecord())
        home_record = records.setdefault(home_team, TeamRecord())

        away_record.games_played += 1
        home_record.games_played += 1

        away_record.points_for += away_score
        away_record.points_against += home_score
        home_record.points_for += home_score
        home_record.points_against += away_score

        if math.isclose(away_score, home_score):
            away_record.ties += 1
            home_record.ties += 1
        elif away_score > home_score:
            away_record.wins += 1
            home_record.losses += 1
        else:
            home_record.wins += 1
            away_record.losses += 1

    return records


def compute_games_remaining(schedule: pd.DataFrame, team_catalog: Dict[str, TeamInfo]) -> Dict[str, int]:
    """Count remaining games for each team."""

    remaining = {team_id: 0 for team_id in team_catalog}
    future_mask = schedule["AwayScore"].isna() | schedule["HomeScore"].isna()
    for _, row in schedule.loc[future_mask].iterrows():
        away_team = canonical_team_id(row["AwayTeam"])
        home_team = canonical_team_id(row["HomeTeam"])
        remaining[away_team] = remaining.get(away_team, 0) + 1
        remaining[home_team] = remaining.get(home_team, 0) + 1
    return remaining


def build_standings_dataframe(
    records: Dict[str, TeamRecord],
    team_catalog: Dict[str, TeamInfo],
    games_remaining: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Turn record map into a sorted standings dataframe."""

    rows: List[Dict[str, Any]] = []
    for team_id, record in records.items():
        info = team_catalog.get(team_id)
        if info is None:
            continue
        remaining_games = games_remaining.get(team_id, 0) if games_remaining else 0
        rows.append(
            {
                "TeamId": team_id,
                "Team": info.display_name,
                "Managers": info.managers,
                "Wins": record.wins,
                "Losses": record.losses,
                "Ties": record.ties,
                "GamesPlayed": record.games_played,
                "GamesRemaining": remaining_games,
                "PointsFor": record.points_for,
                "PointsAgainst": record.points_against,
                "WinPct": record.win_pct,
            }
        )

    standings = pd.DataFrame(rows)
    if standings.empty:
        return standings
    standings = standings.sort_values(
        ["WinPct", "PointsFor", "PointsAgainst", "TeamId"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    standings["Rank"] = standings.index + 1
    return standings


def _historical_scores(schedule: pd.DataFrame) -> Dict[str, List[float]]:
    samples: Dict[str, List[float]] = {}
    played_mask = schedule["AwayScore"].notna() & schedule["HomeScore"].notna()
    for _, row in schedule.loc[played_mask].iterrows():
        away_team = canonical_team_id(row["AwayTeam"])
        home_team = canonical_team_id(row["HomeTeam"])
        samples.setdefault(away_team, []).append(float(row["AwayScore"]))
        samples.setdefault(home_team, []).append(float(row["HomeScore"]))
    return samples


def _logistic_prob(diff: float, scale: float = LOGIT_SCALE) -> float:
    """Convert a rating differential into a win probability."""

    try:
        return 1.0 / (1.0 + math.exp(-scale * diff))
    except OverflowError:
        return 1.0 if diff > 0 else 0.0


def derive_team_metrics(
    schedule: pd.DataFrame,
    team_catalog: Dict[str, TeamInfo],
    league_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, TeamMetrics]:
    """Derive mean/std scoring estimates blending history, projections, and power ratings."""

    history = _historical_scores(schedule)
    metrics: Dict[str, TeamMetrics] = {}

    starter_projections: Dict[str, float] = {}
    combined_scores: Dict[str, float] = {}
    bench_totals: Dict[str, float] = {}
    bench_scores_array = np.array([], dtype=float)
    rating_array = np.array([], dtype=float)

    if league_snapshot:
        starter_projections = league_snapshot.get("starter_projections", {})
        combined_scores = league_snapshot.get("combined_scores", {})
        bench_totals = league_snapshot.get("bench_totals", {})
        if combined_scores:
            rating_array = np.array(list(combined_scores.values()), dtype=float)
        if bench_totals:
            bench_scores_array = np.array(list(bench_totals.values()), dtype=float)

    rating_mean = float(rating_array.mean()) if rating_array.size else 0.0
    rating_std = float(rating_array.std(ddof=0)) if rating_array.size else 0.0
    if rating_std == 0.0:
        rating_std = 1.0

    bench_mean = float(bench_scores_array.mean()) if bench_scores_array.size else 0.0
    bench_std = float(bench_scores_array.std(ddof=0)) if bench_scores_array.size else 0.0

    for team_id, info in team_catalog.items():
        history_samples = np.array(history.get(team_id, []), dtype=float)
        hist_mean = float(history_samples.mean()) if history_samples.size else None
        hist_std = (
            float(history_samples.std(ddof=1)) if history_samples.size > 1 else None
        )

        projection = None
        rating_raw = None
        bench_raw = None

        roster_key = info.roster_key
        if roster_key:
            projection = starter_projections.get(roster_key)
            rating_raw = combined_scores.get(roster_key)
            bench_raw = bench_totals.get(roster_key)

        if rating_raw is None and combined_scores:
            # fall back to league-average rating if the roster key was not found
            rating_raw = rating_mean
        if rating_raw is None:
            rating_raw = rating_mean

        rating_z = (float(rating_raw) - rating_mean) / rating_std if rating_std else 0.0

        samples_count = int(history_samples.size)
        hist_weight = min(0.6, 0.15 * samples_count) if hist_mean is not None else 0.0
        remaining_weight = 1.0 - hist_weight

        base_components: List[tuple[float, float]] = []
        if hist_mean is not None and hist_weight > 0.0:
            base_components.append((hist_weight, hist_mean))

        proj_weight = 0.0
        if projection is not None:
            proj_weight = min(remaining_weight, 0.5 if hist_weight < 0.2 else 0.35)
            base_components.append((proj_weight, float(projection)))
            remaining_weight = max(0.0, remaining_weight - proj_weight)

        baseline_mean = DEFAULT_MEAN_GUESS
        if base_components:
            total_weight = sum(weight for weight, _ in base_components)
            if total_weight > 0:
                baseline_mean = sum(weight * value for weight, value in base_components) / total_weight

        rating_adjust = rating_z * POWER_RATING_POINT_SCALE
        # allocate some of the remaining weight to rating-derived expectation
        rating_weight = max(0.2, 1.0 - (hist_weight + proj_weight))
        mean_score = baseline_mean * (1.0 - rating_weight) + (baseline_mean + rating_adjust) * rating_weight

        if hist_std is not None:
            std_dev = max(MIN_STD_DEV, hist_std)
        elif hist_mean is not None:
            std_dev = max(MIN_STD_DEV, abs(hist_mean) * STD_FRACTION)
        elif projection is not None:
            std_dev = max(MIN_STD_DEV, abs(float(projection)) * STD_FRACTION)
        else:
            std_dev = MIN_STD_DEV * 1.5

        bench_adjust = 1.0
        if bench_raw is not None and bench_std and bench_std > 1e-6:
            bench_z = (float(bench_raw) - bench_mean) / bench_std
            bench_z = float(np.clip(bench_z, -2.5, 2.5))
            bench_adjust = float(np.clip(1.0 - 0.08 * bench_z, BENCH_VARIANCE_FLOOR, BENCH_VARIANCE_CAP))
            std_dev = max(MIN_STD_DEV, std_dev * bench_adjust)

        metrics[team_id] = TeamMetrics(
            mean_score=float(mean_score),
            std_dev=float(std_dev),
            rating=float(rating_raw),
            rating_z=float(rating_z),
            bench_adjust=float(bench_adjust),
            source={
                "history_samples": samples_count,
                "hist_mean": hist_mean,
                "hist_std": hist_std,
                "projection": projection,
                "rating": rating_raw,
                "rating_z": rating_z,
                "bench": bench_raw,
            },
        )
    return metrics


def compute_remaining_sos(
    schedule: pd.DataFrame,
    team_catalog: Dict[str, TeamInfo],
    team_metrics: Dict[str, TeamMetrics],
) -> Dict[str, float]:
    """Estimate remaining strength of schedule using opponent rating z-scores."""

    totals: Dict[str, Dict[str, float]] = {
        team_id: {"sum": 0.0, "count": 0.0} for team_id in team_catalog
    }
    future_mask = schedule["AwayScore"].isna() | schedule["HomeScore"].isna()
    future_games = schedule.loc[future_mask]

    for _, row in future_games.iterrows():
        away_id = canonical_team_id(row["AwayTeam"])
        home_id = canonical_team_id(row["HomeTeam"])
        away_metrics = team_metrics.get(away_id)
        home_metrics = team_metrics.get(home_id)

        if away_metrics is not None:
            totals[home_id]["sum"] += away_metrics.rating_z
            totals[home_id]["count"] += 1.0
        if home_metrics is not None:
            totals[away_id]["sum"] += home_metrics.rating_z
            totals[away_id]["count"] += 1.0

    sos: Dict[str, float] = {}
    for team_id, data in totals.items():
        count = data["count"]
        sos[team_id] = data["sum"] / count if count else 0.0
    return sos


def _fallback_metrics() -> TeamMetrics:
    return TeamMetrics(
        mean_score=DEFAULT_MEAN_GUESS,
        std_dev=MIN_STD_DEV,
        rating=0.0,
        rating_z=0.0,
        sos_remaining=0.0,
        bench_adjust=1.0,
        source={"fallback": True},
    )


def simulate_playoff_odds(
    schedule: pd.DataFrame,
    team_catalog: Dict[str, TeamInfo],
    base_records: Dict[str, TeamRecord],
    team_metrics: Dict[str, TeamMetrics],
    *,
    playoff_teams: int = DEFAULT_PLAYOFF_TEAMS,
    num_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Simulate the rest of the season to estimate playoff odds."""

    future_mask = schedule["AwayScore"].isna() | schedule["HomeScore"].isna()
    future_games = schedule.loc[future_mask].copy()
    if future_games.empty:
        teams = list(team_catalog.keys())
        ordered = sorted(
            teams,
            key=lambda tid: (
                base_records[tid].win_pct,
                base_records[tid].points_for,
                (team_metrics.get(tid) or _fallback_metrics()).rating,
                tid,
            ),
            reverse=True,
        )
        summary: Dict[str, Dict[str, Any]] = {}
        for seed_idx, team_id in enumerate(ordered, start=1):
            summary[team_id] = {
                "playoff_probability": 1.0 if seed_idx <= playoff_teams else 0.0,
                "average_seed": seed_idx,
                "median_seed": seed_idx,
                "best_seed": seed_idx,
                "worst_seed": seed_idx,
            }
        for team_id in team_catalog:
            summary.setdefault(
                team_id,
                {
                    "playoff_probability": 0.0,
                    "average_seed": None,
                    "median_seed": None,
                    "best_seed": None,
                    "worst_seed": None,
                },
            )
        return summary

    rng = np.random.default_rng(seed)
    teams = list(team_catalog.keys())
    sim_data: Dict[str, Dict[str, Any]] = {
        team_id: {
            "playoff_count": 0,
            "seed_sum": 0.0,
            "seeds": [],
            "best_seed": None,
            "worst_seed": None,
        }
        for team_id in teams
    }

    for _ in range(num_simulations):
        season_records = {tid: rec.clone() for tid, rec in base_records.items()}
        for _, row in future_games.iterrows():
            away_id = canonical_team_id(row["AwayTeam"])
            home_id = canonical_team_id(row["HomeTeam"])
            away_metrics = team_metrics.get(away_id) or _fallback_metrics()
            home_metrics = team_metrics.get(home_id) or _fallback_metrics()

            rating_diff = away_metrics.rating_z - home_metrics.rating_z
            prob_away = _logistic_prob(rating_diff)
            outcome_roll = rng.random()
            if outcome_roll < TIE_PROBABILITY:
                outcome = "tie"
            elif outcome_roll < TIE_PROBABILITY + prob_away * (1.0 - TIE_PROBABILITY):
                outcome = "away"
            else:
                outcome = "home"

            rating_shift = (away_metrics.rating_z - home_metrics.rating_z) * (MARGIN_PER_RATING_Z / 2.0)
            away_mean = max(0.0, away_metrics.mean_score + rating_shift)
            home_mean = max(0.0, home_metrics.mean_score - rating_shift)
            away_score = max(
                0.0, float(rng.normal(away_mean, away_metrics.std_dev))
            )
            home_score = max(
                0.0, float(rng.normal(home_mean, home_metrics.std_dev))
            )

            if outcome == "tie":
                base_level = max(0.0, (away_mean + home_mean) / 2.0)
                noise = float(rng.normal(0.0, 4.0))
                score = max(0.0, base_level + noise)
                away_score = home_score = score
            elif outcome == "away" and away_score <= home_score:
                margin = max(1.0, float(rng.normal(6.0, 2.5)))
                away_score = home_score + margin
            elif outcome == "home" and home_score <= away_score:
                margin = max(1.0, float(rng.normal(6.0, 2.5)))
                home_score = away_score + margin

            away_record = season_records[away_id]
            home_record = season_records[home_id]

            away_record.games_played += 1
            home_record.games_played += 1

            away_record.points_for += away_score
            away_record.points_against += home_score
            home_record.points_for += home_score
            home_record.points_against += away_score

            if outcome == "tie" or math.isclose(away_score, home_score, abs_tol=0.5):
                away_record.ties += 1
                home_record.ties += 1
            elif away_score > home_score:
                away_record.wins += 1
                home_record.losses += 1
            else:
                home_record.wins += 1
                away_record.losses += 1

        ordered = sorted(
            teams,
            key=lambda tid: (
                season_records[tid].win_pct,
                season_records[tid].points_for,
                (team_metrics.get(tid) or _fallback_metrics()).rating,
                tid,
            ),
            reverse=True,
        )

        for seed_idx, team_id in enumerate(ordered, start=1):
            entry = sim_data[team_id]
            entry["seed_sum"] += seed_idx
            entry["seeds"].append(seed_idx)
            entry["best_seed"] = seed_idx if entry["best_seed"] is None else min(entry["best_seed"], seed_idx)
            entry["worst_seed"] = seed_idx if entry["worst_seed"] is None else max(entry["worst_seed"], seed_idx)
            if seed_idx <= playoff_teams:
                entry["playoff_count"] += 1

    summary: Dict[str, Dict[str, Any]] = {}
    for team_id, entry in sim_data.items():
        seeds = entry["seeds"]
        summary[team_id] = {
            "playoff_probability": entry["playoff_count"] / num_simulations if num_simulations else 0.0,
            "average_seed": entry["seed_sum"] / num_simulations if num_simulations else None,
            "median_seed": float(np.median(seeds)) if seeds else None,
            "best_seed": entry["best_seed"],
            "worst_seed": entry["worst_seed"],
        }
    return summary


def compute_playoff_predictions(
    *,
    schedule_path: str | Path = DEFAULT_SCHEDULE_PATH,
    rankings_path: Optional[str | Path] = None,
    projections_path: Optional[str | Path] = None,
    num_simulations: int = 5000,
    playoff_teams: int = DEFAULT_PLAYOFF_TEAMS,
    seed: Optional[int] = None,
    league_snapshot: Optional[Dict[str, Any]] = None,
    espn_league: Optional[ESPNLeagueData] = None,
) -> Dict[str, Any]:
    """High-level helper returning standings and playoff probabilities."""

    roster_lookup_override: Optional[Dict[str, str]] = None

    schedule_path_obj: Optional[Path]
    if schedule_path:
        schedule_path_obj = Path(schedule_path)
    else:
        schedule_path_obj = None

    use_espn_schedule = espn_league is not None and (
        schedule_path_obj is None or schedule_path_obj == DEFAULT_SCHEDULE_PATH
    )

    if use_espn_schedule and espn_league is not None:
        schedule = espn_schedule_to_dataframe(espn_league)
        roster_lookup_override = {
            canonical_team_id(team.name): team.slug for team in espn_league.teams.values()
        }
        schedule_source_path = "ESPN"
    else:
        resolved_path = schedule_path_obj if schedule_path_obj else DEFAULT_SCHEDULE_PATH
        schedule = load_schedule(resolved_path)
        schedule_source_path = str(resolved_path)

    team_catalog = build_team_catalog(schedule, roster_lookup_override)
    base_records = compute_records(schedule, team_catalog)
    games_remaining = compute_games_remaining(schedule, team_catalog)
    standings_df = build_standings_dataframe(base_records, team_catalog, games_remaining)

    if league_snapshot is None:
        rankings_resolved = Path(rankings_path) if rankings_path else DEFAULT_RANKINGS
        projections_resolved = (
            Path(projections_path) if projections_path else DEFAULT_PROJECTIONS
        )
        league_snapshot = evaluate_league(
            str(rankings_resolved),
            projections_path=str(projections_resolved),
        )

    team_metrics = derive_team_metrics(schedule, team_catalog, league_snapshot)
    sos_remaining = compute_remaining_sos(schedule, team_catalog, team_metrics)
    for team_id, sos_value in sos_remaining.items():
        metric = team_metrics.get(team_id)
        if metric:
            metric.sos_remaining = float(sos_value)
            metric.source["sos_remaining"] = float(sos_value)

    sim_summary = simulate_playoff_odds(
        schedule,
        team_catalog,
        base_records,
        team_metrics,
        playoff_teams=playoff_teams,
        num_simulations=num_simulations,
        seed=seed,
    )

    standings_records = standings_df.to_dict(orient="records") if not standings_df.empty else []
    future_weeks = sorted(
        schedule.loc[schedule["AwayScore"].isna() | schedule["HomeScore"].isna(), "Week"]
        .dropna()
        .unique()
        .tolist()
    )

    team_rows: List[Dict[str, Any]] = []
    for team_id, info in team_catalog.items():
        record = base_records.get(team_id, TeamRecord())
        sim = sim_summary.get(team_id, {})
        metrics = team_metrics.get(team_id)
        team_rows.append(
            {
                "team_id": team_id,
                "team": info.display_name,
                "managers": info.managers,
                "wins": record.wins,
                "losses": record.losses,
                "ties": record.ties,
                "games_played": record.games_played,
                "games_remaining": games_remaining.get(team_id, 0),
                "points_for": record.points_for,
                "points_against": record.points_against,
                "win_pct": record.win_pct,
                "mean_score": metrics.mean_score if metrics else None,
                "std_dev": metrics.std_dev if metrics else None,
                "rating": metrics.rating if metrics else None,
                "rating_z": metrics.rating_z if metrics else None,
                "sos_remaining": metrics.sos_remaining if metrics else None,
                "bench_adjust": metrics.bench_adjust if metrics else None,
                "playoff_probability": sim.get("playoff_probability"),
                "average_seed": sim.get("average_seed"),
                "median_seed": sim.get("median_seed"),
                "best_seed": sim.get("best_seed"),
                "worst_seed": sim.get("worst_seed"),
            }
        )

    team_rows.sort(
        key=lambda row: (
            row["playoff_probability"] or 0.0,
            row["win_pct"],
            row.get("rating", 0.0) or 0.0,
        ),
        reverse=True,
    )

    return {
        "standings": standings_records,
        "teams": team_rows,
        "simulation": {
            "runs": num_simulations,
            "playoff_spots": playoff_teams,
            "seed": seed,
            "pending_weeks": future_weeks,
            "schedule_path": schedule_source_path,
        },
    }


__all__ = [
    "DEFAULT_SCHEDULE_PATH",
    "DEFAULT_PLAYOFF_TEAMS",
    "TeamInfo",
    "TeamRecord",
    "TeamMetrics",
    "load_schedule",
    "compute_records",
    "compute_games_remaining",
    "build_standings_dataframe",
    "derive_team_metrics",
    "compute_remaining_sos",
    "simulate_playoff_odds",
    "compute_playoff_predictions",
]
