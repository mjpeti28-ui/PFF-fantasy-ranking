from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    PlayoffOddsResponse,
    PlayoffSimulationMeta,
    PlayoffStanding,
    PlayoffTeamDelta,
    PlayoffTeamProjection,
    PlayoffTradeRequest,
    PlayoffTradeResponse,
    TradePiece,
)
from alias import build_alias_map
from config import SCORABLE_POS
from context import ContextManager
from main import evaluate_league
from playoffs import DEFAULT_PLAYOFF_TEAMS, DEFAULT_SCHEDULE_PATH, compute_playoff_predictions
from optimizer import flatten_league_names
from trading import TradeFinder

router = APIRouter(prefix="/playoffs", tags=["playoffs"], dependencies=[Depends(require_api_key)])


def _resolve_schedule_path(schedule_path: Optional[str]) -> Path:
    if not schedule_path:
        return Path(DEFAULT_SCHEDULE_PATH)
    path = Path(schedule_path).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Schedule file not found: {path}")
    return path


def _convert_pieces(pieces: List[TradePiece]) -> List[tuple[str, str]]:
    return [(piece.group, piece.name) for piece in pieces]


def _validate_trade_players(
    rosters: Dict[str, Dict[str, List[str]]],
    team: str,
    pieces: List[tuple[str, str]],
) -> None:
    roster = rosters.get(team)
    if roster is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown team '{team}'.",
        )
    missing: List[str] = []
    for group, name in pieces:
        names = roster.get(group, [])
        if name not in names:
            missing.append(f"{name} ({group})")
    if missing:
        joined = ", ".join(missing)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Team '{team}' does not roster: {joined}",
        )


def _to_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff(base: Dict[str, Any], scenario: Dict[str, Any], key: str) -> Optional[float]:
    base_val = _to_float(base.get(key))
    scenario_val = _to_float(scenario.get(key))
    if base_val is None or scenario_val is None:
        return None
    return scenario_val - base_val


def _build_playoff_response(
    predictions: Dict[str, Any],
    *,
    simulations: int,
    playoff_teams: int,
    schedule_resolved: Path,
) -> PlayoffOddsResponse:
    standings_payload = [
        PlayoffStanding(
            teamId=row.get("TeamId"),
            team=row.get("Team"),
            managers=row.get("Managers"),
            wins=row.get("Wins", 0),
            losses=row.get("Losses", 0),
            ties=row.get("Ties", 0),
            gamesPlayed=row.get("GamesPlayed", 0),
            gamesRemaining=row.get("GamesRemaining", 0),
            pointsFor=row.get("PointsFor", 0.0),
            pointsAgainst=row.get("PointsAgainst", 0.0),
            winPct=row.get("WinPct", 0.0),
            rank=row.get("Rank", 0),
        )
        for row in predictions.get("standings", [])
    ]

    teams_payload = [
        PlayoffTeamProjection(
            teamId=entry.get("team_id"),
            team=entry.get("team"),
            managers=entry.get("managers"),
            wins=entry.get("wins", 0),
            losses=entry.get("losses", 0),
            ties=entry.get("ties", 0),
            gamesPlayed=entry.get("games_played", 0),
            gamesRemaining=entry.get("games_remaining", 0),
            pointsFor=entry.get("points_for", 0.0),
            pointsAgainst=entry.get("points_against", 0.0),
            winPct=entry.get("win_pct", 0.0),
            meanScore=entry.get("mean_score"),
            stdDev=entry.get("std_dev"),
            rating=entry.get("rating"),
            ratingZ=entry.get("rating_z"),
            sosRemaining=entry.get("sos_remaining"),
            benchVolatility=entry.get("bench_adjust"),
            playoffProbability=float(entry.get("playoff_probability") or 0.0),
            averageSeed=entry.get("average_seed"),
            medianSeed=entry.get("median_seed"),
            bestSeed=entry.get("best_seed"),
            worstSeed=entry.get("worst_seed"),
        )
        for entry in predictions.get("teams", [])
    ]

    sim_meta = predictions.get("simulation", {})
    pending_weeks = [
        int(week) for week in sim_meta.get("pending_weeks", []) if week is not None
    ]
    simulation_payload = PlayoffSimulationMeta(
        runs=int(sim_meta.get("runs", simulations)),
        playoffSpots=int(sim_meta.get("playoff_spots", playoff_teams)),
        seed=sim_meta.get("seed"),
        pendingWeeks=pending_weeks,
        schedulePath=sim_meta.get("schedule_path", str(schedule_resolved)),
    )

    return PlayoffOddsResponse(
        standings=standings_payload,
        teams=teams_payload,
        simulation=simulation_payload,
    )


def _build_delta_payload(
    baseline_predictions: Dict[str, Any],
    scenario_predictions: Dict[str, Any],
) -> List[PlayoffTeamDelta]:
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
    deltas: List[PlayoffTeamDelta] = []
    for team_id, base_entry in baseline_map.items():
        scenario_entry = scenario_map.get(team_id)
        if scenario_entry is None:
            continue
        prob_delta = (
            float(scenario_entry.get("playoff_probability") or 0.0)
            - float(base_entry.get("playoff_probability") or 0.0)
        )
        deltas.append(
            PlayoffTeamDelta(
                teamId=team_id,
                team=scenario_entry.get("team") or base_entry.get("team") or team_id,
                playoffProbabilityDelta=prob_delta,
                averageSeedDelta=_diff(base_entry, scenario_entry, "average_seed"),
                medianSeedDelta=_diff(base_entry, scenario_entry, "median_seed"),
                bestSeedDelta=_diff(base_entry, scenario_entry, "best_seed"),
                worstSeedDelta=_diff(base_entry, scenario_entry, "worst_seed"),
                meanScoreDelta=_diff(base_entry, scenario_entry, "mean_score"),
                stdDevDelta=_diff(base_entry, scenario_entry, "std_dev"),
                ratingDelta=_diff(base_entry, scenario_entry, "rating"),
                ratingZDelta=_diff(base_entry, scenario_entry, "rating_z"),
                sosRemainingDelta=_diff(base_entry, scenario_entry, "sos_remaining"),
                benchVolatilityDelta=_diff(base_entry, scenario_entry, "bench_adjust"),
            )
        )
    return deltas


@router.get(
    "/odds",
    response_model=PlayoffOddsResponse,
    summary="Compute playoff odds for the current league context",
)
async def get_playoff_odds(
    simulations: int = Query(5000, ge=500, le=50000, description="Number of Monte Carlo runs to execute."),
    playoff_teams: int = Query(
        DEFAULT_PLAYOFF_TEAMS,
        ge=2,
        description="How many teams qualify for the playoffs.",
        alias="playoffTeams",
    ),
    seed: Optional[int] = Query(
        default=None,
        description="Optional RNG seed for deterministic simulations.",
    ),
    schedule_path: Optional[str] = Query(
        default=None,
        description="Override path for the schedule CSV; defaults to schedule.csv in the project root.",
        alias="schedulePath",
    ),
    manager: ContextManager = Depends(get_context_manager),
) -> PlayoffOddsResponse:
    ctx = manager.get()
    schedule_resolved = _resolve_schedule_path(schedule_path)

    rankings_path = str(ctx.rankings_path)
    projections_path = str(ctx.projections_path) if ctx.projections_path else None

    try:
        predictions = compute_playoff_predictions(
            schedule_path=str(schedule_resolved),
            rankings_path=rankings_path,
            projections_path=projections_path,
            num_simulations=simulations,
            playoff_teams=playoff_teams,
            seed=seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - generic safeguard
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return _build_playoff_response(
        predictions,
        simulations=simulations,
        playoff_teams=playoff_teams,
        schedule_resolved=schedule_resolved,
    )


@router.post(
    "/trade",
    response_model=PlayoffTradeResponse,
    summary="Simulate playoff odds after applying a trade",
)
async def playoff_trade_simulation(
    payload: PlayoffTradeRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> PlayoffTradeResponse:
    ctx = manager.get()
    schedule_resolved = _resolve_schedule_path(payload.schedule_path)

    rosters = ctx.rosters
    if payload.team_a not in rosters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown team '{payload.team_a}'.",
        )
    if payload.team_b not in rosters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown team '{payload.team_b}'.",
        )

    send_a = _convert_pieces(payload.send_a)
    send_b = _convert_pieces(payload.send_b)

    _validate_trade_players(rosters, payload.team_a, send_a)
    _validate_trade_players(rosters, payload.team_b, send_b)

    rankings_path = str(ctx.rankings_path)
    projections_path = str(ctx.projections_path) if ctx.projections_path else None

    baseline_snapshot = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=deepcopy(rosters),
    )

    finder = TradeFinder(rankings_path, projections_path, build_baseline=False)
    finder.rosters = deepcopy(rosters)
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
    mutated_rosters = finder._apply_trade(  # noqa: SLF001
        finder.rosters,
        payload.team_a,
        payload.team_b,
        send_a,
        send_b,
    )

    scenario_snapshot = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=mutated_rosters,
    )

    baseline_predictions = compute_playoff_predictions(
        schedule_path=str(schedule_resolved),
        rankings_path=rankings_path,
        projections_path=projections_path,
        num_simulations=payload.simulations,
        playoff_teams=payload.playoff_teams,
        seed=payload.seed,
        league_snapshot=baseline_snapshot,
    )

    scenario_predictions = compute_playoff_predictions(
        schedule_path=str(schedule_resolved),
        rankings_path=rankings_path,
        projections_path=projections_path,
        num_simulations=payload.simulations,
        playoff_teams=payload.playoff_teams,
        seed=payload.seed,
        league_snapshot=scenario_snapshot,
    )

    baseline_response = _build_playoff_response(
        baseline_predictions,
        simulations=payload.simulations,
        playoff_teams=payload.playoff_teams,
        schedule_resolved=schedule_resolved,
    )
    scenario_response = _build_playoff_response(
        scenario_predictions,
        simulations=payload.simulations,
        playoff_teams=payload.playoff_teams,
        schedule_resolved=schedule_resolved,
    )
    deltas = _build_delta_payload(baseline_predictions, scenario_predictions)

    return PlayoffTradeResponse(
        baseline=baseline_response,
        scenario=scenario_response,
        delta=deltas,
    )
