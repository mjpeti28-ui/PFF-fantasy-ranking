from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import PlayoffOddsResponse, PlayoffSimulationMeta, PlayoffStanding, PlayoffTeamProjection
from context import ContextManager
from playoffs import DEFAULT_PLAYOFF_TEAMS, DEFAULT_SCHEDULE_PATH, compute_playoff_predictions

router = APIRouter(prefix="/playoffs", tags=["playoffs"], dependencies=[Depends(require_api_key)])


def _resolve_schedule_path(schedule_path: Optional[str]) -> Path:
    if not schedule_path:
        return Path(DEFAULT_SCHEDULE_PATH)
    path = Path(schedule_path).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Schedule file not found: {path}")
    return path


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
