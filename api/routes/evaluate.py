from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    EvaluateRequest,
    EvaluateResponse,
    LeaderboardEntry,
    TeamDetail,
    TeamEvaluation,
)
from context import ContextManager
from main import DEFAULT_SUPPLEMENTAL_RANKINGS, evaluate_league
from api.utils import build_leaderboards_map, build_team_details, coerce_float

router = APIRouter(prefix="/evaluate", tags=["evaluate"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=EvaluateResponse, summary="Evaluate league with optional overrides")
async def evaluate_endpoint(
    payload: EvaluateRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> EvaluateResponse:
    ctx = manager.get()

    rankings_path = payload.rankings_path or str(ctx.rankings_path)
    projections_path = payload.projections_path or (str(ctx.projections_path) if ctx.projections_path else None)
    supplemental_path = payload.supplemental_path
    if supplemental_path is None and ctx.supplemental_path is not None:
        supplemental_path = str(ctx.supplemental_path)
    if supplemental_path is None and DEFAULT_SUPPLEMENTAL_RANKINGS.exists():
        supplemental_path = str(DEFAULT_SUPPLEMENTAL_RANKINGS)

    rosters = payload.rosters if payload.rosters is not None else ctx.rosters
    rosters_copy = copy.deepcopy(rosters)

    eval_kwargs = {}
    if payload.projection_scale_beta is not None:
        eval_kwargs["projection_scale_beta"] = payload.projection_scale_beta
    if payload.replacement_skip_pct is not None:
        eval_kwargs["replacement_skip_pct"] = payload.replacement_skip_pct
    if payload.replacement_window is not None:
        eval_kwargs["replacement_window"] = payload.replacement_window
    if payload.bench_ovar_beta is not None:
        eval_kwargs["bench_ovar_beta"] = payload.bench_ovar_beta
    if payload.combined_starters_weight is not None:
        eval_kwargs["combined_starters_weight"] = payload.combined_starters_weight
    if payload.combined_bench_weight is not None:
        eval_kwargs["combined_bench_weight"] = payload.combined_bench_weight
    if payload.bench_z_fallback_threshold is not None:
        eval_kwargs["bench_z_fallback_threshold"] = payload.bench_z_fallback_threshold
    if payload.bench_percentile_clamp is not None:
        eval_kwargs["bench_percentile_clamp"] = payload.bench_percentile_clamp
    if payload.scarcity_sample_step is not None:
        eval_kwargs["scarcity_sample_step"] = payload.scarcity_sample_step

    league = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=rosters_copy,
        supplemental_path=supplemental_path,
        **eval_kwargs,
    )

    combined_scores = league["combined_scores"]
    starters_totals = league["starters_totals"]
    bench_totals = league["bench_totals"]
    starter_projections = league["starter_projections"]
    leaderboards_raw = league["leaderboards"]

    combined_lb = leaderboards_raw.get("combined", [])
    team_entries: List[TeamEvaluation] = []
    for team, value in combined_lb:
        team_entries.append(
            TeamEvaluation(
                team=str(team),
                combined_score=coerce_float(value),
                starter_vor=coerce_float(starters_totals.get(team)),
                bench_score=coerce_float(bench_totals.get(team)),
                starter_projection=coerce_float(starter_projections.get(team)),
            )
        )

    evaluated_at = datetime.now(timezone.utc)
    details_payload = None
    if payload.include_details:
        bench_tables = league.get("bench_tables", {})
        team_results = league.get("results", {})
        ordered_teams = [entry.team for entry in team_entries]
        details_payload = build_team_details(
            ordered_teams,
            results=team_results,
            bench_tables=bench_tables,
            bench_limit=payload.bench_limit,
        )

    return EvaluateResponse(
        evaluated_at=evaluated_at,
        player_count=int(league["df"].shape[0]) if not league["df"].empty else 0,
        teams=team_entries,
        leaderboards=build_leaderboards_map(leaderboards_raw),
        settings=league["settings"],
        replacement_points={k: coerce_float(v) for k, v in league["replacement_points"].items()},
        replacement_targets={k: coerce_float(v) for k, v in league["replacement_targets"].items()},
        scarcity_samples=league.get("scarcity_samples", {}),
        details=details_payload,
    )
