from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, Query

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    EvaluateDeltaResponse,
    EvaluateDeltaSnapshot,
    EvaluateRequest,
    EvaluateResponse,
    LeaderboardEntry,
    TeamDetail,
    TeamEvaluation,
)
from context import ContextManager
from main import DEFAULT_SUPPLEMENTAL_RANKINGS, evaluate_league
from api.utils import (
    build_leaderboards_map,
    build_team_details,
    build_zero_sum_payload,
    build_zero_sum_shift,
    coerce_float,
)

router = APIRouter(prefix="/evaluate", tags=["evaluate"], dependencies=[Depends(require_api_key)])


def _build_team_evaluations(
    combined_leaderboard: List[tuple[Any, Any]],
    starters_totals: Dict[str, float],
    bench_totals: Dict[str, float],
    starter_projections: Dict[str, float],
) -> List[TeamEvaluation]:
    entries: List[TeamEvaluation] = []
    for team, value in combined_leaderboard:
        entries.append(
            TeamEvaluation(
                team=str(team),
                combined_score=coerce_float(value),
                starter_vor=coerce_float(starters_totals.get(team)),
                bench_score=coerce_float(bench_totals.get(team)),
                starter_projection=coerce_float(starter_projections.get(team)),
            )
        )
    return entries


@router.post("", response_model=EvaluateResponse, summary="Evaluate league with optional overrides")
async def evaluate_endpoint(
    payload: EvaluateRequest | None = Body(default=None),
    include_details: bool | None = Query(None, alias="includeDetails"),
    bench_limit: int | None = Query(None, alias="benchLimit"),
    manager: ContextManager = Depends(get_context_manager),
) -> EvaluateResponse:
    ctx = manager.get()

    if payload is None:
        payload = EvaluateRequest(includeDetails=include_details or False, benchLimit=bench_limit)
    else:
        # Allow simple query overrides to coexist with JSON body (query wins if provided)
        if include_details is not None:
            payload.include_details = include_details
        if bench_limit is not None:
            payload.bench_limit = bench_limit

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
    team_entries = _build_team_evaluations(
        combined_lb,
        starters_totals,
        bench_totals,
        starter_projections,
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
        zero_sum=build_zero_sum_payload(league.get("zero_sum", {})),
        details=details_payload,
    )


@router.post("/delta", response_model=EvaluateDeltaResponse, summary="Compare baseline and modified rosters")
async def evaluate_delta_endpoint(
    payload: EvaluateRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> EvaluateDeltaResponse:
    ctx = manager.get()

    rankings_path = payload.rankings_path or str(ctx.rankings_path)
    projections_path = payload.projections_path or (str(ctx.projections_path) if ctx.projections_path else None)
    supplemental_path = payload.supplemental_path
    if supplemental_path is None and ctx.supplemental_path is not None:
        supplemental_path = str(ctx.supplemental_path)
    if supplemental_path is None and DEFAULT_SUPPLEMENTAL_RANKINGS.exists():
        supplemental_path = str(DEFAULT_SUPPLEMENTAL_RANKINGS)

    eval_kwargs: Dict[str, Any] = {}
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

    baseline_league = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=copy.deepcopy(ctx.rosters),
        supplemental_path=supplemental_path,
        **eval_kwargs,
    )

    scenario_rosters = payload.rosters if payload.rosters is not None else ctx.rosters
    scenario_league = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=copy.deepcopy(scenario_rosters),
        supplemental_path=supplemental_path,
        **eval_kwargs,
    )

    baseline_combined_lb = baseline_league["leaderboards"].get("combined", [])
    baseline_team_entries = _build_team_evaluations(
        baseline_combined_lb,
        baseline_league["starters_totals"],
        baseline_league["bench_totals"],
        baseline_league["starter_projections"],
    )

    scenario_combined_lb = scenario_league["leaderboards"].get("combined", [])
    scenario_team_entries = _build_team_evaluations(
        scenario_combined_lb,
        scenario_league["starters_totals"],
        scenario_league["bench_totals"],
        scenario_league["starter_projections"],
    )

    baseline_zero_raw = baseline_league.get("zero_sum", {})
    scenario_zero_raw = scenario_league.get("zero_sum", {})
    baseline_zero = build_zero_sum_payload(baseline_zero_raw)
    scenario_zero = build_zero_sum_payload(scenario_zero_raw)
    zero_shift = build_zero_sum_shift(baseline_zero_raw, scenario_zero_raw)

    evaluated_at = datetime.now(timezone.utc)
    return EvaluateDeltaResponse(
        evaluated_at=evaluated_at,
        baseline=EvaluateDeltaSnapshot(teams=baseline_team_entries, zero_sum=baseline_zero),
        scenario=EvaluateDeltaSnapshot(teams=scenario_team_entries, zero_sum=scenario_zero),
        zero_sum_shift=zero_shift,
    )
