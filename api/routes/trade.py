from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_context_manager, require_api_key
from api.models import (
    TradeEvaluateRequest,
    TradeEvaluateResponse,
    TradeFindRequest,
    TradeFindResponse,
    TradePiece,
    TradeProposal,
    TradeTeamResult,
)
from api.utils import build_leaderboards_map, build_team_details, coerce_float
from context import ContextManager
from trading import TradeFinder

router = APIRouter(prefix="/trade", tags=["trade"], dependencies=[Depends(require_api_key)])


def _instantiate_finder(ctx: ContextManager) -> TradeFinder:
    rankings_path = str(ctx.rankings_path)
    projections_path = str(ctx.projections_path) if ctx.projections_path else None
    return TradeFinder(rankings_path, projections_path, build_baseline=True)


def _convert_pieces(pieces) -> List[Tuple[str, str]]:
    return [(piece.group, piece.name) for piece in pieces]


def _convert_to_tradepieces(entries: List[Tuple[str, str]]) -> List[TradePiece]:
    return [TradePiece(group=grp, name=name) for grp, name in entries]


@router.post("/evaluate", response_model=TradeEvaluateResponse, summary="Evaluate a proposed trade")
async def evaluate_trade_endpoint(
    payload: TradeEvaluateRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> TradeEvaluateResponse:
    ctx = manager.get()
    finder = _instantiate_finder(ctx)

    send_a = _convert_pieces(payload.send_a)
    send_b = _convert_pieces(payload.send_b)

    try:
        trade_data = finder.evaluate_trade(
            payload.team_a,
            payload.team_b,
            send_a,
            send_b,
            include_details=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    baseline = finder.baseline
    if baseline is None:
        baseline = finder._evaluate(finder.rosters, include_details=True)  # noqa: SLF001
        finder.baseline = baseline

    baseline_combined = baseline["combined_scores"]
    combined_scores = trade_data.get("combined_scores", {})

    teams_payload: List[TradeTeamResult] = []
    for team in [payload.team_a, payload.team_b]:
        base_val = coerce_float(baseline_combined.get(team))
        post_val = coerce_float(combined_scores.get(team))
        teams_payload.append(
            TradeTeamResult(
                team=team,
                baseline=base_val,
                post_trade=post_val,
                delta=post_val - base_val,
            )
        )

    details_payload = None
    if payload.include_details:
        results = trade_data.get("results", {})
        bench_tables = trade_data.get("bench_tables", {})
        details_payload = build_team_details(
            [payload.team_a, payload.team_b],
            results=results,
            bench_tables=bench_tables,
            bench_limit=payload.bench_limit,
    )

    leaderboards_raw: Dict[str, List] = {
        "starters": trade_data.get("starters_board", []),
        "bench": trade_data.get("bench_board", []),
        "combined": trade_data.get("combined_board", []),
    }

    evaluated_at = datetime.now(timezone.utc)
    return TradeEvaluateResponse(
        evaluated_at=evaluated_at,
        teams=teams_payload,
        combined_scores={team: coerce_float(val) for team, val in combined_scores.items()},
        replacement_points={k: coerce_float(v) for k, v in trade_data.get("replacement_points", {}).items()},
        replacement_targets={k: coerce_float(v) for k, v in trade_data.get("replacement_targets", {}).items()},
        starter_vor={team: coerce_float(val) for team, val in trade_data.get("starter_vor", {}).items()},
        bench_totals={team: coerce_float(val) for team, val in trade_data.get("bench_totals", {}).items()},
        leaderboards=build_leaderboards_map(leaderboards_raw),
        details=details_payload,
    )


@router.post("/find", response_model=TradeFindResponse, summary="Search for favorable trades")
async def find_trades_endpoint(
    payload: TradeFindRequest,
    manager: ContextManager = Depends(get_context_manager),
) -> TradeFindResponse:
    ctx = manager.get()
    finder = _instantiate_finder(ctx)

    try:
        results = finder.find_trades(
            payload.team_a,
            payload.team_b,
            max_players=payload.max_players,
            player_pool=payload.player_pool,
            top_results=payload.top_results,
            top_bench=payload.top_bench,
            min_gain_a=payload.min_gain_a,
            max_loss_b=payload.max_loss_b,
            prune_margin=payload.prune_margin,
            min_upper_bound=payload.min_upper_bound,
            fairness_mode=payload.fairness_mode,
            fairness_self_bias=payload.fairness_self_bias,
            fairness_penalty_weight=payload.fairness_penalty_weight,
            consolidation_bonus=payload.consolidation_bonus,
            drop_tax_factor=payload.drop_tax_factor,
            acceptance_fairness_weight=payload.acceptance_fairness_weight,
            acceptance_need_weight=payload.acceptance_need_weight,
            acceptance_star_weight=payload.acceptance_star_weight,
            acceptance_need_scale=payload.acceptance_need_scale,
            star_vor_scale=payload.star_vor_scale,
            drop_tax_acceptance_weight=payload.drop_tax_acceptance_weight,
            narrative_on=payload.narrative_on,
            min_acceptance=payload.min_acceptance,
            verbose=False,
            show_progress=False,
            must_receive_from_b=payload.must_receive_b or None,
            must_send_from_a=payload.must_send_a or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    baseline = finder.baseline
    if baseline is None:
        baseline = finder._evaluate(finder.rosters, include_details=True)  # noqa: SLF001
        finder.baseline = baseline

    proposals: List[TradeProposal] = []
    for entry in results:
        combined = {team: coerce_float(val) for team, val in entry.get("combined", {}).items()}
        delta = {team: coerce_float(val) for team, val in entry.get("delta", {}).items()}
        evaluation = entry.get("evaluation", {})
        details_payload = None
        leaderboards_payload = None
        if payload.include_details and evaluation:
            details_payload = build_team_details(
                [payload.team_a, payload.team_b],
                results=evaluation.get("results", {}),
                bench_tables=evaluation.get("bench_tables", {}),
                bench_limit=payload.bench_limit,
            )
        if evaluation:
            leaderboards_payload = build_leaderboards_map(
                {
                    "starters": evaluation.get("starters_board", []),
                    "bench": evaluation.get("bench_board", []),
                    "combined": evaluation.get("combined_board", []),
                }
            )

        proposals.append(
            TradeProposal(
                send_a=_convert_to_tradepieces(entry.get("send_a", [])),
                send_b=_convert_to_tradepieces(entry.get("send_b", [])),
                receive_a=_convert_to_tradepieces(entry.get("receive_a", [])),
                receive_b=_convert_to_tradepieces(entry.get("receive_b", [])),
                combined_scores=combined,
                delta=delta,
                score=coerce_float(entry.get("score")),
                acceptance=coerce_float(entry.get("acceptance")),
                fairness_split=entry.get("fairness_split"),
                drop_tax={team: coerce_float(val) for team, val in entry.get("drop_tax", {}).items()},
                star_gain={team: coerce_float(val) for team, val in entry.get("star_gain", {}).items()},
                narrative={team: str(msg) for team, msg in entry.get("narrative", {}).items()},
                details=details_payload,
                leaderboards=leaderboards_payload,
            )
        )

    evaluated_at = datetime.now(timezone.utc)
    baseline_combined = baseline.get("combined_scores", {}) if baseline else {}
    return TradeFindResponse(
        evaluated_at=evaluated_at,
        baseline_combined={team: coerce_float(val) for team, val in baseline_combined.items()},
        proposals=proposals,
    )
