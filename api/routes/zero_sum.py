from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_context_manager, require_api_key
from api.models import ZeroSumEntry, ZeroSumPositionResponse
from api.utils import build_zero_sum_payload
from context import ContextManager
from power_snapshots import build_snapshot_metadata

router = APIRouter(prefix="/zero-sum", tags=["zero-sum"], dependencies=[Depends(require_api_key)])


def _top_holders(entries: list[ZeroSumEntry], *, limit: int = 3) -> list[dict[str, float]]:
    ranked = sorted(
        (
            {
                "team": entry.team,
                "value": entry.value,
                "share": entry.share,
                "surplus": entry.surplus,
            }
            for entry in entries
            if entry.surplus > 0
        ),
        key=lambda item: item["surplus"],
        reverse=True,
    )
    return ranked[:limit]


@router.get("/positions", response_model=ZeroSumPositionResponse, summary="Inspect zero-sum ledgers for a position")
async def zero_sum_position_endpoint(
    pos: str = Query(..., alias="pos", description="Position code (e.g., RB, WR, TE, QB)"),
    manager: ContextManager = Depends(get_context_manager),
) -> ZeroSumPositionResponse:
    ctx = manager.get()
    from main import evaluate_league  # local import to avoid circular import at module load time

    snapshot_metadata = build_snapshot_metadata(
        "api.zero_sum.position",
        ctx,
        tags=["api", "zero-sum", pos.upper()],
        position=pos.upper(),
    )

    league = evaluate_league(
        str(ctx.rankings_path),
        projections_path=str(ctx.projections_path) if ctx.projections_path else None,
        custom_rosters=ctx.rosters,
        snapshot_metadata=snapshot_metadata,
    )

    zero_sum_raw = league.get("zero_sum", {})
    zero_sum = build_zero_sum_payload(zero_sum_raw)

    position_key = pos.upper()
    starters_group = zero_sum.positions.get(position_key)
    if starters_group is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No zero-sum ledger found for position '{pos}'.",
        )

    bench_group = zero_sum.bench_positions.get(position_key)

    top_holders = _top_holders(starters_group.entries)
    pressure = []
    for team, analytics in zero_sum.analytics.teams.items():
        metric = analytics.scarcity_pressure.get(position_key)
        if metric and metric.deficit > 0:
            pressure.append(
                {
                    "team": team,
                    "deficit": metric.deficit,
                    "pressure": metric.pressure,
                }
            )
    pressure.sort(key=lambda item: item["deficit"], reverse=True)

    analytics_payload = {
        "topHolders": top_holders,
        "pressure": pressure[:3],
    }

    return ZeroSumPositionResponse(
        position=position_key,
        starters=starters_group,
        bench=bench_group,
        analytics=analytics_payload,
    )
