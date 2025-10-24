from __future__ import annotations

from fastapi import FastAPI

from api.routes import (
    config,
    evaluate,
    health,
    history,
    jobs,
    league,
    playoffs,
    players,
    rankings,
    sources,
    stats,
    teams,
    top,
    trade,
    waivers,
    zero_sum,
)

app = FastAPI(title="PFF Fantasy League API", version="0.1.0")
app.include_router(health.router)
app.include_router(config.router)
app.include_router(league.router)
app.include_router(evaluate.router)
app.include_router(players.router)
app.include_router(playoffs.router)
app.include_router(rankings.router)
app.include_router(history.router)
app.include_router(stats.router)
app.include_router(sources.router)
app.include_router(teams.router)
app.include_router(top.router)
app.include_router(trade.router)
app.include_router(waivers.router)
app.include_router(jobs.router)
app.include_router(zero_sum.router)


@app.get("/", summary="Root endpoint", tags=["health"])
async def root() -> dict[str, str]:
    return {"message": "PFF Fantasy League API"}
