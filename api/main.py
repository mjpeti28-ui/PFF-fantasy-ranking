from __future__ import annotations

from fastapi import FastAPI

from api.routes import config, health, league

app = FastAPI(title="PFF Fantasy League API", version="0.1.0")
app.include_router(health.router)
app.include_router(config.router)
app.include_router(league.router)


@app.get("/", summary="Root endpoint", tags=["health"])
async def root() -> dict[str, str]:
    return {"message": "PFF Fantasy League API"}
