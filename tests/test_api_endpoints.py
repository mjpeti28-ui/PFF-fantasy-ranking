from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


def _assert_json_keys(payload: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        assert key in payload, f"Expected key '{key}' in response payload"


def test_league_history_endpoint(client: TestClient) -> None:
    payload = {
        "window": "explicit",
        "weeks": [1],
        "includeRosters": False,
        "includeMatchups": False,
        "includePlayerStats": False,
        "includePowerRankings": False,
    }
    resp = client.post("/league/history", json=payload)
    if resp.status_code == 503:
        pytest.skip("ESPN data unavailable for history endpoint")
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(data, ["generatedAt", "weeks"])
    assert data["weeks"], "Expected at least one week in response"
    week = data["weeks"][0]
    _assert_json_keys(week, ["week", "teams"])


def test_players_endpoint(client: TestClient) -> None:
    resp = client.get("/players", params={"pos": "RB", "metric": "proj", "limit": 3})
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(data, ["items", "total", "limit", "offset", "metric"])
    assert data["limit"] == 3
    assert len(data["items"]) <= 3
    if data["items"]:
        sample = data["items"][0]
        _assert_json_keys(sample, ["id", "name", "position", "rank"])


def test_rankings_endpoint(client: TestClient) -> None:
    resp = client.get("/rankings", params={"metric": "rank", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5
    assert len(data["items"]) <= 5


def test_evaluate_endpoint_basic(client: TestClient) -> None:
    resp = client.post("/evaluate", json={})
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(
        data,
        [
            "evaluatedAt",
            "playerCount",
            "teams",
            "leaderboards",
            "replacementPoints",
            "replacementTargets",
            "zeroSum",
        ],
    )
    assert isinstance(data["teams"], list) and data["teams"]
    zero_sum = data["zeroSum"]
    _assert_json_keys(
        zero_sum,
        [
            "teamCount",
            "starters",
            "bench",
            "combined",
            "positions",
            "benchPositions",
            "slots",
            "analytics",
        ],
    )


def test_evaluate_endpoint_with_details(client: TestClient) -> None:
    payload = {"includeDetails": True, "benchLimit": 2}
    resp = client.post("/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("details")
    detail = data["details"][0]
    _assert_json_keys(detail, ["team", "starters", "bench"])


def test_trade_evaluate_endpoint(client: TestClient) -> None:
    payload = {
        "teamA": "JohnBurrows_School",
        "teamB": "KWood_Super_Marios",
        "sendA": [{"group": "WR", "name": "George Pickens"}],
        "sendB": [{"group": "WR", "name": "Drake London"}],
        "includeDetails": True,
        "benchLimit": 1,
    }
    resp = client.post("/trade/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    teams = {entry["team"] for entry in data.get("teams", [])}
    assert {payload["teamA"], payload["teamB"]} == teams
    if data.get("details"):
        assert len(data["details"]) == 2
    _assert_json_keys(
        data,
        [
            "zeroSumBefore",
            "zeroSumAfter",
            "zeroSumShift",
        ],
    )
    assert "combined" in data["zeroSumShift"]


def test_trade_find_endpoint(client: TestClient) -> None:
    payload = {
        "teamA": "JohnBurrows_School",
        "teamB": "KWood_Super_Marios",
        "maxPlayers": 2,
        "playerPool": 6,
        "topResults": 2,
        "includeDetails": False,
    }
    resp = client.post("/trade/find", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "proposals" in data
    assert isinstance(data["proposals"], list)


def test_teams_endpoints(client: TestClient) -> None:
    list_resp = client.get("/teams")
    assert list_resp.status_code == 200
    teams = list_resp.json()
    assert isinstance(teams, list)
    assert teams
    team_name = teams[0]

    detail_resp = client.get(f"/teams/{team_name}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["team"] == team_name
    assert "starters" in detail


def test_waivers_candidates_endpoint(client: TestClient) -> None:
    resp = client.get("/waivers/candidates", params={"limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5
    assert "items" in data
    assert isinstance(data["items"], list)
    if data["items"]:
        sample = data["items"][0]
        assert "name" in sample


def test_waivers_recommend_endpoint(client: TestClient) -> None:
    candidates = client.get("/waivers/candidates", params={"limit": 1}).json()
    assert candidates["items"]
    add_name = candidates["items"][0]["name"]

    team_name = client.get("/teams").json()[0]
    team_detail = client.get(f"/teams/{team_name}").json()
    drop_name = team_detail["bench"][0]["name"] if team_detail["bench"] else team_detail["starters"][0]["name"]

    payload = {
        "changes": [
            {
                "team": team_name,
                "adds": [add_name],
                "drops": [drop_name],
            }
        ],
        "benchLimit": 1,
        "includeDetails": True,
    }
    resp = client.post("/waivers/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "teams" in data
    assert data["teams"]


def test_top_players_endpoint(client: TestClient) -> None:
    resp = client.get("/top/players", params={"pos": "WR", "metric": "proj", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) <= 5


def test_stats_endpoint(client: TestClient) -> None:
    resp = client.get("/stats/passing", params={"limit": 5, "sort": "-fantasyPts"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5
    assert isinstance(data["items"], list)


def test_sources_projections_endpoint(client: TestClient) -> None:
    resp = client.get("/sources/projections", params={"limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5


def test_evaluate_delta_endpoint(client: TestClient) -> None:
    resp = client.post("/evaluate/delta", json={})
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(data, ["evaluatedAt", "baseline", "scenario", "zeroSumShift"])
    baseline = data["baseline"]
    scenario = data["scenario"]
    _assert_json_keys(baseline, ["teams", "zeroSum"])
    _assert_json_keys(scenario, ["teams", "zeroSum"])


def test_team_leverage_endpoint(client: TestClient) -> None:
    team_name = client.get("/teams").json()[0]
    resp = client.get(f"/teams/{team_name}/leverage")
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(
        data,
        [
            "team",
            "combined",
            "starters",
            "bench",
            "positions",
            "benchPositions",
            "slots",
            "scarcityPressure",
            "concentrationRisk",
            "leveragePositions",
            "needPositions",
        ],
    )


def test_zero_sum_position_endpoint(client: TestClient) -> None:
    resp = client.get("/zero-sum/positions", params={"pos": "RB"})
    assert resp.status_code == 200
    data = resp.json()
    _assert_json_keys(data, ["position", "starters", "analytics"])
