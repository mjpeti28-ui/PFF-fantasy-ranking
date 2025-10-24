from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from context import LeagueDataContext
from espn_client import (
    ESPNLeagueData,
    ESPNMatchup,
    ESPNTeamData,
    IR_SLOT_IDS,
    LINEUP_SLOT_MAP,
    POSITION_ID_MAP,
    convert_league_to_rosters,
    fetch_league_payload_for_week,
    parse_league_payload,
)
from main import evaluate_league
from power_snapshots import build_snapshot_metadata


@dataclass(frozen=True)
class LeagueIdentifiers:
    league_id: str
    season: int
    latest_scoring_period: Optional[int]
    current_matchup_period: Optional[int]


def _load_week_state(
    scoring_period_id: int,
) -> Tuple[Dict[str, Any], ESPNLeagueData, Dict[str, Dict[str, List[str]]]]:
    payload = fetch_league_payload_for_week(scoring_period_id)
    league = parse_league_payload(payload)
    rosters = convert_league_to_rosters(league)
    return payload, league, rosters


@functools.lru_cache(maxsize=48)
def _load_week_state_cached(
    scoring_period_id: int,
) -> Tuple[Dict[str, Any], ESPNLeagueData, Dict[str, Dict[str, List[str]]]]:
    return _load_week_state(scoring_period_id)


def _extract_identifiers(league: ESPNLeagueData) -> LeagueIdentifiers:
    status = league.status or {}
    return LeagueIdentifiers(
        league_id=str(league.league_id),
        season=int(league.season),
        latest_scoring_period=status.get("latestScoringPeriod"),
        current_matchup_period=status.get("currentMatchupPeriod"),
    )


def _resolve_team_filter(
    league: ESPNLeagueData,
    requested: Optional[Sequence[str]],
) -> List[int]:
    if not requested:
        return sorted(league.teams.keys())

    normalized: List[str] = [item.strip().lower() for item in requested if item and item.strip()]
    if not normalized:
        return sorted(league.teams.keys())
    if any(value in {"*", "all"} for value in normalized):
        return sorted(league.teams.keys())

    slug_lookup = {slug.lower(): team_id for slug, team_id in league.team_slugs.items()}
    name_lookup = {team.name.lower(): team.team_id for team in league.teams.values()}
    id_lookup = {str(team.team_id): team.team_id for team in league.teams.values()}

    resolved: List[int] = []
    missing: List[str] = []
    for value in normalized:
        team_id = slug_lookup.get(value) or name_lookup.get(value) or id_lookup.get(value)
        if team_id is None:
            missing.append(value)
        else:
            if team_id not in resolved:
                resolved.append(team_id)

    if missing:
        raise ValueError(f"Unknown team identifiers: {', '.join(sorted(set(missing)))}")
    return sorted(resolved)


def _extract_player_stats(
    player_payload: Dict[str, Any],
    *,
    scoring_period_id: int,
) -> Optional[Dict[str, Any]]:
    stats: Iterable[Dict[str, Any]] = player_payload.get("stats") or []
    actual: Optional[Dict[str, Any]] = None
    projection: Optional[Dict[str, Any]] = None
    for stat in stats:
        if stat.get("scoringPeriodId") != scoring_period_id:
            continue
        stat_source = stat.get("statSourceId")
        payload = {
            "statSourceId": stat_source,
            "statSplitTypeId": stat.get("statSplitTypeId"),
            "fantasyPoints": stat.get("appliedTotal"),
            "appliedStats": stat.get("appliedStats") or {},
            "rawStats": stat.get("stats") or {},
        }
        if stat_source == 1:
            actual = payload
        elif stat_source == 0:
            projection = payload

    if actual or projection:
        result: Dict[str, Any] = {}
        if projection:
            result["projected"] = projection
        if actual:
            result["actual"] = actual
        return result
    return None


def _slot_name(slot_id: Optional[int]) -> Optional[str]:
    if slot_id is None:
        return None
    return LINEUP_SLOT_MAP.get(slot_id, str(slot_id))


def _build_roster_payload(
    team: ESPNTeamData,
    *,
    raw_team: Dict[str, Any],
    scoring_period_id: int,
    include_stats: bool,
) -> List[Dict[str, Any]]:
    entries: Iterable[Dict[str, Any]] = (raw_team.get("roster") or {}).get("entries") or []
    roster_payload: List[Dict[str, Any]] = []
    for entry in entries:
        player_pool_entry = entry.get("playerPoolEntry") or {}
        player = player_pool_entry.get("player") or {}
        default_position_id = player.get("defaultPositionId")
        default_position = POSITION_ID_MAP.get(default_position_id) if default_position_id is not None else None

        eligible_slots = [
            {
                "slotId": slot,
                "slotName": _slot_name(slot),
            }
            for slot in (player.get("eligibleSlots") or [])
        ]

        lineup_slot_id = entry.get("lineupSlotId")
        entry_payload: Dict[str, Any] = {
            "playerId": entry.get("playerId"),
            "name": player.get("fullName") or player.get("name"),
            "positionId": default_position_id,
            "position": default_position,
            "proTeamId": player.get("proTeamId"),
            "eligibleSlots": eligible_slots,
            "lineupSlotId": lineup_slot_id,
            "lineupSlot": _slot_name(lineup_slot_id),
            "injuryStatus": entry.get("injuryStatus") or player.get("injuryStatus"),
            "isInjured": bool(player.get("injured")),
            "onIR": bool(lineup_slot_id in IR_SLOT_IDS),
        }
        if include_stats:
            stats_payload = _extract_player_stats(player, scoring_period_id=scoring_period_id)
            if stats_payload:
                entry_payload["stats"] = stats_payload
        roster_payload.append(entry_payload)
    return roster_payload


def _matchup_to_payload(
    matchup: ESPNMatchup,
    team_lookup: Dict[int, ESPNTeamData],
) -> Dict[str, Any]:
    home_team = team_lookup.get(matchup.home_team_id)
    away_team = team_lookup.get(matchup.away_team_id)
    return {
        "matchupId": matchup.matchup_id,
        "week": matchup.matchup_period,
        "winner": matchup.winner,
        "isPlayoff": matchup.is_playoff,
        "home": {
            "teamId": home_team.team_id if home_team else matchup.home_team_id,
            "teamSlug": home_team.slug if home_team else None,
            "teamName": home_team.name if home_team else None,
            "points": matchup.home_points,
        },
        "away": {
            "teamId": away_team.team_id if away_team else matchup.away_team_id,
            "teamSlug": away_team.slug if away_team else None,
            "teamName": away_team.name if away_team else None,
            "points": matchup.away_points,
        },
    }


def _build_power_rankings(
    ctx: LeagueDataContext,
    roster_map: Dict[str, Dict[str, List[str]]],
    *,
    week: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rankings_path = str(ctx.rankings_path)
    projections_path = str(ctx.projections_path) if ctx.projections_path else None
    supplemental_path = str(ctx.supplemental_path) if ctx.supplemental_path else None

    metadata = build_snapshot_metadata(
        "api.league.history",
        ctx,
        week=week,
        tags=["history", f"week:{week}"] if week is not None else ["history"],
    )

    evaluation = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        custom_rosters=roster_map,
        supplemental_path=supplemental_path,
        snapshot_metadata=metadata,
    )

    combined_leaderboard = evaluation["leaderboards"].get("combined", [])
    starters_totals = evaluation.get("starters_totals", {})
    bench_totals = evaluation.get("bench_totals", {})
    starter_projections = evaluation.get("starter_projections", {})

    rankings: List[Dict[str, Any]] = []
    per_team: Dict[str, Dict[str, Any]] = {}
    for idx, (team_name, combined_score) in enumerate(combined_leaderboard, start=1):
        payload = {
            "teamSlug": team_name,
            "rank": idx,
            "team": team_name,
            "combinedScore": float(combined_score) if combined_score is not None else None,
            "starterVOR": float(starters_totals.get(team_name)) if starters_totals.get(team_name) is not None else None,
            "benchScore": float(bench_totals.get(team_name)) if bench_totals.get(team_name) is not None else None,
            "starterProjection": float(starter_projections.get(team_name)) if starter_projections.get(team_name) is not None else None,
        }
        rankings.append(payload)
        per_team[team_name] = payload

    return rankings, per_team


def collect_week_history(
    ctx: LeagueDataContext,
    *,
    weeks: Sequence[int],
    team_filter: Optional[Sequence[str]] = None,
    include_rosters: bool = True,
    include_matchups: bool = True,
    include_player_stats: bool = False,
    include_power_rankings: bool = True,
    request_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not weeks:
        raise ValueError("At least one week must be requested.")

    ordered_weeks = sorted(set(int(week) for week in weeks if int(week) > 0))
    if not ordered_weeks:
        raise ValueError("Only positive week numbers are supported.")

    generated_at = datetime.now(timezone.utc)
    response: Dict[str, Any] = {
        "generatedAt": generated_at,
        "weeks": [],
    }

    identifiers: Optional[LeagueIdentifiers] = None

    for week in ordered_weeks:
        payload, league, roster_map = _load_week_state_cached(week)
        if identifiers is None:
            identifiers = _extract_identifiers(league)

        team_ids = _resolve_team_filter(league, team_filter)
        team_lookup = {team.team_id: team for team in league.teams.values()}

        # Filter roster map down to requested teams when computing power rankings.
        power_rankings_summary: List[Dict[str, Any]] = []
        team_power_lookup: Dict[str, Dict[str, Any]] = {}
        if include_power_rankings:
            power_rankings_summary, team_power_lookup = _build_power_rankings(ctx, roster_map, week=week)

        raw_teams: List[Dict[str, Any]] = payload.get("teams") or []
        raw_team_lookup: Dict[int, Dict[str, Any]] = {}
        for team_entry in raw_teams:
            team_id = team_entry.get("id")
            if team_id is not None:
                raw_team_lookup[int(team_id)] = team_entry

        teams_payload: List[Dict[str, Any]] = []
        for team_id in team_ids:
            team = team_lookup.get(team_id)
            if team is None:
                continue
            raw_team = raw_team_lookup.get(team_id, {})
            team_payload: Dict[str, Any] = {
                "teamId": team.team_id,
                "teamSlug": team.slug,
                "teamName": team.name,
                "managers": team.managers,
                "record": team.record,
            }

            if include_rosters:
                team_payload["roster"] = _build_roster_payload(
                    team,
                    raw_team=raw_team,
                    scoring_period_id=week,
                    include_stats=include_player_stats,
                )
            if include_power_rankings:
                metrics = team_power_lookup.get(team.slug)
                if metrics:
                    team_payload["powerMetrics"] = {k: v for k, v in metrics.items() if k != "team"}
            teams_payload.append(team_payload)

        week_payload: Dict[str, Any] = {
            "week": week,
            "label": f"Week {week}",
            "teams": teams_payload,
        }

        if include_matchups:
            matchups = [
                matchup
                for matchup in league.schedule
                if matchup.matchup_period == week
                and (matchup.home_team_id in team_ids or matchup.away_team_id in team_ids)
            ]
            week_payload["matchups"] = [
                _matchup_to_payload(matchup, team_lookup) for matchup in matchups
            ]

        if include_power_rankings:
            week_payload["powerRankings"] = power_rankings_summary

        response["weeks"].append(week_payload)

    if identifiers:
        response["league"] = {
            "leagueId": identifiers.league_id,
            "season": identifiers.season,
            "latestScoringPeriod": identifiers.latest_scoring_period,
            "currentMatchupPeriod": identifiers.current_matchup_period,
        }
    if request_summary:
        response["request"] = request_summary

    return response
