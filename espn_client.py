from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx


def _load_local_env() -> None:
    """Populate os.environ with values from .env files if present."""

    env_dir = Path(__file__).resolve().parent
    for filename in (".env.local", ".env"):
        path = env_dir / filename
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


_load_local_env()

ESPN_FANTASY_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games"
DEFAULT_SPORT_KEY = "ffl"  # NFL fantasy football
DEFAULT_LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID", "1891655995")
DEFAULT_SEASON = 2025
DEFAULT_VIEWS: Tuple[str, ...] = (
    "mTeam",
    "mRoster",
    "mSettings",
    "mMatchup",
    "mStandings",
)

POSITION_ID_MAP: Dict[int, str] = {
    0: "QB",
    1: "QB",
    2: "RB",
    3: "RB",
    4: "WR",
    5: "WR",
    6: "TE",
    7: "TE",
    8: "C",
    9: "G",
    10: "T",
    11: "OL",
    12: "DL",
    13: "LB",
    14: "DL",
    15: "DB",
    16: "DST",
    17: "K",
    18: "P",
    19: "HC",
}

PRIMARY_POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")

LINEUP_SLOT_MAP: Dict[int, str] = {
    0: "QB",
    1: "TQB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    8: "DT",
    9: "DE",
    10: "LB",
    11: "DL",
    12: "CB",
    13: "S",
    14: "DP",
    15: "D/ST",
    16: "DST",
    17: "K",
    18: "P",
    19: "HC",
    20: "BENCH",
    21: "IR",
    22: "IR",
    23: "FLEX",
    24: "FLEX",
    25: "FLEX",
    27: "TAXI",
}

IR_SLOT_IDS = {slot for slot, name in LINEUP_SLOT_MAP.items() if "IR" in name}
BENCH_SLOT_IDS = {20, 23, 24, 25}

logger = logging.getLogger(__name__)


def _slugify_team_name(name: str, fallback: str) -> str:
    clean = name.replace("’", "").replace("'", "")
    clean = re.sub(r"[^A-Za-z0-9]+", "_", clean)
    clean = clean.strip("_")
    return clean or fallback


@dataclass(frozen=True)
class ESPNConfig:
    league_id: str
    season: int
    sport_key: str = DEFAULT_SPORT_KEY
    espn_s2: Optional[str] = None
    swid: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ESPNConfig":
        league_id = os.getenv("ESPN_LEAGUE_ID", DEFAULT_LEAGUE_ID)
        season_raw = os.getenv("ESPN_SEASON")
        season = DEFAULT_SEASON
        if season_raw:
            try:
                season = int(season_raw)
            except ValueError:
                logger.warning("Invalid ESPN_SEASON '%s'; defaulting to %s", season_raw, DEFAULT_SEASON)
        sport_key = os.getenv("ESPN_SPORT_KEY", DEFAULT_SPORT_KEY)
        espn_s2 = os.getenv("ESPN_S2") or os.getenv("espn_s2")
        swid = os.getenv("ESPN_SWID") or os.getenv("swid")

        return cls(
            league_id=league_id,
            season=season,
            sport_key=sport_key,
            espn_s2=espn_s2,
            swid=swid,
        )


@dataclass(frozen=True)
class ESPNRosterPlayer:
    player_id: Optional[int]
    full_name: str
    position: Optional[str]
    default_position_id: Optional[int]
    lineup_slot_id: Optional[int]
    lineup_slot: Optional[str]
    pro_team_id: Optional[int]
    injury_status: Optional[str]
    is_injured: bool
    on_ir: bool
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ESPNTeamData:
    team_id: int
    slug: str
    name: str
    abbreviation: Optional[str]
    owners: List[str] = field(default_factory=list)
    primary_owner: Optional[str] = None
    managers: List[str] = field(default_factory=list)
    record: Dict[str, Any] = field(default_factory=dict)
    points_for: Optional[float] = None
    points_against: Optional[float] = None
    playoff_seed: Optional[int] = None
    roster: List[ESPNRosterPlayer] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ESPNMatchup:
    matchup_id: Optional[int]
    matchup_period: Optional[int]
    home_team_id: Optional[int]
    away_team_id: Optional[int]
    home_points: Optional[float]
    away_points: Optional[float]
    winner: Optional[str]
    is_playoff: Optional[bool]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ESPNLeagueData:
    league_id: str
    season: int
    scoring_period_id: Optional[int]
    status: Dict[str, Any]
    settings: Dict[str, Any]
    teams: Dict[int, ESPNTeamData]
    team_slugs: Dict[str, int]
    schedule: List[ESPNMatchup]
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_roster_map(self) -> Dict[str, Dict[str, List[str]]]:
        rosters: Dict[str, Dict[str, List[str]]] = {}
        for team in self.teams.values():
            team_roster: Dict[str, List[str]] = {pos: [] for pos in PRIMARY_POSITIONS}
            for entry in team.roster:
                position = entry.position or POSITION_ID_MAP.get(entry.default_position_id or -1)
                if position not in PRIMARY_POSITIONS:
                    continue
                label = entry.full_name
                if entry.on_ir:
                    label = f"{label} (IR)"
                bucket = team_roster.setdefault(position, [])
                if label not in bucket:
                    bucket.append(label)
            for pos in PRIMARY_POSITIONS:
                team_roster.setdefault(pos, [])
            rosters[team.slug] = team_roster
        return rosters


class ESPNClient:
    """Minimal ESPN Fantasy API client."""

    def __init__(self, config: ESPNConfig) -> None:
        self.config = config
        base = f"{ESPN_FANTASY_BASE}/{config.sport_key}"
        if config.season < 2018:
            self.league_url = f"{base}/leagueHistory/{config.league_id}"
            self.league_params = {"seasonId": config.season}
        else:
            self.league_url = (
                f"{base}/seasons/{config.season}/segments/0/leagues/{config.league_id}"
            )
            self.league_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Low-level HTTP helpers
    # ------------------------------------------------------------------ #
    @functools.cached_property
    def _cookie_header(self) -> Optional[str]:
        if not self.config.espn_s2 or not self.config.swid:
            return None
        return f"espn_s2={self.config.espn_s2}; SWID={self.config.swid}"

    def _request(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        extend: str = "",
    ) -> Dict[str, Any]:
        url = f"{self.league_url}{extend}"
        query: Dict[str, Any] = dict(self.league_params)
        if params:
            query.update({k: v for k, v in params.items() if v is not None})

        request_headers = {"User-Agent": "PFF-Fantasy-Ranking/espn-client"}
        if headers:
            request_headers.update(headers)
        if self._cookie_header:
            request_headers["Cookie"] = self._cookie_header

        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=query, headers=request_headers)

        if response.status_code == 401:
            raise PermissionError("Unauthorized: validate ESPN_S2/SWID cookies.")
        if response.status_code == 404:
            raise FileNotFoundError(
                f"League {self.config.league_id} not found for season {self.config.season}."
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return payload[0]
        return payload

    def fetch_league_views(self, views: Sequence[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if views:
            params["view"] = list(views)
        return self._request(params=params)

    def fetch_endpoint(
        self,
        *,
        view: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        extend: str = "",
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = dict(params or {})
        if view:
            query["view"] = list(view)
        return self._request(params=query, headers=headers, extend=extend)

    def fetch_player_detail(
        self,
        player_id: int,
        *,
        view: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{ESPN_FANTASY_BASE}/{self.config.sport_key}/seasons/{self.config.season}/players/{player_id}"
        query: Dict[str, Any] = dict(params or {})
        if view:
            query["view"] = list(view)

        request_headers = {"User-Agent": "PFF-Fantasy-Ranking/espn-client"}
        if self._cookie_header:
            request_headers["Cookie"] = self._cookie_header

        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=query, headers=request_headers)

        if response.status_code == 401:
            raise PermissionError("Unauthorized: validate ESPN_S2/SWID cookies.")
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # League helpers
    # ------------------------------------------------------------------ #
    def fetch_league_full(self) -> Dict[str, Any]:
        return self.fetch_league_views(DEFAULT_VIEWS)

    def fetch_team_rosters(self) -> Dict[str, Any]:
        return self.fetch_league_views(["mTeam", "mRoster"])


_LAST_LEAGUE_PAYLOAD: Optional[Dict[str, Any]] = None
_LAST_LEAGUE_STATE: Optional[ESPNLeagueData] = None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_member_directory(payload: Dict[str, Any]) -> Dict[str, str]:
    directory: Dict[str, str] = {}
    for member in payload.get("members", []):
        member_id = member.get("id")
        if member_id is None:
            continue
        parts = [
            member.get("displayName"),
            " ".join(filter(None, [member.get("firstName"), member.get("lastName")])).strip(),
            member.get("nickname"),
        ]
        name = next((part for part in parts if isinstance(part, str) and part.strip()), None)
        directory[str(member_id)] = name.strip() if isinstance(name, str) else str(member_id)
    return directory


def _resolve_owner_name(owner_id: Any, members: Dict[str, str]) -> Optional[str]:
    if owner_id is None:
        return None
    key = str(owner_id)
    return members.get(key, key)


def _assign_unique_slug(name: str, team_id: int, existing: Dict[str, int]) -> str:
    base = _slugify_team_name(name, fallback=f"Team_{team_id}")
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def _parse_roster_entries(team_payload: Dict[str, Any]) -> List[ESPNRosterPlayer]:
    entries_payload = (team_payload.get("roster") or {}).get("entries", []) or []
    entries: List[ESPNRosterPlayer] = []
    for entry in entries_payload:
        player_pool = entry.get("playerPoolEntry") or {}
        player = player_pool.get("player") or {}
        player_id = _coerce_int(player.get("id"))
        full_name = player.get("fullName") or player.get("name") or "Unknown Player"
        default_position_id = _coerce_int(player.get("defaultPositionId"))
        position = POSITION_ID_MAP.get(default_position_id) if default_position_id is not None else None
        lineup_slot_id = _coerce_int(entry.get("lineupSlotId"))
        lineup_slot = LINEUP_SLOT_MAP.get(lineup_slot_id) if lineup_slot_id is not None else None
        pro_team_id = _coerce_int(player.get("proTeamId"))
        injury_status = (
            player.get("injuryStatus")
            or entry.get("injuryStatus")
            or entry.get("status")
        )
        is_injured = bool(player.get("injured"))
        on_ir = bool(lineup_slot_id in IR_SLOT_IDS)
        entries.append(
            ESPNRosterPlayer(
                player_id=player_id,
                full_name=str(full_name),
                position=position,
                default_position_id=default_position_id,
                lineup_slot_id=lineup_slot_id,
                lineup_slot=lineup_slot,
                pro_team_id=pro_team_id,
                injury_status=injury_status,
                is_injured=is_injured,
                on_ir=on_ir,
                raw=entry,
            )
        )
    return entries


def _parse_team(
    team_payload: Dict[str, Any],
    members: Dict[str, str],
    slug_registry: Dict[str, int],
) -> Optional[ESPNTeamData]:
    team_id = _coerce_int(team_payload.get("id"))
    if team_id is None:
        return None
    name = team_payload.get("name") or f"Team {team_id}"
    slug = _assign_unique_slug(name, team_id, slug_registry)
    abbreviation = team_payload.get("abbrev")
    owner_ids = team_payload.get("owners") or []
    owners = [owner_name for owner_id in owner_ids if (owner_name := _resolve_owner_name(owner_id, members))]
    primary_owner = _resolve_owner_name(team_payload.get("primaryOwner"), members)
    record_overall = (team_payload.get("record") or {}).get("overall") or {}
    points_for = _coerce_float(record_overall.get("pointsFor") or team_payload.get("points"))
    points_against = _coerce_float(record_overall.get("pointsAgainst") or team_payload.get("pointsAgainst"))
    playoff_seed = _coerce_int(team_payload.get("playoffSeed"))
    roster_entries = _parse_roster_entries(team_payload)
    managers = owners or []
    team_data = ESPNTeamData(
        team_id=team_id,
        slug=slug,
        name=str(name),
        abbreviation=str(abbreviation) if abbreviation else None,
        owners=owners,
        primary_owner=primary_owner,
        managers=managers,
        record=record_overall,
        points_for=points_for,
        points_against=points_against,
        playoff_seed=playoff_seed,
        roster=roster_entries,
        raw=team_payload,
    )
    slug_registry[slug] = team_id
    return team_data


def _parse_matchup(matchup_payload: Dict[str, Any]) -> ESPNMatchup:
    home = matchup_payload.get("home") or {}
    away = matchup_payload.get("away") or {}
    return ESPNMatchup(
        matchup_id=_coerce_int(matchup_payload.get("id")),
        matchup_period=_coerce_int(matchup_payload.get("matchupPeriodId")),
        home_team_id=_coerce_int(home.get("teamId")),
        away_team_id=_coerce_int(away.get("teamId")),
        home_points=_coerce_float(home.get("totalPoints")),
        away_points=_coerce_float(away.get("totalPoints")),
        winner=matchup_payload.get("winner"),
        is_playoff=bool(matchup_payload.get("playoffMatchup")),
        raw=matchup_payload,
    )


def _parse_league_payload(payload: Dict[str, Any]) -> ESPNLeagueData:
    members = _build_member_directory(payload)
    slug_registry: Dict[str, int] = {}
    teams: Dict[int, ESPNTeamData] = {}
    for team_payload in payload.get("teams", []) or []:
        parsed_team = _parse_team(team_payload, members, slug_registry)
        if parsed_team is not None:
            teams[parsed_team.team_id] = parsed_team
    schedule = [_parse_matchup(item) for item in payload.get("schedule", []) or []]
    league_id = str(payload.get("id") or DEFAULT_LEAGUE_ID)
    season = _coerce_int(payload.get("seasonId")) or DEFAULT_SEASON
    scoring_period_id = _coerce_int(payload.get("scoringPeriodId"))
    return ESPNLeagueData(
        league_id=league_id,
        season=season,
        scoring_period_id=scoring_period_id,
        status=payload.get("status", {}),
        settings=payload.get("settings", {}),
        teams=teams,
        team_slugs=slug_registry,
        schedule=schedule,
        raw=payload,
    )


def _default_client() -> Optional[ESPNClient]:
    config = ESPNConfig.from_environment()
    if not config.espn_s2 or not config.swid:
        return None
    return ESPNClient(config)


def fetch_league_state() -> ESPNLeagueData:
    client = _default_client()
    if client is None:
        raise RuntimeError("ESPN credentials not configured (ESPN_S2 / ESPN_SWID).")

    league_payload = client.fetch_league_full()
    league_state = _parse_league_payload(league_payload)

    global _LAST_LEAGUE_PAYLOAD
    global _LAST_LEAGUE_STATE
    _LAST_LEAGUE_PAYLOAD = league_payload
    _LAST_LEAGUE_STATE = league_state
    return league_state


def fetch_rosters_from_espn() -> Tuple[Dict[str, Dict[str, List[str]]], ESPNLeagueData]:
    """Fetch team rosters from ESPN and convert to the internal structure."""

    league_state = fetch_league_state()
    rosters = convert_league_to_rosters(league_state)
    return rosters, league_state


def get_last_league_payload() -> Optional[Dict[str, Any]]:
    return _LAST_LEAGUE_PAYLOAD


def get_last_league_state() -> Optional[ESPNLeagueData]:
    return _LAST_LEAGUE_STATE


def fetch_player_detail(player_id: int, *, view: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    client = _default_client()
    if client is None:
        return None
    return client.fetch_player_detail(player_id, view=view)


def convert_league_to_rosters(
    league: ESPNLeagueData | Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    if isinstance(league, ESPNLeagueData):
        return league.to_roster_map()
    parsed = _parse_league_payload(league)
    return parsed.to_roster_map()
