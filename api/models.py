"""Pydantic schemas used by the API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConfigResponse(BaseModel):
    knobs: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)


class LeagueMetadataResponse(BaseModel):
    num_teams: int = Field(..., alias="team_count")
    player_count: int
    last_reload: datetime
    rankings_path: str
    projections_path: Optional[str] = None
    supplemental_path: Optional[str] = None
    settings: Dict[str, Any]


class LeagueReloadRequest(BaseModel):
    rankings_path: Optional[str] = None
    projections_path: Optional[str] = None
    supplemental_path: Optional[str] = None
    projection_scale_beta: Optional[float] = None


class MessageResponse(BaseModel):
    message: str


class PlayerSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    position: str
    team: Optional[str] = None
    rank: Optional[int] = None
    pos_rank: Optional[int] = Field(default=None, alias="posRank")
    proj_points: Optional[float] = Field(default=None, alias="projPoints")
    proj_z: Optional[float] = Field(default=None, alias="projZ")


class PlayerListResponse(BaseModel):
    items: List[PlayerSummary]
    total: int
    limit: int
    offset: int
    metric: str


class PlayerDetail(PlayerSummary):
    model_config = ConfigDict(populate_by_name=True)

    rank_original: Optional[int] = Field(default=None, alias="rankOriginal")
    bye_week: Optional[int] = Field(default=None, alias="byeWeek")
    adp: Optional[float] = Field(default=None, alias="adp")
    auction_value: Optional[float] = Field(default=None, alias="auctionValue")
    proj_points_csv: Optional[float] = Field(default=None, alias="projPointsCsv")
    extras: Dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    projection_scale_beta: Optional[float] = Field(default=None, alias="projectionScaleBeta")
    replacement_skip_pct: Optional[float] = Field(default=None, alias="replacementSkipPct")
    replacement_window: Optional[int] = Field(default=None, alias="replacementWindow")
    bench_ovar_beta: Optional[float] = Field(default=None, alias="benchOvarBeta")
    combined_starters_weight: Optional[float] = Field(default=None, alias="combinedStartersWeight")
    combined_bench_weight: Optional[float] = Field(default=None, alias="combinedBenchWeight")
    bench_z_fallback_threshold: Optional[float] = Field(default=None, alias="benchZFallbackThreshold")
    bench_percentile_clamp: Optional[float] = Field(default=None, alias="benchPercentileClamp")
    scarcity_sample_step: Optional[float] = Field(default=None, alias="scarcitySampleStep")
    rankings_path: Optional[str] = Field(default=None, alias="rankingsPath")
    projections_path: Optional[str] = Field(default=None, alias="projectionsPath")
    supplemental_path: Optional[str] = Field(default=None, alias="supplementalPath")
    rosters: Optional[Dict[str, Dict[str, List[str]]]] = None
    include_details: bool = Field(default=False, alias="includeDetails")
    bench_limit: Optional[int] = Field(default=None, alias="benchLimit")


class TeamEvaluation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    combined_score: float = Field(..., alias="combinedScore")
    starter_vor: float = Field(..., alias="starterVOR")
    bench_score: float = Field(..., alias="benchScore")
    starter_projection: float = Field(..., alias="starterProjection")


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    value: float


class StarterDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    position: Optional[str] = Field(default=None, alias="position")
    csv_name: Optional[str] = Field(default=None, alias="csvName")
    rank: Optional[int] = None
    pos_rank: Optional[int] = Field(default=None, alias="posRank")
    projection: Optional[float] = None
    vor: Optional[float] = None


class BenchDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    position: Optional[str] = None
    rank: Optional[int] = None
    pos_rank: Optional[int] = Field(default=None, alias="posRank")
    projection: Optional[float] = None
    vor: Optional[float] = None
    ovar: Optional[float] = Field(default=None, alias="oVAR")
    bench_score: Optional[float] = Field(default=None, alias="benchScore")


class TeamDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    starters: List[StarterDetail]
    bench: List[BenchDetail]
    bench_limit: Optional[int] = Field(default=None, alias="benchLimit")


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    player_count: int = Field(..., alias="playerCount")
    teams: List[TeamEvaluation]
    leaderboards: Dict[str, List[LeaderboardEntry]]
    settings: Dict[str, Any]
    replacement_points: Dict[str, float] = Field(..., alias="replacementPoints")
    replacement_targets: Dict[str, float] = Field(..., alias="replacementTargets")
    scarcity_samples: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, alias="scarcitySamples")
    details: Optional[List[TeamDetail]] = None


class TradePiece(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    group: str
    name: str


class TradeEvaluateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team_a: str = Field(..., alias="teamA")
    team_b: str = Field(..., alias="teamB")
    send_a: List[TradePiece] = Field(default_factory=list, alias="sendA")
    send_b: List[TradePiece] = Field(default_factory=list, alias="sendB")
    include_details: bool = Field(default=True, alias="includeDetails")
    bench_limit: Optional[int] = Field(default=None, alias="benchLimit")


class TradeTeamResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    baseline: float
    post_trade: float = Field(..., alias="postTrade")
    delta: float


class TradeEvaluateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    teams: List[TradeTeamResult]
    combined_scores: Dict[str, float] = Field(..., alias="combinedScores")
    replacement_points: Dict[str, float] = Field(..., alias="replacementPoints")
    replacement_targets: Dict[str, float] = Field(..., alias="replacementTargets")
    starter_vor: Dict[str, float] = Field(..., alias="starterVOR")
    bench_totals: Dict[str, float] = Field(..., alias="benchTotals")
    leaderboards: Dict[str, List[LeaderboardEntry]]
    details: Optional[List[TeamDetail]] = None


class TradeFindRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team_a: str = Field(..., alias="teamA")
    team_b: str = Field(..., alias="teamB")
    max_players: int = Field(3, alias="maxPlayers")
    player_pool: int = Field(15, alias="playerPool")
    top_results: int = Field(5, alias="topResults")
    top_bench: int = Field(5, alias="topBench")
    min_gain_a: float = Field(0.0, alias="minGainA")
    max_loss_b: float = Field(0.25, alias="maxLossB")
    prune_margin: float = Field(0.05, alias="pruneMargin")
    min_upper_bound: float = Field(-5.0, alias="minUpperBound")
    fairness_mode: str = Field("sum", alias="fairnessMode")
    fairness_self_bias: float = Field(0.6, alias="fairnessSelfBias")
    fairness_penalty_weight: float = Field(0.5, alias="fairnessPenaltyWeight")
    consolidation_bonus: float = Field(0.0, alias="consolidationBonus")
    drop_tax_factor: float = Field(0.5, alias="dropTaxFactor")
    acceptance_fairness_weight: float = Field(0.4, alias="acceptanceFairnessWeight")
    acceptance_need_weight: float = Field(0.35, alias="acceptanceNeedWeight")
    acceptance_star_weight: float = Field(0.25, alias="acceptanceStarWeight")
    acceptance_need_scale: float = Field(1.0, alias="acceptanceNeedScale")
    star_vor_scale: float = Field(60.0, alias="starVorScale")
    drop_tax_acceptance_weight: float = Field(0.02, alias="dropTaxAcceptanceWeight")
    narrative_on: bool = Field(True, alias="narrativeOn")
    min_acceptance: float = Field(0.2, alias="minAcceptance")
    must_send_a: List[str] = Field(default_factory=list, alias="mustSendA")
    must_receive_b: List[str] = Field(default_factory=list, alias="mustReceiveB")
    include_details: bool = Field(default=True, alias="includeDetails")
    bench_limit: Optional[int] = Field(default=None, alias="benchLimit")


class TradeProposalTeamSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    delta: float
    combined_score: float = Field(..., alias="combinedScore")


class TradeProposal(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    send_a: List[TradePiece] = Field(..., alias="sendA")
    send_b: List[TradePiece] = Field(..., alias="sendB")
    receive_a: List[TradePiece] = Field(..., alias="receiveA")
    receive_b: List[TradePiece] = Field(..., alias="receiveB")
    combined_scores: Dict[str, float] = Field(..., alias="combinedScores")
    delta: Dict[str, float]
    score: float
    acceptance: float
    fairness_split: Optional[float] = Field(default=None, alias="fairnessSplit")
    drop_tax: Dict[str, float] = Field(default_factory=dict, alias="dropTax")
    star_gain: Dict[str, float] = Field(default_factory=dict, alias="starGain")
    narrative: Dict[str, str] = Field(default_factory=dict)
    details: Optional[List[TeamDetail]] = None
    leaderboards: Optional[Dict[str, List[LeaderboardEntry]]] = None


class TradeFindResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    baseline_combined: Dict[str, float] = Field(..., alias="baselineCombined")
    proposals: List[TradeProposal]
