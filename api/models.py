"""Pydantic schemas used by the API endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConfigResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    knobs: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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


class ZeroSumEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    value: float
    share: float
    surplus: float


class ZeroSumGroup(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total: float
    baseline: float
    share_sum: float = Field(..., alias="shareSum")
    surplus_sum: float = Field(..., alias="surplusSum")
    entries: List[ZeroSumEntry]


class ZeroSumCombinedGroup(ZeroSumGroup):
    weights: Dict[str, float]


class ZeroSumScarcityMetric(BaseModel):
    deficit: float
    pressure: float


class ZeroSumHerfindahl(BaseModel):
    starters: float = 0.0
    bench: float = 0.0
    slots: float = 0.0


class ZeroSumConcentrationRisk(BaseModel):
    starter_positions: Dict[str, float] = Field(default_factory=dict, alias="starterPositions")
    bench_positions: Dict[str, float] = Field(default_factory=dict, alias="benchPositions")
    slot_shares: Dict[str, float] = Field(default_factory=dict, alias="slotShares")
    flex_share: float = Field(default=0.0, alias="flexShare")
    herfindahl: ZeroSumHerfindahl = Field(default_factory=ZeroSumHerfindahl)


class ZeroSumTeamAnalytics(BaseModel):
    scarcity_pressure: Dict[str, ZeroSumScarcityMetric] = Field(default_factory=dict, alias="scarcityPressure")
    concentration_risk: ZeroSumConcentrationRisk = Field(alias="concentrationRisk")


class ZeroSumAnalyticsLeagueEntry(BaseModel):
    position: str
    aggregate_deficit: float = Field(..., alias="aggregateDeficit")


class ZeroSumAnalyticsLeague(BaseModel):
    high_pressure_positions: List[ZeroSumAnalyticsLeagueEntry] = Field(default_factory=list, alias="highPressurePositions")


class ZeroSumAnalytics(BaseModel):
    teams: Dict[str, ZeroSumTeamAnalytics] = Field(default_factory=dict)
    league: ZeroSumAnalyticsLeague = Field(default_factory=ZeroSumAnalyticsLeague)


class ZeroSumResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team_count: int = Field(..., alias="teamCount")
    starters: ZeroSumGroup
    bench: ZeroSumGroup
    combined: ZeroSumCombinedGroup
    positions: Dict[str, ZeroSumGroup] = Field(default_factory=dict)
    bench_positions: Dict[str, ZeroSumGroup] = Field(default_factory=dict, alias="benchPositions")
    slots: Dict[str, ZeroSumGroup] = Field(default_factory=dict)
    flex: Optional[ZeroSumGroup] = None
    analytics: ZeroSumAnalytics = Field(default_factory=ZeroSumAnalytics)


class ZeroSumShiftMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    share_delta: float = Field(..., alias="shareDelta")
    surplus_delta: float = Field(..., alias="surplusDelta")
    value_delta: float = Field(..., alias="valueDelta")


class ZeroSumShift(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    combined: Dict[str, ZeroSumShiftMetrics] = Field(default_factory=dict)
    starters: Dict[str, ZeroSumShiftMetrics] = Field(default_factory=dict)
    bench: Dict[str, ZeroSumShiftMetrics] = Field(default_factory=dict)
    positions: Dict[str, Dict[str, ZeroSumShiftMetrics]] = Field(default_factory=dict)
    bench_positions: Dict[str, Dict[str, ZeroSumShiftMetrics]] = Field(default_factory=dict, alias="benchPositions")
    slots: Dict[str, Dict[str, ZeroSumShiftMetrics]] = Field(default_factory=dict)
    flex: Dict[str, ZeroSumShiftMetrics] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)


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
    zero_sum: ZeroSumResponse = Field(..., alias="zeroSum")
    details: Optional[List[TeamDetail]] = None


class EvaluateDeltaSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    teams: List[TeamEvaluation]
    zero_sum: ZeroSumResponse = Field(..., alias="zeroSum")


class EvaluateDeltaResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    baseline: EvaluateDeltaSnapshot
    scenario: EvaluateDeltaSnapshot
    zero_sum_shift: ZeroSumShift = Field(..., alias="zeroSumShift")


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
    zero_sum_before: ZeroSumResponse = Field(..., alias="zeroSumBefore")
    zero_sum_after: ZeroSumResponse = Field(..., alias="zeroSumAfter")
    zero_sum_shift: ZeroSumShift = Field(..., alias="zeroSumShift")
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


class TeamLeverageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    combined: ZeroSumEntry
    starters: ZeroSumEntry
    bench: ZeroSumEntry
    positions: Dict[str, ZeroSumEntry] = Field(default_factory=dict)
    bench_positions: Dict[str, ZeroSumEntry] = Field(default_factory=dict, alias="benchPositions")
    slots: Dict[str, ZeroSumEntry] = Field(default_factory=dict)
    scarcity_pressure: Dict[str, ZeroSumScarcityMetric] = Field(default_factory=dict, alias="scarcityPressure")
    concentration_risk: ZeroSumConcentrationRisk = Field(alias="concentrationRisk")
    leverage_positions: List[str] = Field(default_factory=list, alias="leveragePositions")
    need_positions: List[str] = Field(default_factory=list, alias="needPositions")


class ZeroSumPositionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    position: str
    starters: ZeroSumGroup
    bench: Optional[ZeroSumGroup] = None
    analytics: Dict[str, Any] = Field(default_factory=dict)


class WaiverCandidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    position: Optional[str] = None
    team: Optional[str] = None
    rank: Optional[int] = None
    pos_rank: Optional[int] = Field(default=None, alias="posRank")
    proj_points: Optional[float] = Field(default=None, alias="projPoints")
    vor: Optional[float] = None
    bench_score: Optional[float] = Field(default=None, alias="benchScore")
    ovar: Optional[float] = Field(default=None, alias="oVAR")
    need_factor: Optional[float] = Field(default=None, alias="needFactor")


class WaiverListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: List[WaiverCandidate]
    total: int
    limit: int
    offset: int
    position_filter: Optional[str] = Field(default=None, alias="positionFilter")
    team_filter: Optional[str] = Field(default=None, alias="teamFilter")


class WaiverChange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    adds: List[str] = Field(default_factory=list)
    drops: List[str] = Field(default_factory=list)


class WaiverRecommendRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    changes: List[WaiverChange]
    include_details: bool = Field(default=True, alias="includeDetails")
    bench_limit: Optional[int] = Field(default=None, alias="benchLimit")


class WaiverTeamResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: str
    baseline: float
    post_change: float = Field(..., alias="postChange")
    delta: float


class WaiverRecommendResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    teams: List[WaiverTeamResult]
    combined_scores: Dict[str, float] = Field(..., alias="combinedScores")
    leaderboards: Dict[str, List[LeaderboardEntry]]
    details: Optional[List[TeamDetail]] = None


class DataTableResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(..., alias="jobId")
    job_type: str = Field(..., alias="jobType")
    status: JobStatus
    created_at: datetime = Field(..., alias="createdAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobCreatedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(..., alias="jobId")
    status: JobStatus
    job_type: str = Field(..., alias="jobType")
    poll_url: str = Field(..., alias="pollUrl")


class PlayerComparisonRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    players: List[str]
    include_stats: bool = Field(default=True, alias="includeStats")
    include_projections: bool = Field(default=True, alias="includeProjections")
    include_aliases: bool = Field(default=False, alias="includeAliases")


class PlayerOwnership(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team: Optional[str] = None
    roster_slot: Optional[str] = Field(default=None, alias="rosterSlot")
    raw_name: Optional[str] = Field(default=None, alias="rawName")
    is_ir: bool = Field(default=False, alias="isIR")
    is_free_agent: bool = Field(default=False, alias="isFreeAgent")


class PlayerComparison(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str
    canonical: Optional[str] = None
    matches: List[str] = Field(default_factory=list)
    position: Optional[str] = None
    team: Optional[str] = None
    rank: Optional[int] = None
    pos_rank: Optional[int] = Field(default=None, alias="posRank")
    proj_points: Optional[float] = Field(default=None, alias="projPoints")
    proj_z: Optional[float] = Field(default=None, alias="projZ")
    vor: Optional[float] = None
    ownership: Optional[PlayerOwnership] = None
    rankings: Dict[str, Any] = Field(default_factory=dict)
    projections: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    aliases: List[str] = Field(default_factory=list)
    notes: Dict[str, Any] = Field(default_factory=dict)


class PlayerComparisonResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: List[PlayerComparison]
    unresolved: List[str] = Field(default_factory=list)
