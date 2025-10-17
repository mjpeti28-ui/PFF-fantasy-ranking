from copy import deepcopy
from itertools import combinations, count
from pathlib import Path
from heapq import heappush, heappushpop
import multiprocessing as mp
import multiprocessing.pool as mppool
import os
from typing import Optional, Callable, List

import atexit

from config import PROJECTIONS_CSV, SCORABLE_POS, settings
from data import load_rankings, build_lookups, load_rosters
from optimizer import (
    flatten_league_names,
    optimize_lineups_first_pass,
    compute_worst_starter_bounds,
    optimize_lineups_second_pass,
)
from alias import build_alias_map
from scoring import (
    starter_values,
    replacement_counts,
    replacement_points,
    compute_worst_bench_bounds,
    bench_generous_finalize,
    bench_tables,
    leaderboards,
    build_scarcity_curves,
    compute_zero_sum_view,
)
from main import (
    hr,
    print_board,
    print_kv,
    print_team_starters,
    print_team_bench,
    DEFAULT_RANKINGS,
    DEFAULT_PROJECTIONS,
    DEFAULT_SUPPLEMENTAL_RANKINGS,
)


_WORKER_FINDER: Optional["TradeFinder"] = None
_PROCESS_POOL: Optional[mppool.Pool] = None
_PROCESS_POOL_KEY: Optional[tuple] = None


def _close_pool() -> None:
    global _PROCESS_POOL, _PROCESS_POOL_KEY
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.close()
        _PROCESS_POOL.join()
        _PROCESS_POOL = None
    _PROCESS_POOL_KEY = None


atexit.register(_close_pool)


def _worker_init(rankings_path: str, projections_path: str, rosters_snapshot: dict) -> None:
    global _WORKER_FINDER
    finder = TradeFinder(rankings_path, projections_path, build_baseline=False)
    finder.rosters = deepcopy(rosters_snapshot)
    finder.team_targets = {
        team: finder._team_player_count(roster)
        for team, roster in finder.rosters.items()
    }
    finder.team_groups = {
        team: {grp for grp in roster if grp in SCORABLE_POS}
        for team, roster in finder.rosters.items()
    }
    names = [
        name
        for name in flatten_league_names(finder.rosters)
        if not name.startswith("Replacement ")
    ]
    finder.alias_map = build_alias_map(names, finder.df)
    _WORKER_FINDER = finder


def _worker_eval(payload: tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]) -> dict:
    team_a, team_b, send_a, send_b = payload
    finder = _WORKER_FINDER
    if finder is None:
        raise RuntimeError("Worker not initialized")
    try:
        evaluation = finder.evaluate_trade(team_a, team_b, list(send_a), list(send_b), include_details=False)
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "team_a": team_a,
            "team_b": team_b,
        }
    combined_scores = evaluation["combined_scores"]
    return {
        "team_a": team_a,
        "team_b": team_b,
        "send_a": send_a,
        "send_b": send_b,
        "combined": {team_a: combined_scores[team_a], team_b: combined_scores[team_b]},
    }


def _get_pool(processes: int, rankings_path: str, projections_path: str, rosters_snapshot: dict) -> mppool.Pool:
    global _PROCESS_POOL, _PROCESS_POOL_KEY
    key = (processes, rankings_path, projections_path, id(rosters_snapshot))
    if _PROCESS_POOL is None or _PROCESS_POOL_KEY != key:
        _close_pool()
        ctx = mp.get_context("spawn")
        _PROCESS_POOL = ctx.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(rankings_path, projections_path, rosters_snapshot),
        )
        _PROCESS_POOL_KEY = key
    return _PROCESS_POOL


class TradeFinder:
    """Evaluate hypothetical trades between two teams using projection-based VOR."""

    def __init__(
        self,
        rankings_path: str | None = None,
        projections_path: str | None = None,
        *,
        build_baseline: bool = True,
    ) -> None:
        self.rankings_path = Path(rankings_path) if rankings_path else DEFAULT_RANKINGS
        projections_path = projections_path or str(DEFAULT_PROJECTIONS)
        self.projections_path = projections_path
        supplemental = (
            str(DEFAULT_SUPPLEMENTAL_RANKINGS)
            if DEFAULT_SUPPLEMENTAL_RANKINGS.exists()
            else None
        )
        self.df = load_rankings(
            str(self.rankings_path),
            projections_path,
            supplemental_path=supplemental,
        )
        (
            self.rank_by_name,
            self.pos_by_name,
            self.posrank_by_name,
            self.proj_by_name,
        ) = build_lookups(self.df)
        self.scarcity_curves, self.scarcity_samples = build_scarcity_curves(self.df)
        self.rosters = load_rosters()
        self.csv_max_per_pos = self.df.groupby("Position")["Rank"].max().to_dict()
        self.max_rank = int(self.df["Rank"].max()) if not self.df.empty else 0
        base_names = [
            name for name in flatten_league_names(self.rosters)
            if not name.startswith("Replacement ")
        ]
        self.alias_map = build_alias_map(base_names, self.df)
        self.team_targets = {
            team: self._team_player_count(roster)
            for team, roster in self.rosters.items()
        }
        self.team_groups = {
            team: {grp for grp in roster.keys() if grp in SCORABLE_POS}
            for team, roster in self.rosters.items()
        }
        self.placeholder_ids = count(1)
        self.baseline = None
        self.baseline_team_results = {}
        self.baseline_starter_vor = {}
        self.baseline_bench_totals = {}
        self.baseline_starter_proj = {}
        self.player_impacts: dict[str, dict[str, float]] = {}
        self.baseline_bench_vor: dict[str, dict[str, float]] = {}
        self.combined_starters_weight = settings.get("combined_starters_weight")
        self.combined_bench_weight = settings.get("combined_bench_weight")

        if build_baseline:
            self.baseline = self._evaluate(self.rosters, include_details=True)
            self.baseline_team_results = {
                team: deepcopy(res)
                for team, res in self.baseline["results"].items()
            }
            self.baseline_starter_vor = dict(self.baseline["starter_vor"])
            self.baseline_bench_totals = dict(self.baseline["bench_totals"])
            self.baseline_starter_proj = dict(self.baseline["starter_proj"])
            self.baseline_bench_vor = {
                team: {
                    row.get("name"): float(row.get("vor", 0.0))
                    for row in rows
                }
                for team, rows in self.baseline.get("bench_tables", {}).items()
            }
            self._build_player_impacts()

    # ───────────────────────────────────────────── Internal helpers ──
    def _team_player_count(self, roster: dict[str, list[str]]) -> int:
        return sum(len(names) for grp, names in roster.items() if grp in SCORABLE_POS)

    def _is_placeholder(self, name: str) -> bool:
        return name.startswith("Replacement ")

    def _make_placeholder(self, group: str) -> str:
        return f"Replacement {group} #{next(self.placeholder_ids)}"

    def _player_projection(self, name: str) -> float:
        proj = self.proj_by_name.get(name)
        return float(proj) if proj is not None else 0.0

    def _player_expected_vor(self, team: str, name: str) -> float:
        impacts = self.player_impacts.get(team, {})
        vor = impacts.get(name)
        if vor is not None:
            return float(vor)
        return 0.0

    def _compute_drop_tax(
        self,
        team: str,
        bench_tables: dict,
        traded_away: set[str],
    ) -> float:
        baseline_map = self.baseline_bench_vor.get(team, {})
        current_bench = {
            row.get("name")
            for row in bench_tables.get(team, [])
        }
        drop_sum = 0.0
        for name, vor in baseline_map.items():
            if vor <= 0:
                continue
            if name in traded_away:
                continue
            if name not in current_bench:
                drop_sum += vor
        return drop_sum

    def _objective_score(self, delta_a: float, delta_b: float, cfg: dict) -> float:
        mode = cfg.get("fairness_mode", "sum").lower()
        total = delta_a + delta_b
        if mode == "nash":
            if delta_a <= 0 or delta_b <= 0:
                return -abs(total)
            return delta_a * delta_b
        if mode == "weighted":
            if total <= 0:
                return total
            target = cfg.get("fairness_self_bias", 0.6)
            penalty_weight = cfg.get("fairness_penalty_weight", 0.5)
            split = delta_a / total if total else target
            penalty = penalty_weight * abs(split - target)
            return total - penalty
        return total

    def _build_narrative(
        self,
        team: str,
        delta_combined: float,
        starter_changes: float,
        received: list[tuple[str, str]],
        drop_tax: float,
        star_gain: float,
    ) -> str:
        positions = sorted({grp for grp, _ in received}) or ["depth"]
        pos_text = ", ".join(positions)
        direction = "boosts" if delta_combined >= 0 else "trims"
        pieces = [f"{direction} {pos_text}", f"ΔCombined {delta_combined:+.2f}"]
        if starter_changes:
            pieces.append(f"ΔStarter VOR {starter_changes:+.1f}")
        if star_gain:
            pieces.append(f"adds {star_gain:.1f} VOR of incoming talent")
        if drop_tax:
            pieces.append(f"may drop {drop_tax:.1f} bench VOR")
        return "; ".join(pieces)

    def _trade_metrics(
        self,
        team_a: str,
        team_b: str,
        send_a: list[tuple[str, str]],
        send_b: list[tuple[str, str]],
        delta_a: float,
        delta_b: float,
        evaluation: dict,
        scoring_cfg: dict,
    ) -> dict:
        bench_totals = evaluation.get("bench_totals", {})
        starter_vor = evaluation.get("starter_vor", {})
        bench_tables = evaluation.get("bench_tables", {})
        drop_tax_a = self._compute_drop_tax(team_a, bench_tables, {name for _, name in send_a})
        drop_tax_b = self._compute_drop_tax(team_b, bench_tables, {name for _, name in send_b})
        drop_penalty = scoring_cfg.get("drop_tax_factor", 0.5) * (drop_tax_a + drop_tax_b)

        score = self._objective_score(delta_a, delta_b, scoring_cfg)
        if len(send_a) > len(send_b) and delta_a > 0:
            score += scoring_cfg.get("consolidation_bonus", 0.0) * (len(send_a) - len(send_b))
        if len(send_b) > len(send_a) and delta_b > 0:
            score += scoring_cfg.get("consolidation_bonus", 0.0) * (len(send_b) - len(send_a))
        score -= drop_penalty

        total = delta_a + delta_b
        fairness_split = None
        if delta_a > 0 and delta_b > 0:
            fairness_split = delta_a / total if total else 0.5

        target_split = scoring_cfg.get("fairness_self_bias", 0.6)
        penalty_weight = scoring_cfg.get("fairness_penalty_weight", 0.5)
        fairness_accept = 0.0
        if total > 0 and delta_a > 0 and delta_b > 0:
            fairness_accept = max(0.0, 1.0 - penalty_weight * abs(fairness_split - target_split))
            fairness_accept = min(1.0, fairness_accept)

        need_scale = scoring_cfg.get("acceptance_need_scale", 1.0) or 1.0
        need_a = max(0.0, delta_a) / need_scale
        need_b = max(0.0, delta_b) / need_scale
        need_score = max(0.0, min(1.0, (need_a + need_b) / 2.0))

        star_gain_a = sum(
            max(0.0, self.player_impacts.get(team_b, {}).get(name, 0.0))
            for _, name in send_b
        )
        star_gain_b = sum(
            max(0.0, self.player_impacts.get(team_a, {}).get(name, 0.0))
            for _, name in send_a
        )
        star_scale = scoring_cfg.get("star_vor_scale", 60.0) or 1.0
        star_score = max(0.0, min(1.0, (star_gain_a + star_gain_b) / star_scale))

        acceptance = (
            scoring_cfg.get("acceptance_fairness_weight", 0.4) * fairness_accept
            + scoring_cfg.get("acceptance_need_weight", 0.35) * need_score
            + scoring_cfg.get("acceptance_star_weight", 0.25) * star_score
        )
        acceptance -= (drop_tax_a + drop_tax_b) * scoring_cfg.get("drop_tax_acceptance_weight", 0.02)
        acceptance = max(0.0, min(1.0, acceptance))

        narrative = {}
        if scoring_cfg.get("narrative_on", True):
            delta_starter_a = starter_vor.get(team_a, 0.0) - self.baseline_starter_vor.get(team_a, 0.0)
            delta_starter_b = starter_vor.get(team_b, 0.0) - self.baseline_starter_vor.get(team_b, 0.0)
            narrative[team_a] = self._build_narrative(
                team_a,
                delta_a,
                delta_starter_a,
                send_b,
                drop_tax_a,
                star_gain_a,
            )
            narrative[team_b] = self._build_narrative(
                team_b,
                delta_b,
                delta_starter_b,
                send_a,
                drop_tax_b,
                star_gain_b,
            )

        return {
            "score": score,
            "acceptance": acceptance,
            "fairness_split": fairness_split,
            "narrative": narrative,
            "drop_tax": {team_a: drop_tax_a, team_b: drop_tax_b},
            "star_gain": {team_a: star_gain_a, team_b: star_gain_b},
        }

    def _build_player_impacts(self) -> None:
        baseline = self.baseline
        if baseline is None:
            self.player_impacts = {}
            return
        impacts: dict[str, dict[str, float]] = {}
        bench_tables = baseline.get("bench_tables", {})
        for team, res in baseline["results"].items():
            team_impacts: dict[str, float] = {}
            for player in res.get("starters", []):
                name = player.get("name")
                if not name:
                    continue
                vor = float(player.get("vor", 0.0))
                team_impacts[name] = vor
            for row in bench_tables.get(team, []):
                name = row.get("name")
                if not name:
                    continue
                vor = float(row.get("vor", 0.0))
                if vor > team_impacts.get(name, 0.0):
                    team_impacts[name] = vor
            impacts[team] = team_impacts
        self.player_impacts = impacts

    def _ensure_group_min(self, team: str, roster: dict[str, list[str]]) -> None:
        groups = self.team_groups.get(team, set())
        for grp in groups:
            names = roster.get(grp)
            if names is None:
                roster[grp] = [self._make_placeholder(grp)]
            elif not names:
                roster[grp].append(self._make_placeholder(grp))

    def _cleanup_placeholders(self, roster: dict[str, list[str]]) -> None:
        for grp, names in roster.items():
            if grp not in SCORABLE_POS:
                continue
            if len(names) > 1:
                real = [n for n in names if not self._is_placeholder(n)]
                if real:
                    roster[grp] = real

    def _drop_worst(self, team: str, roster: dict[str, list[str]]) -> bool:
        candidates = []
        fallbacks = []
        for grp, names in roster.items():
            if grp not in SCORABLE_POS or not names:
                continue
            for name in names:
                score = self._player_projection(name)
                entry = (score, grp, name, len(names))
                if len(names) > 1:
                    candidates.append(entry)
                else:
                    fallbacks.append(entry)
        pool = candidates or fallbacks
        if not pool:
            return False
        score, grp, name, _ = min(pool, key=lambda x: (x[0], x[2]))
        roster[grp].remove(name)
        if not roster[grp]:
            roster[grp].append(self._make_placeholder(grp))
        return True

    def _normalize_team(self, team: str, roster: dict[str, list[str]]) -> None:
        self._ensure_group_min(team, roster)
        target = self.team_targets.get(team)
        total = self._team_player_count(roster)
        while target is not None and total > target:
            if not self._drop_worst(team, roster):
                break
            total = self._team_player_count(roster)
        self._cleanup_placeholders(roster)
        self._ensure_group_min(team, roster)

    def _remove_players(
        self,
        team: str,
        roster: dict[str, list[str]],
        players: list[tuple[str, str]],
    ) -> None:
        for grp, name in players:
            if grp not in roster:
                continue
            try:
                roster[grp].remove(name)
            except ValueError:
                continue
            if grp in SCORABLE_POS and not roster[grp]:
                roster[grp].append(self._make_placeholder(grp))

    def _add_players(self, roster: dict[str, list[str]], players: list[tuple[str, str]]) -> None:
        for grp, name in players:
            roster.setdefault(grp, []).append(name)
        self._cleanup_placeholders(roster)

    def _apply_trade(
        self,
        base_rosters: dict[str, dict[str, list[str]]],
        team_a: str,
        team_b: str,
        send_a: list[tuple[str, str]],
        send_b: list[tuple[str, str]],
    ) -> dict[str, dict[str, list[str]]]:
        rosters = deepcopy(base_rosters)
        roster_a = rosters[team_a]
        roster_b = rosters[team_b]

        self._remove_players(team_a, roster_a, send_a)
        self._remove_players(team_b, roster_b, send_b)

        self._add_players(roster_a, send_b)
        self._add_players(roster_b, send_a)

        self._cleanup_placeholders(roster_a)
        self._cleanup_placeholders(roster_b)

        self._normalize_team(team_a, roster_a)
        self._normalize_team(team_b, roster_b)

        for team in (team_a, team_b):
            for grp, names in rosters[team].items():
                for name in names:
                    base = name.replace(" (IR)", "")
                    self.alias_map.setdefault(base, None)
        return rosters

    def evaluate_trade(
        self,
        team_a: str,
        team_b: str,
        send_a: list[tuple[str, str]],
        send_b: list[tuple[str, str]],
        *,
        include_details: bool = True,
    ):
        trial_rosters = self._apply_trade(self.rosters, team_a, team_b, send_a, send_b)
        return self._evaluate(
            trial_rosters,
            include_details=include_details,
            changed_teams=[team_a, team_b],
        )

    def _select_players(
        self,
        team: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        roster = self.rosters[team]
        players = []
        fallback = []
        impacts = self.player_impacts.get(team, {})
        for grp, names in roster.items():
            if grp not in SCORABLE_POS:
                continue
            for name in names:
                if self._is_placeholder(name):
                    continue
                proj = self._player_projection(name)
                vor = impacts.get(name, 0.0)
                entry = (grp, name, proj, vor)
                fallback.append(entry)
                if vor > 0:
                    players.append(entry)
        if players:
            players.sort(key=lambda x: (x[3], x[2]), reverse=True)
        else:
            fallback.sort(key=lambda x: x[2], reverse=True)
            players = fallback
        if limit:
            players = players[:limit]
        return [(grp, name, proj) for grp, name, proj, _ in players]

    def _evaluate(
        self,
        rosters: dict[str, dict[str, list[str]]],
        *,
        include_details: bool = True,
        changed_teams: Optional[list[str]] = None,
    ):
        if changed_teams and not self.baseline_team_results:
            changed_teams = None

        alias_map = dict(self.alias_map)

        if not changed_teams:
            first_pass = optimize_lineups_first_pass(
                rosters,
                alias_map,
                self.rank_by_name,
                self.pos_by_name,
                self.posrank_by_name,
                self.proj_by_name,
                self.csv_max_per_pos,
            )
            worst_starter_overall, worst_starter_posrank = compute_worst_starter_bounds(first_pass)
            results = optimize_lineups_second_pass(
                rosters,
                alias_map,
                self.rank_by_name,
                self.pos_by_name,
                self.posrank_by_name,
                self.proj_by_name,
                worst_starter_overall,
                worst_starter_posrank,
            )
        else:
            subset = {team: rosters[team] for team in changed_teams}
            # Ensure alias map contains any new names (placeholders)
            for team in changed_teams:
                for grp, names in rosters[team].items():
                    for name in names:
                        base = name.replace(" (IR)", "")
                        alias_map.setdefault(base, None)
            first_pass = optimize_lineups_first_pass(
                subset,
                alias_map,
                self.rank_by_name,
                self.pos_by_name,
                self.posrank_by_name,
                self.proj_by_name,
                self.csv_max_per_pos,
            )
            worst_starter_overall, worst_starter_posrank = compute_worst_starter_bounds(first_pass)
            subset_results = optimize_lineups_second_pass(
                subset,
                alias_map,
                self.rank_by_name,
                self.pos_by_name,
                self.posrank_by_name,
                self.proj_by_name,
                worst_starter_overall,
                worst_starter_posrank,
            )
            results = {
                team: deepcopy(res)
                for team, res in self.baseline_team_results.items()
            }
            for team, res in subset_results.items():
                results[team] = res

        repl_counts = replacement_counts(results)
        repl_points, repl_targets = replacement_points(
            self.df,
            repl_counts,
            scarcity_curves=self.scarcity_curves,
        )
        starters_stat = starter_values(results, repl_points)
        starters_totals = {t: v["StarterVOR"] for t, v in starters_stat.items()}
        starter_proj = {t: v["StarterProjection"] for t, v in starters_stat.items()}
        for team, stats in starters_stat.items():
            results[team]["StarterVOR"] = stats["StarterVOR"]
            results[team]["StarterProjection"] = stats["StarterProjection"]

        wb_overall, wb_posrank = compute_worst_bench_bounds(results)
        results = bench_generous_finalize(results, wb_overall, wb_posrank, repl_points)
        bench_tbls, bench_totals = bench_tables(results, repl_points, self.max_rank)
        starters_board, bench_board, combined_board, combined_scores = leaderboards(starters_totals, bench_totals)

        zero_sum = compute_zero_sum_view(
            starters_totals,
            bench_totals,
            combined_starters_weight=self.combined_starters_weight,
            combined_bench_weight=self.combined_bench_weight,
            team_results=results,
            replacement_points=repl_points,
            bench_tables=bench_tbls,
        )

        data = {
            "combined_scores": combined_scores,
            "zero_sum": zero_sum,
        }

        if include_details:
            data.update(
                {
                    "results": results,
                    "starter_vor": starters_totals,
                    "starter_proj": starter_proj,
                    "bench_tables": bench_tbls,
                    "bench_totals": bench_totals,
                    "replacement_points": repl_points,
                    "replacement_targets": repl_targets,
                    "starters_board": starters_board,
                    "bench_board": bench_board,
                    "combined_board": combined_board,
                }
            )

        return data

    # ───────────────────────────────────────────── Trade search ──
    def find_trades(
        self,
        team_a: str,
        team_b: str,
        max_players: int = 2,
        player_pool: int = 8,
        top_results: int = 3,
        top_bench: int = 5,
        min_gain_a: float = 0.0,
        max_loss_b: float = 0.25,
        prune_margin: float = 0.05,
        min_upper_bound: float = -5.0,
        fairness_mode: str = "sum",
        fairness_self_bias: float = 0.6,
        fairness_penalty_weight: float = 0.5,
        consolidation_bonus: float = 0.0,
        drop_tax_factor: float = 0.5,
        acceptance_fairness_weight: float = 0.4,
        acceptance_need_weight: float = 0.35,
        acceptance_star_weight: float = 0.25,
        acceptance_need_scale: float = 1.0,
        star_vor_scale: float = 60.0,
        drop_tax_acceptance_weight: float = 0.02,
        narrative_on: bool = True,
        min_acceptance: float = 0.2,
        verbose: bool = True,
        show_progress: bool = True,
        must_receive_from_b: Optional[List[str]] = None,
        must_send_from_a: Optional[List[str]] = None,
    ) -> list[dict]:
        if team_a not in self.rosters or team_b not in self.rosters:
            raise ValueError("Both teams must exist in the league rosters.")

        pool_a = self._select_players(team_a, player_pool)
        pool_b = self._select_players(team_b, player_pool)
        if not pool_a or not pool_b:
            raise ValueError("One of the teams has no tradable players.")

        baseline = self.baseline
        if baseline is None:
            baseline = self._evaluate(self.rosters, include_details=True)
            self.baseline = baseline
            self.baseline_team_results = {
                team: deepcopy(res)
                for team, res in baseline["results"].items()
            }
            self.baseline_starter_vor = dict(baseline["starter_vor"])
            self.baseline_bench_totals = dict(baseline["bench_totals"])
            self.baseline_starter_proj = dict(baseline["starter_proj"])
            self.baseline_bench_vor = {
                team: {
                    row.get("name"): float(row.get("vor", 0.0))
                    for row in rows
                }
                for team, rows in baseline.get("bench_tables", {}).items()
            }
            self._build_player_impacts()
        baseline_combined = baseline["combined_scores"]

        fairness_mode_norm = fairness_mode.lower()
        effective_prune_margin = 0.0 if fairness_mode_norm == "nash" else prune_margin
        effective_max_loss_b = 0.0 if fairness_mode_norm == "nash" else max_loss_b
        scoring_config = {
            "fairness_mode": fairness_mode,
            "fairness_self_bias": fairness_self_bias,
            "fairness_penalty_weight": fairness_penalty_weight,
            "consolidation_bonus": consolidation_bonus,
            "drop_tax_factor": drop_tax_factor,
            "acceptance_fairness_weight": acceptance_fairness_weight,
            "acceptance_need_weight": acceptance_need_weight,
            "acceptance_star_weight": acceptance_star_weight,
            "acceptance_need_scale": acceptance_need_scale,
            "star_vor_scale": star_vor_scale,
            "drop_tax_acceptance_weight": drop_tax_acceptance_weight,
            "narrative_on": narrative_on,
            "min_acceptance": min_acceptance,
        }

        seen = set()
        tasks: list[tuple[float, tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]]] = []

        max_a = min(max_players, len(pool_a))
        max_b = min(max_players, len(pool_b))

        required_send = set(must_send_from_a or [])
        required_receive = set(must_receive_from_b or [])
        valid_required_send = {name for _, name, _ in pool_a} & required_send
        valid_required_receive = {name for _, name, _ in pool_b} & required_receive

        for size_a in range(1, max_a + 1):
            for size_b in range(1, max_b + 1):
                for combo_a in combinations(pool_a, size_a):
                    names_a = {name for _, name, _ in combo_a}
                    if valid_required_send and not valid_required_send.issubset(names_a):
                        continue
                    for combo_b in combinations(pool_b, size_b):
                        names_b = {name for _, name, _ in combo_b}
                        if valid_required_receive and not valid_required_receive.issubset(names_b):
                            continue
                        key = (
                            tuple(sorted(f"{grp}:{name}" for grp, name, _ in combo_a)),
                            tuple(sorted(f"{grp}:{name}" for grp, name, _ in combo_b)),
                        )
                        if key in seen:
                            continue
                        seen.add(key)

                        send_a = tuple((grp, name) for grp, name, _ in combo_a)
                        send_b = tuple((grp, name) for grp, name, _ in combo_b)

                        # Upper-bound estimation based on baseline impacts
                        potential_a = sum(
                            max(self.player_impacts.get(team_b, {}).get(name, 0.0), 0.0)
                            for _, name in send_b
                        ) - sum(
                            max(self.player_impacts.get(team_a, {}).get(name, 0.0), 0.0)
                            for _, name in send_a
                        )
                        potential_b = sum(
                            max(self.player_impacts.get(team_a, {}).get(name, 0.0), 0.0)
                            for _, name in send_a
                        ) - sum(
                            max(self.player_impacts.get(team_b, {}).get(name, 0.0), 0.0)
                            for _, name in send_b
                        )
                        upper_bound = potential_a + potential_b

                        margin = effective_prune_margin
                        if fairness_mode_norm == "nash":
                            if upper_bound < min_upper_bound:
                                continue
                        else:
                            if potential_a < (min_gain_a - margin):
                                continue
                            if potential_b < -(effective_max_loss_b + margin):
                                continue
                            if upper_bound < min_upper_bound:
                                continue

                        tasks.append((upper_bound, (team_a, team_b, send_a, send_b)))

        if not tasks:
            if verbose:
                print("No trade scenarios evaluated.")
            return []

        tasks.sort(reverse=True, key=lambda x: x[0])
        payloads = [payload for _, payload in tasks]

        available = max(1, (os.cpu_count() or 2) - 1)
        processes = min(len(payloads), available)
        processes = max(1, processes)
        chunksize = max(1, len(payloads) // (processes * 8))

        top_trades: list[tuple[float, int, dict]] = []
        counter = count()

        total_tasks = len(payloads)
        progress_step = max(1, total_tasks // 50)
        processed = 0

        if total_tasks and show_progress:
            print(
                f"Evaluating trades: 0/{total_tasks} (  0.0%)",
                end="",
                flush=True,
            )

        pool = _get_pool(processes, str(self.rankings_path), self.projections_path, self.rosters)

        for res in pool.imap_unordered(_worker_eval, payloads, chunksize):
            processed += 1
            if show_progress and (processed % progress_step == 0 or processed == total_tasks):
                percent = processed / total_tasks * 100
                print(
                    f"\rEvaluating trades: {processed}/{total_tasks} ({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
            if "error" in res:
                message = res.get("error", "Unknown error")
                opponent = res.get("team_b", team_b)
                print(f"\nSkipping {team_a} ↔ {opponent}: {message}")
                continue
            send_a = list(res["send_a"])
            send_b = list(res["send_b"])
            combined = res["combined"]
            delta_a = combined[team_a] - baseline_combined[team_a]
            delta_b = combined[team_b] - baseline_combined[team_b]

            if fairness_mode_norm == "nash" and (delta_a <= 0 or delta_b <= 0):
                continue
            if delta_a < min_gain_a or delta_b < -effective_max_loss_b:
                continue

            objective = self._objective_score(delta_a, delta_b, scoring_config)
            if fairness_mode_norm == "nash" and objective <= 0:
                continue

            data = {
                "team_a": team_a,
                "team_b": team_b,
                "send_a": send_a,
                "send_b": send_b,
                "receive_a": send_b,
                "receive_b": send_a,
                "combined": combined,
                "delta": {team_a: delta_a, team_b: delta_b},
            }

            entry = (objective, next(counter), data)
            if len(top_trades) < top_results:
                heappush(top_trades, entry)
            else:
                heappushpop(top_trades, entry)

        if processed and show_progress:
            print()

        if not top_trades:
            if verbose:
                print("No trade scenarios evaluated.")
            return []

        top_trades_sorted = sorted(top_trades, key=lambda x: x[0], reverse=True)

        if verbose:
            print()
            hr("=")
            print(f"Baseline Combined Scores for {team_a} & {team_b}")
            hr("=")
            print_kv(
                "Combined",
                {team_a: baseline_combined[team_a], team_b: baseline_combined[team_b]},
                val_hdr="Score",
                sort="value",
                reverse=True,
                fmt=lambda v: f"{v:.3f}",
            )

        results_out = []
        for _, _, data in top_trades_sorted:
            full_eval = self.evaluate_trade(
                data["team_a"],
                data["team_b"],
                data["send_a"],
                data["send_b"],
                include_details=True,
            )
            combined_full = full_eval["combined_scores"]
            delta = {
                data["team_a"]: combined_full[data["team_a"]] - baseline_combined[data["team_a"]],
                data["team_b"]: combined_full[data["team_b"]] - baseline_combined[data["team_b"]],
            }
            if fairness_mode_norm == "nash" and (delta[data["team_a"]] <= 0 or delta[data["team_b"]] <= 0):
                continue
            if delta[data["team_a"]] < min_gain_a or delta[data["team_b"]] < -effective_max_loss_b:
                continue
            data.update(
                {
                    "combined": combined_full,
                    "delta": delta,
                    "evaluation": full_eval,
                    "bench_totals": full_eval["bench_totals"],
                }
            )
            metrics = self._trade_metrics(
                data["team_a"],
                data["team_b"],
                data["send_a"],
                data["send_b"],
                delta[data["team_a"]],
                delta[data["team_b"]],
                full_eval,
                scoring_config,
            )
            data.update(metrics)
            if metrics.get("acceptance", 0.0) < scoring_config.get("min_acceptance", 0.0):
                continue
            results_out.append(data)
        if not results_out:
            if verbose:
                print("No trades met the requested thresholds.")
            return []

        results_out.sort(
            key=lambda d: (d.get("score", float("-inf")), d.get("acceptance", 0.0)),
            reverse=True,
        )

        if verbose:
            for idx, data in enumerate(results_out, start=1):
                self._print_trade(idx, data, baseline_combined, top_bench)
        return results_out

    def _format_players(self, players: list[tuple[str, str]]) -> str:
        return ", ".join(f"{name} ({grp})" for grp, name in players) if players else "None"

    def _print_trade(
        self,
        idx: int,
        data: dict,
        baseline_combined: dict[str, float],
        top_bench: int,
    ) -> None:
        team_a = data["team_a"]
        team_b = data["team_b"]
        send_a = data["send_a"]
        send_b = data["send_b"]
        combined = data["combined"]
        delta = data["delta"]
        evaluation = data["evaluation"]
        score = data.get("score", 0.0)
        acceptance = data.get("acceptance", 0.0)
        fairness_split = data.get("fairness_split", 0.5)
        narrative = data.get("narrative", {})
        drop_tax = data.get("drop_tax", {})
        star_gain = data.get("star_gain", {})

        hr("=")
        print(f"Trade #{idx}: {team_a} ↔ {team_b}")
        hr("=")
        print(f"{team_a} sends:     {self._format_players(send_a)}")
        print(f"{team_b} sends:     {self._format_players(send_b)}")
        print()
        print(f"{team_a} Combined: {baseline_combined[team_a]:.3f} → {combined[team_a]:.3f} ({delta[team_a]:+.3f})")
        print(f"{team_b} Combined: {baseline_combined[team_b]:.3f} → {combined[team_b]:.3f} ({delta[team_b]:+.3f})")
        split_display = f"{fairness_split:.2f}" if fairness_split is not None else "--"
        print(f"Score: {score:.3f} | Acceptance: {acceptance:.2f} | Split: {split_display}")
        if drop_tax:
            print(
                f"Drop tax (bench VOR lost): {team_a} {drop_tax.get(team_a, 0.0):.2f} | {team_b} {drop_tax.get(team_b, 0.0):.2f}"
            )
        if star_gain:
            print(
                f"Incoming star VOR: {team_a} {star_gain.get(team_a, 0.0):.1f} | {team_b} {star_gain.get(team_b, 0.0):.1f}"
            )
        if narrative:
            print(f"Pitch — {team_a}: {narrative.get(team_a, '')}")
            print(f"Pitch — {team_b}: {narrative.get(team_b, '')}")
        print()

        print_board(
            "Combined Leaderboard (Post-Trade)",
            evaluation["combined_board"],
            "Combined",
            fmt=lambda v: f"{v:.3f}",
        )
        replacement_points = evaluation["replacement_points"]
        bench_tables = evaluation["bench_tables"]
        results = evaluation["results"]

        for team in (team_a, team_b):
            hr()
            print_team_starters(team, results[team]["starters"], replacement_points)
            print_team_bench(team, bench_tables.get(team, []), topN=top_bench)


TEAM_A = "The_Little_Stinkers"
TEAM_B = "Hinesward_Jersey"

TRADE_CONFIG = {
    "max_players": 3,
    "player_pool": 17,
    "top_results": 7,
    "top_bench": 6,
    
    # show progress bars while evaluating trades
    "show_progress": True,
    
    # when TEAM_B="All", show full per-opponent trade details (True) or suppress until summary (False)
    "all_show_details": False,
    
    "min_gain_a": 0.0,      # minimum combined delta required for Team A
    "max_loss_b": 0.25,     # maximum combined drop tolerated for Team B
    "prune_margin": 0.5,   # slack for fairness pruning at candidate stage
    "min_upper_bound": -20.0,  # trades with optimistic total delta below this are skipped
    
    # scoring mode: 'sum' favors total surplus, 'nash' requires both sides to win, 'weighted' targets a split
    "fairness_mode": "weighted",
    
    # higher self bias lets Team A keep a larger share of surplus without penalty (weighted mode)
    "fairness_self_bias": 0.6,
    
    # increase to penalize imbalanced surplus splits more heavily
    "fairness_penalty_weight": 0.5,
    
    # extra reward for consolidating multiple pieces into a single upgrade
    "consolidation_bonus": 0.3,
    
    # bigger factor means sharper penalty when a team must cut productive bench players
    "drop_tax_factor": 0.3,
    
    # weight for fairness component in acceptance likelihood (raise to demand balanced trades)
    "acceptance_fairness_weight": 0.6,
    
    # weight for direct team improvement in acceptance likelihood (raise to favor need-based gains)
    "acceptance_need_weight": 0.5,
    
    # weight for star power when evaluating acceptance (raise if headline names sway decisions)
    "acceptance_star_weight": 0.25,
    
    # scale factor for translating combined delta into "need" satisfaction (lower = stricter)
    "acceptance_need_scale": 1.0,
    
    # scale for star VOR when computing star_score (lower makes star additions count more)
    "star_vor_scale": 60.0,

    # multiplier converting drop-tax VOR into acceptance penalty (raise to discourage forced drops)
    "drop_tax_acceptance_weight": 0.02,

    # toggle trade narratives on/off (True shows need-focused one-liners)
    "narrative_on": True,

    # minimum acceptance probability required to surface a trade (raise for safer suggestions)
    "min_acceptance": 0.2,
}

RANKINGS_PATH = str(DEFAULT_RANKINGS)
PROJECTIONS_PATH = str(DEFAULT_PROJECTIONS)


def main() -> None:
    finder = TradeFinder(RANKINGS_PATH, PROJECTIONS_PATH)
    show_progress = TRADE_CONFIG.get("show_progress", True)
    all_mode = TEAM_B.lower() == "all"
    opponents = [TEAM_B] if not all_mode else [team for team in finder.rosters.keys() if team != TEAM_A]
    detail_flag = TRADE_CONFIG.get("all_show_details", False) if all_mode else True
    aggregate_results = []
    for opp in opponents:
        if detail_flag:
            print()
            hr("=")
            print(f"Evaluating trades: {TEAM_A} ↔ {opp}")
            hr("=")
        else:
            print(f"\nScanning {TEAM_A} ↔ {opp}…")
        results = finder.find_trades(
            TEAM_A,
            opp,
            max_players=TRADE_CONFIG["max_players"],
            player_pool=TRADE_CONFIG["player_pool"],
            top_results=TRADE_CONFIG["top_results"],
            top_bench=TRADE_CONFIG["top_bench"],
            min_gain_a=TRADE_CONFIG["min_gain_a"],
            max_loss_b=TRADE_CONFIG["max_loss_b"],
            prune_margin=TRADE_CONFIG["prune_margin"],
            min_upper_bound=TRADE_CONFIG["min_upper_bound"],
            fairness_mode=TRADE_CONFIG["fairness_mode"],
            fairness_self_bias=TRADE_CONFIG["fairness_self_bias"],
            fairness_penalty_weight=TRADE_CONFIG["fairness_penalty_weight"],
            consolidation_bonus=TRADE_CONFIG["consolidation_bonus"],
            drop_tax_factor=TRADE_CONFIG["drop_tax_factor"],
            acceptance_fairness_weight=TRADE_CONFIG["acceptance_fairness_weight"],
            acceptance_need_weight=TRADE_CONFIG["acceptance_need_weight"],
            acceptance_star_weight=TRADE_CONFIG["acceptance_star_weight"],
            acceptance_need_scale=TRADE_CONFIG["acceptance_need_scale"],
            star_vor_scale=TRADE_CONFIG["star_vor_scale"],
            drop_tax_acceptance_weight=TRADE_CONFIG["drop_tax_acceptance_weight"],
            narrative_on=TRADE_CONFIG["narrative_on"],
            min_acceptance=TRADE_CONFIG["min_acceptance"],
            verbose=detail_flag,
            show_progress=show_progress,
        )
        if results:
            aggregate_results.extend(results)
        elif not detail_flag:
            print(f"No qualifying trades for {TEAM_A} ↔ {opp}.")

    if all_mode:
        if not aggregate_results:
            print("\nNo trades met the requested thresholds for any opponent.")
            return
        aggregate_results.sort(
            key=lambda d: (d.get("score", float("-inf")), d.get("acceptance", 0.0)),
            reverse=True,
        )
        top = aggregate_results[:TRADE_CONFIG["top_results"]]
        print()
        hr("=")
        print(f"Top aggregated trades for {TEAM_A} across league")
        hr("=")
        baseline_combined = finder.baseline["combined_scores"] if finder.baseline else {}
        for idx, data in enumerate(top, start=1):
            opp = data["team_b"]
            print(f"\nBest Trade #{idx} vs {opp}")
            finder._print_trade(idx, data, baseline_combined, TRADE_CONFIG["top_bench"])


if __name__ == "__main__":
    main()
