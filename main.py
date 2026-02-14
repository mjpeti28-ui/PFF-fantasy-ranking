# main.py  (print-only version)

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from data import load_rankings, build_lookups, load_rosters
from datetime import datetime, timezone
from optimizer import (
    flatten_league_names, optimize_lineups_first_pass, compute_worst_starter_bounds,
    optimize_lineups_second_pass
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
from config import PROJECTIONS_CSV, settings
from power_snapshots import build_snapshot_metadata

def hr(char="─", n=80):  # horizontal rule
    print(char * n)

def print_kv(title, kvs, key_hdr="Team", val_hdr="Value", sort=None, reverse=False, fmt=str):
    print(title); hr()
    items = list(kvs.items())
    if sort == "value":
        items = sorted(items, key=lambda x: x[1], reverse=reverse)
    elif sort == "key":
        items = sorted(items, key=lambda x: x[0], reverse=reverse)
    print(f"{key_hdr:<28} {val_hdr}")
    hr("—", 80)
    for k, v in items:
        print(f"{k:<28} {fmt(v)}")
    print()

def print_board(title, board, val_hdr, fmt=str):
    print(title); hr()
    print(f"{'#':<4}{'Team':<28} {val_hdr}")
    hr("—", 80)
    for i, (team, val) in enumerate(board, 1):
        print(f"{i:<4}{team:<28} {fmt(val)}")
    print()

def print_team_starters(team, starters, replacement_points):
    print(f"Starters — {team}"); hr()
    print(f"{'Slot/Nativity':<14} {'Name':<26} {'Proj':>7} {'VOR':>7} {'Overall':>7} {'PosRank':>8}")
    hr("—", 80)
    for p in sorted(starters, key=lambda x: (x['pos'], x['rank'])):
        name = p['name']
        pos = p['pos']
        proj = p.get('proj')
        repl = replacement_points.get(pos, 0.0)
        if proj is None:
            proj = repl
        vor = p.get('vor')
        if vor is None:
            vor = proj - repl
        rnk = p.get('rank')
        prank = p.get('posrank')
        rnk_s = str(rnk) if rnk is not None else "-"
        prank_s = str(prank) if prank is not None else "-"
        proj_s = f"{proj:.1f}" if proj is not None else "-"
        vor_s = f"{vor:.1f}" if vor is not None else "-"
        print(f"{pos:<14} {name:<26} {proj_s:>7} {vor_s:>7} {rnk_s:>7} {prank_s:>8}")
    print()

def print_team_bench(team, rows, topN=None, bench_beta: float | None = None):
    if bench_beta is None:
        bench_beta = settings.get("bench_ovar_beta")
    print(f"Bench — {team}  (beta={bench_beta})"); hr()
    print(f"{'Name':<26} {'Pos':<4} {'Proj':>7} {'VOR':>7} {'oVAR':>7} {'Overall':>7} {'PosRank':>8} {'BenchScore':>11}")
    hr("—", 80)
    shown = rows if topN is None else rows[:topN]
    for r in shown:
        proj = r.get('proj')
        vor = r.get('vor')
        ovar = r.get('oVAR')
        rank = r.get('rank')
        posrank = r.get('posrank')
        proj_s = f"{proj:7.1f}" if isinstance(proj, (int, float)) else f"{'-':>7}"
        vor_s = f"{vor:7.2f}" if isinstance(vor, (int, float)) else f"{'-':>7}"
        ovar_s = f"{ovar:7.0f}" if isinstance(ovar, (int, float)) else f"{'-':>7}"
        rank_s = f"{int(rank):7d}" if rank is not None else f"{'-':>7}"
        posrank_s = f"{str(posrank):>8}"
        print(f"{r['name']:<26} {r['pos']:<4} {proj_s} {vor_s} {ovar_s} {rank_s} {posrank_s} {r['BenchScore']:>11.2f}")
    print()


def print_zero_sum_group(title, group):
    entries = group.get("entries", [])
    if not entries:
        return
    total = group.get("total", 0.0)
    baseline = group.get("baseline", 0.0)
    surplus_sum = group.get("surplusSum", 0.0)
    share_sum = group.get("shareSum", 0.0)
    print(title)
    hr()
    print(f"{'Team':<28} {'Value':>10} {'Share%':>8} {'Surplus':>10}")
    hr("—", 80)
    for entry in entries:
        share_pct = entry.get("share", 0.0) * 100.0
        print(
            f"{entry['team']:<28} "
            f"{entry.get('value', 0.0):>10.2f} "
            f"{share_pct:>7.2f}% "
            f"{entry.get('surplus', 0.0):>10.2f}"
        )
    print(
        f"Total={total:.2f}  Baseline={baseline:.2f}  "
        f"ShareΣ={share_sum:.3f}  SurplusΣ={surplus_sum:.4f}"
    )
    print()


def print_zero_sum_section(zero_sum):
    if not zero_sum:
        return
    combined = zero_sum.get("combined", {})
    weights = combined.get("weights", {})
    w_s = weights.get("starters")
    w_b = weights.get("bench")
    title = "Zero-Sum Combined Ledger"
    if w_s is not None and w_b is not None:
        title += f" (weights starters={w_s:.2f} bench={w_b:.2f})"
    print_zero_sum_group(title, combined)
    print_zero_sum_group("Zero-Sum Starter Ledger", zero_sum.get("starters", {}))
    print_zero_sum_group("Zero-Sum Bench Ledger", zero_sum.get("bench", {}))

    positions = zero_sum.get("positions", {})
    for pos, group in sorted(positions.items()):
        print_zero_sum_group(f"Zero-Sum {pos} Starter Ledger", group)


def print_projection_sample(df, count=10):
    print("Sample Projection-Scaled Players"); hr()
    print(f"{'Rank':>5} {'Pos':<4} {'PosRank':>7} {'ProjPts':>9}  {'Name'}")
    hr("—", 80)
    sample = df.sort_values("Rank").head(count)
    for _, row in sample.iterrows():
        proj = row.get("ProjPoints")
        proj_str = f"{proj:.1f}" if proj is not None else "-"
        print(f"{int(row['Rank']):>5} {row['Position']:<4} {int(row['PosRank']):>7} {proj_str:>9}  {row['Name']}")
    print()

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_path(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = BASE_DIR / name
    return path


DEFAULT_RANKINGS = _resolve_data_path(os.getenv("RANKINGS_CSV", "ROS_week_2_PFF_rankings.csv"))
DEFAULT_PROJECTIONS = _resolve_data_path(PROJECTIONS_CSV)
DEFAULT_SUPPLEMENTAL_RANKINGS = _resolve_data_path(os.getenv("SUPPLEMENTAL_RANKINGS_CSV", "PFF_rankings.csv"))


def evaluate_league(
    rankings_path: str,
    *,
    projections_path: str | None = None,
    projection_scale_beta: float | None = None,
    replacement_skip_pct: float = 0.1,
    replacement_window: int = 3,
    bench_ovar_beta: float | None = None,
    combined_starters_weight: float | None = None,
    combined_bench_weight: float | None = None,
    bench_z_fallback_threshold: float | None = None,
    bench_percentile_clamp: float | None = None,
    scarcity_sample_step: float = 0.5,
    custom_rosters: Dict[str, Dict[str, List[str]]] | None = None,
    supplemental_path: str | None = None,
    save_snapshot: bool = True,
    snapshot_metadata: Optional[Dict[str, Any]] = None,
):
    default_beta = settings.get("projection_scale_beta")
    beta = default_beta if projection_scale_beta is None else projection_scale_beta

    if bench_ovar_beta is None:
        bench_ovar_beta = settings.get("bench_ovar_beta")
    if combined_starters_weight is None:
        combined_starters_weight = settings.get("combined_starters_weight")
    if combined_bench_weight is None:
        combined_bench_weight = settings.get("combined_bench_weight")
    if bench_z_fallback_threshold is None:
        bench_z_fallback_threshold = settings.get("bench_z_fallback_threshold")
    if bench_percentile_clamp is None:
        bench_percentile_clamp = settings.get("bench_percentile_clamp")

    if supplemental_path is None and DEFAULT_SUPPLEMENTAL_RANKINGS.exists():
        supplemental_path = str(DEFAULT_SUPPLEMENTAL_RANKINGS)

    df = load_rankings(
        rankings_path,
        projections_path,
        projection_scale_beta=projection_scale_beta,
        supplemental_path=supplemental_path,
    )
    rank_by_name, pos_by_name, posrank_by_name, proj_by_name = build_lookups(df)
    csv_max_per_pos = df.groupby("Position")["Rank"].max().to_dict()

    scarcity_curves, scarcity_samples = build_scarcity_curves(df, sample_step=scarcity_sample_step)

    rosters = load_rosters() if custom_rosters is None else copy.deepcopy(custom_rosters)
    league_names = flatten_league_names(rosters)
    alias_map = build_alias_map(league_names, df)

    first_pass = optimize_lineups_first_pass(
        rosters,
        alias_map,
        rank_by_name,
        pos_by_name,
        posrank_by_name,
        proj_by_name,
        csv_max_per_pos,
    )
    worst_starter_overall, worst_starter_posrank = compute_worst_starter_bounds(first_pass)

    results = optimize_lineups_second_pass(
        rosters,
        alias_map,
        rank_by_name,
        pos_by_name,
        posrank_by_name,
        proj_by_name,
        worst_starter_overall,
        worst_starter_posrank,
    )

    repl_counts = replacement_counts(results)
    repl_points, repl_targets = replacement_points(
        df,
        repl_counts,
        skip_pct=replacement_skip_pct,
        window=replacement_window,
        scarcity_curves=scarcity_curves,
    )
    starters_stat = starter_values(results, repl_points)
    starters_totals = {t: v["StarterVOR"] for t, v in starters_stat.items()}
    starter_projections = {t: v["StarterProjection"] for t, v in starters_stat.items()}
    for t, v in starters_stat.items():
        results[t]["StarterVOR"] = v["StarterVOR"]
        results[t]["StarterProjection"] = v["StarterProjection"]

    wb_overall, wb_posrank = compute_worst_bench_bounds(results)
    results = bench_generous_finalize(results, wb_overall, wb_posrank, repl_points)

    max_rank = int(df["Rank"].max()) if not df.empty else 0
    bench_tbls, bench_totals = bench_tables(
        results,
        repl_points,
        max_rank,
        beta=bench_ovar_beta,
    )

    starters_board, bench_board, combined_board, combined_scores = leaderboards(
        starters_totals,
        bench_totals,
        combined_starters_weight=combined_starters_weight,
        combined_bench_weight=combined_bench_weight,
        bench_z_fallback_threshold=bench_z_fallback_threshold,
        bench_percentile_clamp=bench_percentile_clamp,
    )

    zero_sum_view = compute_zero_sum_view(
        starters_totals,
        bench_totals,
        combined_starters_weight=combined_starters_weight,
        combined_bench_weight=combined_bench_weight,
        team_results=results,
        replacement_points=repl_points,
        bench_tables=bench_tbls,
    )

    settings_snapshot = {
        "projection_scale_beta": beta,
        "replacement_skip_pct": replacement_skip_pct,
        "replacement_window": replacement_window,
        "bench_ovar_beta": bench_ovar_beta,
        "combined_starters_weight": combined_starters_weight,
        "combined_bench_weight": combined_bench_weight,
        "bench_z_fallback_threshold": bench_z_fallback_threshold,
        "bench_percentile_clamp": bench_percentile_clamp,
        "scarcity_method": "monotone_cubic",
        "scarcity_positions": sorted(scarcity_curves.keys()),
        "scarcity_sample_step": scarcity_sample_step,
    }

    payload = {
        "df": df,
        "results": results,
        "repl_counts": repl_counts,
        "replacement_points": repl_points,
        "replacement_targets": repl_targets,
        "starters_totals": starters_totals,
        "starter_projections": starter_projections,
        "bench_tables": bench_tbls,
        "bench_totals": bench_totals,
        "leaderboards": {
            "starters": starters_board,
            "bench": bench_board,
            "combined": combined_board,
        },
        "zero_sum": zero_sum_view,
        "combined_scores": combined_scores,
        "max_rank": max_rank,
        "scarcity_curves": scarcity_curves,
        "scarcity_samples": scarcity_samples,
        "settings": settings_snapshot,
        "rosters": copy.deepcopy(rosters),
    }

    if save_snapshot:
        try:
            from power_snapshots import save_power_ranking_snapshot
        except Exception:
            # Snapshot persistence should never block evaluation results.
            pass
        else:
            snapshot_meta = dict(snapshot_metadata or {})
            snapshot_meta.setdefault("timestamp", datetime.now(timezone.utc))
            save_power_ranking_snapshot(
                leaderboard=payload["leaderboards"].get("combined"),
                starters_totals=starters_totals,
                bench_totals=bench_totals,
                starter_projections=starter_projections,
                metadata=snapshot_meta,
                settings=settings_snapshot,
                rankings_path=rankings_path,
                projections_path=projections_path,
                supplemental_path=supplemental_path,
            )

    return payload


def evaluate_trade_scenario(
    league_data: dict,
    team_a: str,
    team_b: str,
    send_a: List[str],
    send_b: List[str],
) -> Dict[str, float]:
    base_rosters = league_data.get("rosters")
    rosters = copy.deepcopy(base_rosters if base_rosters is not None else load_rosters())

    def adjust(team: str, outgoing: List[str], incoming: List[str]) -> None:
        roster = rosters.get(team)
        if roster is None:
            raise ValueError(f"Unknown team {team}")
        for name in outgoing:
            found = False
            for group, names in roster.items():
                if name in names:
                    names.remove(name)
                    found = True
                    break
            if not found:
                raise ValueError(f"{team} does not roster {name}")
        position_map = league_data.get("position_map")
        if position_map is None:
            df = league_data["df"]
            position_map = df.set_index("Name")["Position"].to_dict()
            league_data["position_map"] = position_map
        for name in incoming:
            pos = position_map.get(name)
            if pos is None:
                continue
            roster.setdefault(pos, []).append(name)

    adjust(team_a, send_a, send_b)
    adjust(team_b, send_b, send_a)

    snapshot_metadata = build_snapshot_metadata(
        "cli.trade.scenario",
        None,
        tags=["cli", "trade"],
        teamA=team_a,
        teamB=team_b,
    )

    scenario = evaluate_league(
        str(DEFAULT_RANKINGS),
        projections_path=str(DEFAULT_PROJECTIONS),
        custom_rosters=rosters,
        snapshot_metadata=snapshot_metadata,
    )
    combined = scenario["combined_scores"]
    baseline = league_data["combined_scores"]
    return {
        team_a: combined[team_a] - baseline[team_a],
        team_b: combined[team_b] - baseline[team_b],
    }


def main(rankings_path: str, projections_path: str | None = None, show_top_bench: int = 10, show_all_teams: bool = False):
    snapshot_metadata = build_snapshot_metadata(
        "cli.main",
        None,
        tags=["cli", "summary"],
        showTopBench=show_top_bench,
        showAllTeams=show_all_teams,
    )

    league = evaluate_league(
        rankings_path,
        projections_path=projections_path,
        snapshot_metadata=snapshot_metadata,
    )

    df = league["df"]
    results = league["results"]
    starters_totals = league["starters_totals"]
    starter_projections = league["starter_projections"]
    bench_tbls = league["bench_tables"]
    bench_totals = league["bench_totals"]
    combined_scores = league["combined_scores"]
    replacement_points_map = league["replacement_points"]
    leaderboards_map = league["leaderboards"]
    settings = league["settings"]

    # ──────────────────────────────────────────────────────────────────────────
    # Print everything
    hr("="); print("LEAGUE SUMMARY"); hr("=")
    print(
        "Scoring knobs: Bench oVAR beta = "
        f"{settings['bench_ovar_beta']}, Combined weights = "
        f"{settings['combined_starters_weight']}/{settings['combined_bench_weight']}, Projection beta = "
        f"{settings['projection_scale_beta']}"
    )
    print(
        "Replacement baseline window = "
        f"{settings['replacement_window']} (skip {settings['replacement_skip_pct']:.0%}), "
        "Bench z threshold = "
        f"{settings['bench_z_fallback_threshold']} clamp={settings['bench_percentile_clamp']}"
    )
    print()

    print_projection_sample(df)

    # League leaderboards
    print_board(
        "Starter VOR Leaderboard",
        leaderboards_map["starters"],
        "Starter VOR",
        fmt=lambda v: f"{v:.2f}"
    )
    print_board(
        "Bench Leaderboard",
        leaderboards_map["bench"],
        "Bench Score",
        fmt=lambda v: f"{v:.2f}"
    )
    print_board(
        "Combined Leaderboard",
        leaderboards_map["combined"],
        "Combined Score",
        fmt=lambda v: f"{v:.3f}"
    )

    # Team totals for quick scanning
    print_kv(
        "Starter VOR by Team",
        starters_totals,
        val_hdr="Starter VOR",
        sort="value",
        reverse=True,
        fmt=lambda v: f"{v:.2f}"
    )
    print_kv(
        "Starter Projections by Team",
        starter_projections,
        val_hdr="Starter Proj",
        sort="value",
        reverse=True,
        fmt=lambda v: f"{v:.1f}"
    )
    print_kv(
        "Bench Totals by Team",
        bench_totals,
        val_hdr="Bench Score",
        sort="value",
        reverse=True,
        fmt=lambda v: f"{v:.2f}"
    )
    print_kv(
        "Combined Scores by Team",
        combined_scores,
        val_hdr="Combined Score",
        sort="value",
        reverse=True,
        fmt=lambda v: f"{v:.3f}"
    )

    print_zero_sum_section(league.get("zero_sum"))

    # Detailed team readouts (combined leaderboard order)
    bench_limit = None if show_top_bench is None or show_top_bench <= 0 else show_top_bench
    ranked_teams = [team for team, _ in leaderboards_map["combined"]]
    detail_teams = ranked_teams if show_all_teams else ranked_teams[:5]

    for team in detail_teams:
        hr()
        print_team_starters(team, results[team]["starters"], replacement_points_map)
        print_team_bench(
            team,
            bench_tbls.get(team, []),
            topN=bench_limit,
            bench_beta=settings["bench_ovar_beta"],
        )

    if not show_all_teams and len(ranked_teams) > len(detail_teams):
        remaining = len(ranked_teams) - len(detail_teams)
        print(f"(Use --all-teams to see details for the remaining {remaining} team(s).)")

    return league


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank fantasy rosters using PFF rankings data.")
    parser.add_argument(
        "rankings",
        nargs="?",
        default=str(DEFAULT_RANKINGS),
        help=(
            "Path to the PFF rankings CSV file. Defaults to the packaged sample "
            f"at {DEFAULT_RANKINGS.name}."
        ),
    )
    parser.add_argument(
        "--projections",
        default=str(DEFAULT_PROJECTIONS),
        help=(
            "Path to the projections CSV file used for scaling ranks. "
            f"Defaults to {DEFAULT_PROJECTIONS.name}."
        ),
    )
    parser.add_argument(
        "--top-bench",
        type=int,
        default=10,
        help="Number of bench players to show per team (<=0 shows everyone).",
    )
    parser.add_argument(
        "--all-teams",
        action="store_true",
        default=False,
        help="Show starters/bench details for every team instead of just the top five.",
    )

    args = parser.parse_args()
    main(
        args.rankings,
        projections_path=args.projections,
        show_top_bench=args.top_bench,
        show_all_teams=args.all_teams,
    )
