import math
import statistics as stats
from bisect import bisect_left, bisect_right
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from config import (
    SCORABLE_POS,
    COMBINED_STARTERS_WEIGHT,
    COMBINED_BENCH_WEIGHT,
    BENCH_OVAR_BETA,
    BENCH_Z_FALLBACK_THRESHOLD,
    BENCH_PERCENTILE_CLAMP,
)


def _build_monotone_cubic_spline(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    # Fritsch-Carlson monotone cubic interpolation to preserve ordering
    if len(x) == 1:
        return {"x": x.copy(), "y": y.copy(), "m": np.zeros_like(y)}

    delta = np.diff(y) / np.diff(x)
    m = np.zeros_like(y)
    m[0] = delta[0]
    m[-1] = delta[-1]
    if len(y) > 2:
        m[1:-1] = (delta[:-1] + delta[1:]) / 2.0

    for i in range(len(delta)):
        if delta[i] == 0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            denom = a * a + b * b
            if denom > 9.0:
                tau = 3.0 / math.sqrt(denom)
                m[i] = tau * a * delta[i]
                m[i + 1] = tau * b * delta[i]

    return {"x": x.copy(), "y": y.copy(), "m": m}


def _evaluate_monotone_cubic(spline: Dict[str, np.ndarray], x_val: float) -> float:
    x = spline["x"]
    y = spline["y"]
    m = spline["m"]
    if len(x) == 1:
        return float(y[0])

    if x_val <= x[0]:
        return float(y[0])
    if x_val >= x[-1]:
        return float(y[-1])

    idx = np.searchsorted(x, x_val) - 1
    idx = max(0, min(idx, len(x) - 2))
    h = x[idx + 1] - x[idx]
    if h == 0:
        return float(y[idx])
    t = (x_val - x[idx]) / h
    h00 = (2 * t ** 3) - (3 * t ** 2) + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = (-2 * t ** 3) + (3 * t ** 2)
    h11 = t ** 3 - t ** 2
    value = (
        h00 * y[idx]
        + h10 * h * m[idx]
        + h01 * y[idx + 1]
        + h11 * h * m[idx + 1]
    )
    return float(value)


def build_scarcity_curves(
    df: pd.DataFrame,
    *,
    sample_step: float = 0.5,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[Dict[str, float]]]]:
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    samples: Dict[str, List[Dict[str, float]]] = {}

    for pos in SCORABLE_POS:
        pos_df = (
            df[df["Position"] == pos]
            .dropna(subset=["ProjPoints"])
            .sort_values("ProjPoints", ascending=False)
        )
        if pos_df.empty:
            continue

        projections = pos_df["ProjPoints"].astype(float).to_numpy()
        slots = np.arange(1.0, len(projections) + 1.0)
        if len(slots) == 0:
            continue

        curves[pos] = _build_monotone_cubic_spline(slots, projections)

        max_slot = slots[-1]
        sample_slots = np.arange(1.0, max_slot + sample_step, sample_step)
        samples[pos] = [
            {
                "slot": round(float(s), 2),
                "projection": round(_evaluate_monotone_cubic(curves[pos], float(s)), 3),
            }
            for s in sample_slots
        ]

    return curves, samples


def starter_values(team_results, replacement_points: Dict[str, float]) -> Dict[str, Dict]:
    out = {}
    for t, res in team_results.items():
        total_proj = 0.0
        total_vor = 0.0
        cnt = 0
        for p in res["starters"]:
            pos = p.get("pos")
            if pos not in SCORABLE_POS:
                continue
            proj = p.get("proj")
            repl = replacement_points.get(pos, 0.0)
            if proj is None:
                proj = repl
            total_proj += proj
            total_vor += proj - repl
            cnt += 1
            p["vor"] = round(proj - repl, 2)
            p["proj"] = proj
        out[t] = {
            "StarterProjection": round(total_proj, 2),
            "StarterVOR": round(total_vor, 2),
            "StarterCount": cnt,
        }
    return out

def replacement_counts(team_results) -> Dict[str, int]:
    cnt = Counter(p["pos"] for team in team_results.values() for p in team["starters"] if p["pos"] in SCORABLE_POS)
    return {pos: cnt.get(pos, 0) for pos in SCORABLE_POS}


def replacement_points(
    df: pd.DataFrame,
    repl_counts: Dict[str, int],
    *,
    skip_pct: float = 0.1,
    window: int = 3,
    scarcity_curves: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bases: Dict[str, float] = {}
    targets: Dict[str, float] = {}
    skip_pct = max(0.0, skip_pct)
    window_size = max(1, int(window))
    if scarcity_curves is None:
        scarcity_curves, _ = build_scarcity_curves(df)

    for pos in SCORABLE_POS:
        count = max(repl_counts.get(pos, 0), 0)
        pos_df = (
            df[df["Position"] == pos]
            .dropna(subset=["ProjPoints"])
            .sort_values("ProjPoints", ascending=False)
        )
        if pos_df.empty:
            bases[pos] = 0.0
            targets[pos] = float(count)
            continue
        total_players = len(pos_df)
        skip_offset = skip_pct * total_players
        depth = max(1.0, float(window_size))
        target_slot = count + skip_offset + (depth - 1.0) / 2.0
        target_slot = float(min(total_players, max(1.0, target_slot)))
        targets[pos] = target_slot

        curve = scarcity_curves.get(pos)
        if curve:
            bases[pos] = _evaluate_monotone_cubic(curve, target_slot)
        else:
            idx_start = max(0, min(total_players - 1, int(round(target_slot)) - 1))
            idx_end = min(total_players, idx_start + window_size)
            window_vals = pos_df.iloc[idx_start:idx_end]["ProjPoints"].astype(float)
            bases[pos] = float(window_vals.mean()) if not window_vals.empty else float(pos_df.iloc[-1]["ProjPoints"])
    return bases, targets

def bench_generous_finalize(team_results, worst_bench_overall, worst_bench_posrank, replacement_points):
    # Apply generous missing for bench: worst bench +10/+1 (fallback to starter if none)
    for t, res in team_results.items():
        new_bench = []
        for p in res["bench"]:
            pos = p.get("pos")
            if pos in SCORABLE_POS and (p.get("csv") is None or p.get("rank") is None):
                rnk = (worst_bench_overall.get(pos) if worst_bench_overall.get(pos) is not None else 0) + 10
                prank = (worst_bench_posrank.get(pos) if worst_bench_posrank.get(pos) is not None else 0) + 1
                proj = p.get("proj")
                repl = replacement_points.get(pos, 0.0)
                if proj is None:
                    proj = repl
                p = {**p, "rank": rnk, "posrank": prank, "proj": proj}
            elif pos in SCORABLE_POS and p.get("proj") is None:
                repl = replacement_points.get(pos, 0.0)
                p = {**p, "proj": repl}
            new_bench.append(p)
        team_results[t]["bench"] = new_bench
    return team_results

def compute_worst_bench_bounds(team_results):
    # scan bench items with known ranks/posranks
    wb_overall = {}
    wb_posrank = {}
    for pos in SCORABLE_POS:
        rs = [p["rank"] for res in team_results.values() for p in res["bench"] if p["pos"]==pos and p.get("rank") is not None]
        prs = [p.get("posrank") for res in team_results.values() for p in res["bench"] if p["pos"]==pos and p.get("posrank") is not None]
        wb_overall[pos] = max(rs) if rs else None
        wb_posrank[pos] = max(prs) if prs else None
    return wb_overall, wb_posrank

def bench_tables(team_results, replacement_points: Dict[str,float], max_rank: int, beta: float = BENCH_OVAR_BETA):
    tables = {}
    totals = {}
    for team, res in team_results.items():
        rows = []
        for p in res["bench"]:
            pos = p["pos"]; r = p.get("rank")
            if pos not in SCORABLE_POS or r is None:
                continue
            pr = p.get("posrank")
            proj = p.get("proj")
            repl = replacement_points.get(pos, 0.0)
            if proj is None:
                proj = repl
            vor = max(0.0, proj - repl)
            ovar = max(0, max_rank - r + 1)
            rows.append({
                "name": p["name"], "pos": pos, "rank": r, "posrank": pr,
                "proj": float(proj) if proj is not None else None,
                "vor": round(vor, 2), "pVAR": round(vor, 2), "oVAR": ovar,
                "BenchScore": round(vor + beta*ovar, 2)
            })
        rows = sorted(rows, key=lambda x: x["BenchScore"], reverse=True)
        tables[team] = rows
        totals[team] = round(sum(x["BenchScore"] for x in rows), 2)
    return tables, totals

def _percentile_z_scores(values: Dict[str, float], clamp: float) -> Dict[str, float]:
    """Map values to z-like scores using percentile ranks and the Normal CDF."""
    if not values:
        return {}
    sorted_vals = sorted(values.values())
    n = len(sorted_vals)
    dist = stats.NormalDist()
    eps = max(1e-6, min(clamp, 0.49))

    scores = {}
    for team, val in values.items():
        left = bisect_left(sorted_vals, val)
        right = bisect_right(sorted_vals, val)
        percentile = (left + right) / (2 * n) if n > 0 else 0.5
        percentile = max(eps, min(1 - eps, percentile))
        scores[team] = dist.inv_cdf(percentile)
    return scores


def leaderboards(
    starters_totals: Dict[str, float],
    bench_totals: Dict[str, float],
    *,
    combined_starters_weight: float = COMBINED_STARTERS_WEIGHT,
    combined_bench_weight: float = COMBINED_BENCH_WEIGHT,
    bench_z_fallback_threshold: float = BENCH_Z_FALLBACK_THRESHOLD,
    bench_percentile_clamp: float = BENCH_PERCENTILE_CLAMP,
):
    starter_vals = list(starters_totals.values())
    bench_vals = list(bench_totals.values())
    mu_s = stats.mean(starter_vals)
    sd_s = stats.pstdev(starter_vals) or 1.0
    sd_b = stats.pstdev(bench_vals)
    mu_b = stats.mean(bench_vals)

    use_percentile = sd_b < bench_z_fallback_threshold
    if use_percentile:
        bench_z = _percentile_z_scores(bench_totals, bench_percentile_clamp)
    else:
        sd_b = sd_b or 1.0
        bench_z = {team: (bench_totals[team] - mu_b) / sd_b for team in bench_totals}

    bench_mean = stats.mean(bench_z.values())
    bench_z = {team: val - bench_mean for team, val in bench_z.items()}

    combined = {}
    for team in starters_totals:
        Zs = (starters_totals[team] - mu_s) / sd_s
        Zb = bench_z[team]
        combined[team] = round(combined_starters_weight * Zs + combined_bench_weight * Zb, 3)

    starters_board = sorted(starters_totals.items(), key=lambda x: x[1], reverse=True)
    bench_board = sorted(bench_totals.items(), key=lambda x: x[1], reverse=True)
    combined_board = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return starters_board, bench_board, combined_board, combined
