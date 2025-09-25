import math
from typing import Dict, List, Tuple
from config import SLOT_DEFS, SCORABLE_POS
from data import normalize_name
from alias import build_alias_map

def flatten_league_names(rosters: Dict) -> List[str]:
    acc = []
    for team, groups in rosters.items():
        for group, names in groups.items():
            for nm in names:
                acc.append(nm.replace(" (IR)", ""))
    return sorted(set(acc))

def build_team_players_pre(team_dict, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, csv_max_per_pos):
    """Pre-pass: give unknowns provisional finite ranks (CSV pos max +10) so DFS doesn't die."""
    players = []
    for group, names in team_dict.items():
        if group in ("K","DST"):
            continue
        for raw in names:
            base = raw.replace(" (IR)", "")
            csv_name = alias_map.get(base)
            if csv_name:
                pos = pos_by_name[csv_name]
                rnk = rank_by_name[csv_name]
                prank = posrank_by_name[csv_name]
                proj = proj_by_name.get(csv_name)
            else:
                pos = group if group in SCORABLE_POS else None
                if pos is not None:
                    mx = csv_max_per_pos.get(pos, max(rank_by_name.values()))
                    rnk = int(mx) + 10
                    prank = None  # will set later
                    proj = None
                else:
                    rnk = None; prank = None; proj = None
            players.append({"name": base, "csv": csv_name, "pos": pos, "rank": rnk, "posrank": prank, "proj": proj, "group": group})
    return players

def dfs_pick(players, slots):
    # Build eligibility
    eligible = []
    for _, elig in slots:
        cand = [i for i,p in enumerate(players) if (p["pos"] in elig) and (p["rank"] is not None)]
        eligible.append(cand)
    best = {"sum": math.inf, "assign": None}
    used = set(); assignment = [-1]*len(slots)
    order = sorted(range(len(slots)), key=lambda s: len(eligible[s]))
    def dfs(idx, curr_sum):
        nonlocal best
        if curr_sum >= best["sum"]: return
        if idx == len(order):
            best = {"sum": curr_sum, "assign": assignment.copy()}; return
        s = order[idx]
        if not eligible[s]: return
        cand_sorted = sorted(eligible[s], key=lambda i: players[i]["rank"])
        for i in cand_sorted:
            if i in used: continue
            used.add(i); assignment[s] = i
            dfs(idx+1, curr_sum + players[i]["rank"])
            assignment[s] = -1; used.remove(i)
    dfs(0,0)
    return best

def optimize_lineups_first_pass(rosters, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, csv_max_per_pos):
    results = {}
    for team, td in rosters.items():
        plist = build_team_players_pre(td, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, csv_max_per_pos)
        best = dfs_pick(plist, SLOT_DEFS)
        if best["assign"] is None:
            raise RuntimeError(f"Could not build a valid lineup for team {team}.")
        starters = [plist[i] for i in best["assign"]]
        bench = [p for j,p in enumerate(plist) if j not in set(best["assign"])]
        results[team] = {"starters": starters, "bench": bench}
    return results

def compute_worst_starter_bounds(first_pass):
    worst_overall = {}
    worst_posrank = {}
    for pos in ("QB","RB","WR","TE"):
        ranks = [p["rank"] for t in first_pass.values() for p in t["starters"] if p["pos"]==pos and p["rank"] is not None]
        pranks = [p.get("posrank") for t in first_pass.values() for p in t["starters"] if p["pos"]==pos and p.get("posrank") is not None]
        worst_overall[pos] = max(ranks) if ranks else None
        worst_posrank[pos] = max(pranks) if pranks else None
    return worst_overall, worst_posrank

def build_team_players_final(team_dict, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, worst_starter_overall, worst_starter_posrank):
    """Second pass: assign generous missing for potential starters using worst STARTER +10/+1."""
    players = []
    for group, names in team_dict.items():
        if group in ("K","DST"):
            continue
        for raw in names:
            base = raw.replace(" (IR)", "")
            csv_name = alias_map.get(base)
            if csv_name:
                pos = pos_by_name[csv_name]
                rnk = rank_by_name[csv_name]
                prank = posrank_by_name[csv_name]
                proj = proj_by_name.get(csv_name)
            else:
                pos = group if group in ("QB","RB","WR","TE") else None
                if pos is not None:
                    rnk = (worst_starter_overall[pos] or 0) + 10
                    prank = (worst_starter_posrank[pos] or 0) + 1
                    proj = None
                else:
                    rnk = None; prank = None; proj = None
            players.append({"name": base, "csv": csv_name, "pos": pos, "rank": rnk, "posrank": prank, "proj": proj, "group": group})
    return players

def optimize_lineups_second_pass(rosters, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, worst_starter_overall, worst_starter_posrank):
    results = {}
    for team, td in rosters.items():
        plist = build_team_players_final(td, alias_map, rank_by_name, pos_by_name, posrank_by_name, proj_by_name, worst_starter_overall, worst_starter_posrank)
        best = dfs_pick(plist, SLOT_DEFS)
        starters = [plist[i] for i in best["assign"]]
        bench = [p for j,p in enumerate(plist) if j not in set(best["assign"])]
        results[team] = {"starters": starters, "bench": bench}
    return results
