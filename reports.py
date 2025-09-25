import pandas as pd
from typing import Dict, List

def build_reports(team_results, starters_totals, bench_tables, bench_totals, combined_scores):
    # Starters detail
    starter_rows = []
    for team, res in team_results.items():
        for p in res["starters"]:
            starter_rows.append({
                "Team": team,
                "Slot/Nativity": p["pos"],
                "Name": p["name"],
                "CSV_Name": p.get("csv"),
                "OverallRank": p.get("rank"),
                "PosRank": p.get("posrank"),
            })
    df_starters = pd.DataFrame(starter_rows).sort_values(["Team","OverallRank"])

    # Bench detail (flatten)
    bench_rows = []
    for team, rows in bench_tables.items():
        for p in rows:
            bench_rows.append({
                "Team": team,
                "Name": p["name"],
                "Pos": p["pos"],
                "OverallRank": p["rank"],
                "PosRank": p["posrank"],
                "pVAR": p["pVAR"],
                "oVAR": p["oVAR"],
                "BenchScore": p["BenchScore"]
            })
    df_bench = pd.DataFrame(bench_rows).sort_values(["Team","BenchScore"], ascending=[True,False])

    # Team summary
    df_summary = pd.DataFrame([
        {"Team": t, "StarterSum": starters_totals[t],
         "AvgStarterRank": round(starters_totals[t]/7,2),
         "BenchScoreTotal": bench_totals[t],
         "CombinedScore": combined_scores[t]} for t in starters_totals
    ]).sort_values("CombinedScore", ascending=False)

    # Leaderboards
    lb_starters = df_summary[["Team","StarterSum"]].sort_values("StarterSum")
    lb_bench = df_summary[["Team","BenchScoreTotal"]].sort_values("BenchScoreTotal", ascending=False)
    lb_combined = df_summary[["Team","CombinedScore"]].sort_values("CombinedScore", ascending=False)

    # Positional group sums over starters (native)
    pos_groups = []
    for team, res in team_results.items():
        row = {"Team": team, "QB_Sum": 0, "RB_Sum": 0, "WR_Sum": 0, "TE_Sum": 0,
               "QB_Cnt":0, "RB_Cnt":0, "WR_Cnt":0, "TE_Cnt":0}
        for p in res["starters"]:
            if p["pos"] in ("QB","RB","WR","TE"):
                row[f"{p['pos']}_Sum"] += p["rank"]
                row[f"{p['pos']}_Cnt"] += 1
        pos_groups.append(row)
    df_pos_groups = pd.DataFrame(pos_groups)

    lb_qb = df_pos_groups[["Team","QB_Sum","QB_Cnt"]].sort_values(["QB_Sum","Team"])
    lb_rb = df_pos_groups[["Team","RB_Sum","RB_Cnt"]].sort_values(["RB_Sum","Team"])
    lb_wr = df_pos_groups[["Team","WR_Sum","WR_Cnt"]].sort_values(["WR_Sum","Team"])
    lb_te = df_pos_groups[["Team","TE_Sum","TE_Cnt"]].sort_values(["TE_Sum","Team"])

    # Starter matrix
    df_starters_matrix = df_starters.pivot_table(index="Team", columns="Slot/Nativity", values="OverallRank", aggfunc=lambda x: sorted(x)[0]).reset_index()

    return {
        "Starters_Detail": df_starters,
        "Bench_Detail": df_bench,
        "Team_Summary": df_summary,
        "Leaderboard_Starters": lb_starters,
        "Leaderboard_Bench": lb_bench,
        "Leaderboard_Combined": lb_combined,
        "Positional_Groups_Starters": df_pos_groups,
        "LB_QB": lb_qb,
        "LB_RB": lb_rb,
        "LB_WR": lb_wr,
        "LB_TE": lb_te,
        "Starters_Matrix": df_starters_matrix,
    }

def write_excel(tables: Dict[str, pd.DataFrame], out_path: str, methodology_rows: List[List[str]], replacement_counts: Dict[str,int]):
    df_assumptions = pd.DataFrame(methodology_rows, columns=["Item","Details"])
    df_repl = pd.DataFrame(list(replacement_counts.items()), columns=["Position","ReplacementCount"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name)
        df_repl.to_excel(writer, index=False, sheet_name="Replacement_Levels")
        df_assumptions.to_excel(writer, index=False, sheet_name="Methodology")
