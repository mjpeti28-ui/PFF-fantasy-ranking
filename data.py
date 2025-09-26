import copy
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Optional
from config import CSV_RENAME, PROJECTIONS_CSV, settings
from rosters import Fantasy_Rosters

SUFFIXES = [' jr', ' sr', ' iii', ' ii', ' iv', ' v']
DEFAULT_PROJECTIONS_PATH = Path(__file__).with_name(PROJECTIONS_CSV)

def normalize_name(s: str) -> str:
    s = s.lower()
    s = s.replace('.', '').replace("’", "'").replace("'", '')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    for suf in SUFFIXES:
        s = re.sub(rf'{suf}$', '', s.strip())
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_projections(path: Optional[str] = None) -> pd.DataFrame:
    target = Path(path) if path is not None else DEFAULT_PROJECTIONS_PATH
    if not target.exists():
        return pd.DataFrame(columns=["name_norm", "ProjPoints"])
    proj_df = pd.read_csv(target)
    if "playerName" not in proj_df.columns or "fantasyPoints" not in proj_df.columns:
        raise ValueError("Projections CSV must contain 'playerName' and 'fantasyPoints' columns.")
    proj_df["name_norm"] = proj_df["playerName"].apply(normalize_name)
    proj_df["ProjPoints"] = proj_df["fantasyPoints"].astype(float)
    proj_df = proj_df[["name_norm", "ProjPoints"]]
    proj_df = proj_df.drop_duplicates("name_norm", keep="first")
    return proj_df


def _read_rankings_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, skiprows=1)
    # If the file still comes back as a single generic column, try skipping the first row
    if len(df.columns) == 1:
        try:
            df_alt = pd.read_csv(path, skiprows=1)
            if len(df_alt.columns) > 1:
                df = df_alt
        except pd.errors.ParserError:
            pass
    return df


def _prepare_rankings_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

    rename_map = dict(CSV_RENAME)
    rename_map.update({
        "Overall Rank": "Rank",
        "Full Name": "Name",
        "Team Abbreviation": "Team",
        "Team": "Team",
        "Name": "Name",
        "Position Rank": "PosRank",
    })
    df = df.rename(columns=rename_map)
    if "Rank" not in df.columns or "Position" not in df.columns:
        raise ValueError("Rankings CSV must have columns that map to ['Rank','Position','Name','Team']")

    extra_cols = {
        "Bye Week": "ByeWeek",
        "ADP": "ADP",
        "Projected Points": "ProjPointsCsv",
        "Auction Value": "AuctionValue",
    }
    for src, dst in extra_cols.items():
        if src in df.columns:
            df[dst] = df[src]
    df["Rank"] = df["Rank"].astype(int)
    df["name_norm"] = df["Name"].apply(normalize_name)
    df["PosRank"] = df.groupby("Position")["Rank"].rank(method="min").astype(int)
    df["RankOriginal"] = df["Rank"]
    return df


def load_rankings(
    csv_path: str,
    projections_path: Optional[str] = None,
    *,
    projection_scale_beta: Optional[float] = None,
    supplemental_path: Optional[str] = None,
) -> pd.DataFrame:
    df_primary = _prepare_rankings_dataframe(_read_rankings_csv(csv_path))

    if supplemental_path:
        df_supp = _prepare_rankings_dataframe(_read_rankings_csv(supplemental_path))
        df_supp = df_supp[~df_supp["Position"].isin({"K", "DST"})]
        existing = set(df_primary["name_norm"])
        df_supp = df_supp[~df_supp["name_norm"].isin(existing)]
        df = pd.concat([df_primary, df_supp], ignore_index=True)
        df = df.sort_values(["Position", "RankOriginal", "Rank"]).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["PosRank"] = df.groupby("Position")["Rank"].rank(method="min").astype(int)
    else:
        df = df_primary

    proj_df = load_projections(projections_path)
    beta_default = settings.get("projection_scale_beta")
    beta = beta_default if projection_scale_beta is None else projection_scale_beta
    if not proj_df.empty:
        df = df.merge(proj_df, on="name_norm", how="left", suffixes=('', '_proj'))
        if df["ProjPoints"].notna().any():
            overall_median = df["ProjPoints"].median()
            if pd.isna(overall_median):
                overall_median = 0.0
            df["ProjPoints"] = df.groupby("Position")["ProjPoints"].transform(
                lambda s: s.fillna(s.median() if s.notna().any() else overall_median)
            )
            df["ProjPoints"] = df["ProjPoints"].fillna(overall_median)
            pos_means = df.groupby("Position")["ProjPoints"].transform("mean")
            pos_stds = df.groupby("Position")["ProjPoints"].transform("std").fillna(0.0)
            pos_stds = pos_stds.replace(0, 1.0)
            df["ProjZ"] = (df["ProjPoints"] - pos_means) / pos_stds
            denom = (1 + beta * df["ProjZ"]).clip(lower=0.1)
            scaled = df["RankOriginal"] / denom
            df["RankScaled"] = scaled
            df["Rank"] = (
                df["RankScaled"].rank(method="dense", ascending=True)
                .astype(int)
                .clip(lower=1)
            )
        else:
            df["ProjPoints"] = None
            df["ProjZ"] = 0.0
            df["RankScaled"] = df["Rank"].astype(float)
    else:
        df["ProjPoints"] = None
        df["ProjZ"] = 0.0
        df["RankScaled"] = df["Rank"].astype(float)
    return df

def build_lookups(df: pd.DataFrame):
    rank_by_name = {row["Name"]: int(row["Rank"]) for _, row in df.iterrows()}
    pos_by_name = {row["Name"]: row["Position"] for _, row in df.iterrows()}
    posrank_by_name = df.set_index("Name")["PosRank"].to_dict()
    proj_by_name = df.set_index("Name")["ProjPoints"].to_dict()
    return rank_by_name, pos_by_name, posrank_by_name, proj_by_name

def load_rosters() -> Dict:
    """Return a deep copy of the canonical roster data."""
    return copy.deepcopy(Fantasy_Rosters)
