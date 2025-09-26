from difflib import SequenceMatcher
from typing import Dict, Optional, Iterable
from data import normalize_name
from config import settings

def find_best(name: str, df) -> Optional[str]:
    """Return canonical CSV Name or None."""
    n = normalize_name(name)
    exact = df[df["name_norm"] == n]
    if len(exact) == 1:
        return exact.iloc[0]["Name"]
    elif len(exact) > 1:
        # if pathological, choose best overall (lowest Rank)
        return exact.sort_values("Rank").iloc[0]["Name"]
    # fuzzy
    best = None
    best_score = 0.0
    cutoff = settings.get("fuzzy_cutoff")
    for _, row in df.iterrows():
        score = SequenceMatcher(None, n, row["name_norm"]).ratio()
        if score > best_score:
            best, best_score = row, score
    if best is not None and best_score >= cutoff:
        return best["Name"]
    return None

def build_alias_map(league_names: Iterable[str], df) -> Dict[str, Optional[str]]:
    m = {}
    for nm in league_names:
        nm = nm.replace(" (IR)", "")
        m[nm] = find_best(nm, df)
    return m
