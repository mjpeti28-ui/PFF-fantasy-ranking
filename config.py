# Global knobs to tweak behavior

# Fuzzy match cutoff for name aliasing (0..1); higher = stricter
FUZZY_CUTOFF = 0.84

# Bench scoring weight for overall VAR tie-break
BENCH_OVAR_BETA = 0.25

# Combined score weights
COMBINED_STARTERS_WEIGHT = 0.80
COMBINED_BENCH_WEIGHT = 0.20

# Projection scaling
PROJECTIONS_CSV = "projections.csv"
PROJECTION_SCALE_BETA = 0.50  # higher values amplify projection-driven separation

# Slots used in optimization (K/DST omitted from scoring)
SLOT_DEFS = [
    ("QB", {"QB"}),
    ("RB", {"RB"}),
    ("RB/WR", {"RB", "WR"}),
    ("WR", {"WR"}),
    ("WR/TE", {"WR", "TE"}),
    ("TE", {"TE"}),
    ("WR/RB/TE", {"WR", "RB", "TE"}),
]

# Positions we actually score
SCORABLE_POS = {"QB", "RB", "WR", "TE"}

# Bench Z-score stabilization
BENCH_Z_FALLBACK_THRESHOLD = 3.0  # minimum bench stddev before falling back
BENCH_PERCENTILE_CLAMP = 0.05      # percentile clamp when converting to z

# Columns expected in rankings CSV (rename map)
CSV_RENAME = {"RankRk.": "Rank", "PositionPos.": "Position"}
