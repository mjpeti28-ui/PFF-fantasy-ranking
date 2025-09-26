# Fantasy League Explorer

This repository evaluates custom fantasy football leagues on top of publicly available PFF data, blends projection-driven heuristics with value-over-replacement logic, and exposes the tooling through both a command-line report and an interactive Streamlit application. The code is configured for a redraft-style league, but almost every knob is exposed so you can retune the engine for new data sources or league rules.

## Ethos and Guiding Principles
- **Transparency first**: Every scoring weight, fallback, and tie-breaker lives in `config.py` so you can audit or override the math.
- **Reproducibility**: Input CSVs are snapshotted to `history/` whenever the Streamlit app runs, and simulation runs are versioned in `simulations/`.
- **Human-tuned heuristics**: Two-pass lineup construction, generous fallback ranks, and percentile-based bench z-scores prevent missing data from torpedoing teams.
- **Composable tooling**: Core primitives (`evaluate_league`, `TradeFinder`, `SimulationStore`) are pure-Python functions that can be scripted, tested, or embedded elsewhere.

## Quick Start
1. **Python environment**: Python 3.11+ recommended. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pandas numpy scipy openpyxl streamlit altair
   ```
2. **CLI report**: Generate the league digest with the bundled sample data.
   ```bash
   python main.py --projections projections.csv
   ```
   Common flags:
   - `--top-bench 15` to show more bench players per team.
   - `--all-teams` to print every roster breakdown.
3. **Streamlit UI**: Launch the interactive dashboard with trade finder, power rankings, simulations, and add/drop tools.
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Excel output (optional)**: Use `reports.py` to export the tables produced by `evaluate_league` into an Excel workbook with methodology notes.

## Data Inputs and Saved Artifacts
- `ROS_week_2_PFF_rankings.csv`, `PFF_rankings.csv`: Baseline and supplemental PFF rankings (combined during ingest).
- `projections.csv`: Player projection feed (`playerName`, `fantasyPoints`) used to scale raw ranks.
- `Stats/`: Season-to-date stat aggregates consumed by the EPW (expected points per week) engine.
- `StrengthOfSchedule/`: Weekly SOS z-scores by position, folded into EPW adjustments.
- `rosters.py`: Canonical league rosters (NFL and college players) split by positional groups.
- `history/`: Auto-versioned copies of the latest rankings, projections, stats, and SOS files (timestamped by `history_tracker.py`).
- `simulations/`: Persistent simulation runs with parameter sweeps, team summaries, and replacement levels.

## Core Evaluation Flow
1. **Rankings ingest (`data.py`)**: Load primary and supplemental CSVs, normalize column names, compute position ranks, and merge projections (with configurable scaling via `PROJECTION_SCALE_BETA`).
2. **Roster aliasing (`alias.py`)**: Fuzzy-match fantasy roster names to canonical CSV names (`FUZZY_CUTOFF`) while stripping IR markers and suffixes.
3. **Two-pass lineup optimization (`optimizer.py`)**:
   - Pass 1 assigns optimistic ranks so DFS can fill every fantasy slot (`SLOT_DEFS`).
   - Pass 2 back-fills missing starters with "worst starter +10 overall/+1 positional" heuristics.
4. **Replacement baselines (`scoring.py`)**: Build monotone cubic splines over projection data by position, skip the elite top `replacement_skip_pct`, and average across a `replacement_window` to derive replacement-level points.
5. **Starter and bench scoring (`scoring.py`)**:
   - Starter VOR totals weigh projection surplus against replacement.
   - Bench scoring combines projection VOR and overall rank advantage (`BenchScore = VOR + beta * oVAR`).
   - Bench variance converts to z-scores or percentile z-scores once variance drops below `BENCH_Z_FALLBACK_THRESHOLD`.
6. **Leaderboards and reporting (`main.py`, `reports.py`)**: Produce starter/bench/combined leaderboards, positional summaries, and optional Excel exports.
7. **Expected points engine (`epw.py`)**: Distribute projection points across remaining weeks, blend with per-game stats, and adjust for strength of schedule to evaluate trades or EPW leaderboards.

## Module Guide and Notable Heuristics
- `config.py`: Central switchboard for fuzzy matching cutoffs, projection scaling, combined score weights, bench heuristics, and slot definitions. Adjust here before touching code.
- `alias.py`: Uses `difflib.SequenceMatcher` to resolve roster names against ranking data, prioritizing exact matches before fuzzy matches.
- `data.py`: Normalizes name strings, merges projections, rescales ranks using a projection z-score multiplier, and exposes convenience lookups for ranks/positions.
- `rosters.py`: Frozen league rosters partitioned by positional group; IR tags are preserved for UI context but stripped before scoring.
- `optimizer.py`: Depth-first search lineup allocator with a two-pass safety net so missing ranks or projections never crash optimization.
- `scoring.py`: Implements scarcity curves (monotone cubic splines), replacement-level search, bench generosity pass (adds +10 overall/+1 positional rank for missing bench data), percentile-based z-score fallback, and combined score weighting.
- `reports.py`: Flattens results into `pandas` DataFrames, builds leaderboards, and writes Excel reports including methodology and replacement-level tabs.
- `main.py`: Command-line entry point. Orchestrates the full evaluation pipeline, prints human-readable summaries, and exposes `evaluate_league` for downstream consumers.
- `epw.py`: Expected-points simulator blending projections, historical stats, and SOS adjustments; supports full-league EPW summaries and trade deltas.
- `trading.py`: Home of `TradeFinder`, the exhaustive trade search engine. Key heuristics:
  - Three fairness modes (`sum`, `weighted`, `nash`) determine the optimization objective.
  - Drop-tax penalties charge teams for cutting positive-VOR bench players.
  - Acceptance probability blends fairness, combined-score need, star-power inflow, and drop-tax friction.
  - Optional natural-language narratives summarize each side's gains.
- `streamlit_app.py`: Multi-tool UI wrapping trade finder, power rankings, simulation playground, add/drop impact, and manual trade analyzer. Archives inputs via `history_tracker` and caches simulation results with `SimulationStore`.
- `simulation.py`: Batch runner for sweeping configuration spaces. Stores run metadata, raw team results, parameter grids, replacement levels, and scarcity samples with unique run IDs.
- `history_tracker.py`: Lightweight content-addressable archive that snapshots input files when their hash changes.
- `simulations/` & `history/`: File-system persistence layers for auditability; never edit manually.

## Interactive Toolkit Highlights
- **Trade Finder** (`streamlit_app.py` → `trading.py`): Explore win-win packages with knobs for fairness mode, acceptance thresholds, drop-tax scaling, and narrative output.
- **Power Rankings**: Replay the core model with different projection scaling, bench weights, scarcity sampling density, or EPW overlays.
- **Simulation Playground** (`SimulationStore`): Sample hundreds of configurations, refine around high-variance settings, and visualize combined score distributions.
- **Add/Drop Impact**: Mutate a single team's roster and instantly recompute league standings and bench depth.
- **Trade Analyzer**: Manually select players from two teams and simulate post-trade combined scores and EPW trajectories.

## Key Heuristic Reference
| Lever | Location | Effect |
| ----- | -------- | ------ |
| `FUZZY_CUTOFF` | `config.py` | Minimum similarity for aliasing roster names to ranking entries. |
| `PROJECTION_SCALE_BETA` | `config.py` & `data.py` | Scales raw ranks using projection z-scores to spread tiers. |
| `SLOT_DEFS` | `config.py` | Fantasy lineup template used by both optimizer and EPW simulations. |
| `BENCH_OVAR_BETA` | `config.py` & `scoring.py` | Weight assigned to overall rank advantage on the bench. |
| `BENCH_Z_FALLBACK_THRESHOLD`/`BENCH_PERCENTILE_CLAMP` | `config.py` & `scoring.py` | Switch to percentile z-scores when bench variance collapses. |
| `replacement_skip_pct` & `replacement_window` | `evaluate_league` kwargs | Control how replacement-level baselines ignore elites and smooth projections. |
| Trade `fairness_mode`, `drop_tax_factor`, `acceptance_*` | `trading.py` | Shape the trade search objective and acceptance probability. |
| EPW `alpha` | `epw.py`, UI sliders | Determines how strongly schedule strength adjusts weekly expectations. |

## Extending or Customizing
- **Swap in new data**: Replace the CSVs in the project root (and optionally add to `history/` for provenance). The next Streamlit session will archive them automatically.
- **Change league structure**: Edit `rosters.py` (and possibly `SLOT_DEFS`) to reflect new teams, dynasty rules, or positional slots.
- **Tune scoring**: Modify the constants in `config.py` or pass overrides into `evaluate_league`/`evaluate_league_safe` (e.g., different `combined_starters_weight`).
- **Add new reports**: Use `evaluate_league` outputs (starter totals, bench tables, scarcity curves) as building blocks for custom analytics or exports.
- **Integrate with other systems**: `evaluate_league`, `TradeFinder`, `SimulationStore`, and `compute_league_epw` are all self-contained and usable from notebooks or alternative UIs.

## Repository Layout
```
.
├── alias.py                # Roster ↔ ranking name reconciliation
├── config.py               # Tunable knobs for scoring and heuristics
├── data.py                 # CSV ingest, normalization, projection scaling
├── epw.py                  # Expected-points engine for league/trade analysis
├── history/                # Archived copies of upstream data (auto-managed)
├── history_tracker.py      # Hash-based snapshot mechanism
├── main.py                 # CLI entry point & evaluate_league orchestrator
├── optimizer.py            # Two-pass lineup optimizer using DFS and heuristics
├── projections.csv         # Sample projection feed (fantasyPoints)
├── reports.py              # DataFrame builders + Excel writer helpers
├── rosters.py              # Canonical fantasy rosters grouped by slot
├── scoring.py              # Replacement levels, bench scoring, leaderboards
├── simulation.py           # Configuration sweeps and persistence helpers
├── simulations/            # Stored simulation runs
├── streamlit_app.py        # Streamlit front-end wiring up all tooling
├── StrengthOfSchedule/     # Position-specific SOS data ingested by EPW
├── Stats/                  # PFF fantasy stats ingested by EPW
├── trading.py              # TradeFinder search engine and heuristics
├── ROS_week_2_PFF_rankings.csv / PFF_rankings.csv # Sample ranking feeds
└── README.md               # You are here
```

## Testing and Validation Tips
- Run `python main.py` after any scoring tweak to make sure starter/bench leaderboards remain sensible.
- When altering projections or rankings, confirm `history/manifest.json` updates (indicating a new snapshot was archived).
- Use the Streamlit "Simulation Playground" to sanity-check the sensitivity of combined scores to new heuristics before shipping them.
- For trade logic changes, re-run the Streamlit Trade Finder against known scenarios and verify acceptance scores and narratives are aligned with expectations.
- Run `pytest` (or `python -m pytest`) to exercise the FastAPI endpoints via the in-process tests in `tests/`. These validate `/players`, `/rankings`, `/evaluate`, and `/trade/evaluate` without requiring a live Uvicorn server.

Happy roster tinkering!
