from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from main import DEFAULT_PROJECTIONS, DEFAULT_RANKINGS, evaluate_league


def _canonicalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw)
    if "combined_starters_weight" in cfg and "combined_bench_weight" not in cfg:
        starters_w = float(cfg["combined_starters_weight"])
        cfg["combined_bench_weight"] = max(0.0, min(1.0, 1.0 - starters_w))
    if "combined_bench_weight" in cfg and "combined_starters_weight" not in cfg:
        bench_w = float(cfg["combined_bench_weight"])
        cfg["combined_starters_weight"] = max(0.0, min(1.0, 1.0 - bench_w))
    return cfg


def generate_random_configs(
    *,
    num_samples: int,
    base_config: Optional[Dict[str, Any]] = None,
    ranges: Optional[Dict[str, tuple[float, float]]] = None,
    integer_params: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    base = dict(base_config or {})
    res: List[Dict[str, Any]] = []
    ranges = ranges or {}
    integer_params = set(integer_params or [])

    for _ in range(num_samples):
        cfg = dict(base)
        for key, (lo, hi) in ranges.items():
            if key in integer_params:
                value = int(round(rng.uniform(lo, hi)))
            else:
                value = float(rng.uniform(lo, hi))
            cfg[key] = value
        res.append(_canonicalize_config(cfg))
    return res


@dataclass
class SimulationRunRecord:
    run_id: str
    created_at: str
    rankings_path: str
    projections_path: str
    num_configs: int
    metadata: Dict[str, Any]


class SimulationStore:
    def __init__(self, root: str | Path = "simulations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"

    def list_runs(self) -> List[SimulationRunRecord]:
        if not self._index_path.exists():
            return []
        data = json.loads(self._index_path.read_text())
        return [SimulationRunRecord(**row) for row in data]

    def _write_index(self, runs: List[SimulationRunRecord]) -> None:
        payload = [row.__dict__ for row in runs]
        self._index_path.write_text(json.dumps(payload, indent=2))

    def run_batch(
        self,
        configs: List[Dict[str, Any]],
        *,
        rankings_path: Optional[str] = None,
        projections_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        precomputed: Optional[List[Dict[str, Any]]] = None,
    ) -> SimulationRunRecord:
        if not configs and not precomputed:
            raise ValueError("No configurations provided for simulation batch")

        rankings_path = rankings_path or str(DEFAULT_RANKINGS)
        projections_path = projections_path or str(DEFAULT_PROJECTIONS)

        if precomputed is not None:
            canonical_configs = [_canonicalize_config(rec["config"]) for rec in precomputed]
            leagues_precomputed = [rec["league"] for rec in precomputed]
        else:
            canonical_configs = [_canonicalize_config(cfg) for cfg in configs]
            leagues_precomputed = None

        if not canonical_configs:
            raise ValueError("Configuration list resolved to empty after canonicalization")

        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:6]}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        team_rows: List[Dict[str, Any]] = []
        config_rows: List[Dict[str, Any]] = []
        replacement_rows: List[Dict[str, Any]] = []
        scarcity_rows: List[Dict[str, Any]] = []

        total = len(canonical_configs)

        for cfg_idx, cfg in enumerate(canonical_configs):
            if leagues_precomputed is None:
                league = evaluate_league(
                    rankings_path,
                    projections_path=projections_path,
                    **cfg,
                )
            else:
                league = leagues_precomputed[cfg_idx]

            settings = league.get("settings", {})

            team_count = len(league["combined_scores"])
            leaderboards = league["leaderboards"]

            combined_rank_map = {team: i + 1 for i, (team, _) in enumerate(leaderboards["combined"])}
            starter_rank_map = {team: i + 1 for i, (team, _) in enumerate(leaderboards["starters"])}
            bench_rank_map = {team: i + 1 for i, (team, _) in enumerate(leaderboards["bench"])}

            combined_scores = league["combined_scores"]
            starters_totals = league["starters_totals"]
            bench_totals = league["bench_totals"]
            starter_projections = league["starter_projections"]

            for team, combined_value in combined_scores.items():
                team_rows.append(
                    {
                        "run_id": run_id,
                        "config_id": cfg_idx,
                        "team": team,
                        "combined_score": combined_value,
                        "combined_rank": combined_rank_map[team],
                        "starter_vor": starters_totals.get(team),
                        "starter_rank": starter_rank_map.get(team),
                        "bench_score": bench_totals.get(team),
                        "bench_rank": bench_rank_map.get(team),
                        "starter_projection": starter_projections.get(team),
                    }
                )

            config_record: Dict[str, Any] = {
                "run_id": run_id,
                "config_id": cfg_idx,
                "team_count": team_count,
                "settings_json": json.dumps(settings, default=str),
            }

            for key, value in settings.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    config_record[key] = value
                else:
                    config_record[key] = json.dumps(value, default=str)

            for key, value in cfg.items():
                column = f"input_{key}"
                config_record[column] = value

            combined_array = np.array(list(combined_scores.values()), dtype=float)
            config_record["combined_mean"] = float(combined_array.mean())
            config_record["combined_std"] = float(combined_array.std(ddof=0))

            config_rows.append(config_record)

            replacement_points = league.get("replacement_points", {})
            replacement_targets = league.get("replacement_targets", {})
            for pos, value in replacement_points.items():
                replacement_rows.append(
                    {
                        "run_id": run_id,
                        "config_id": cfg_idx,
                        "position": pos,
                        "replacement_points": value,
                        "replacement_slot": replacement_targets.get(pos),
                    }
                )

            scarcity_samples = league.get("scarcity_samples", {})
            for pos, samples in scarcity_samples.items():
                for sample in samples:
                    scarcity_rows.append(
                        {
                            "run_id": run_id,
                            "config_id": cfg_idx,
                            "position": pos,
                            "slot": sample.get("slot"),
                            "projection": sample.get("projection"),
                        }
                    )

            if progress_callback and leagues_precomputed is None:
                progress_callback(cfg_idx + 1, total)

        teams_df = pd.DataFrame(team_rows)
        configs_df = pd.DataFrame(config_rows)
        replacements_df = pd.DataFrame(replacement_rows)
        scarcity_df = pd.DataFrame(scarcity_rows)

        if not teams_df.empty:
            teams_df.to_parquet(run_dir / "teams.parquet", index=False)
        if not configs_df.empty:
            configs_df.to_parquet(run_dir / "configs.parquet", index=False)
        if not replacements_df.empty:
            replacements_df.to_parquet(run_dir / "replacement_levels.parquet", index=False)
        if not scarcity_df.empty:
            scarcity_df.to_parquet(run_dir / "scarcity_samples.parquet", index=False)

        run_metadata = {
            "run_id": run_id,
            "created_at": timestamp.isoformat(),
            "rankings_path": rankings_path,
            "projections_path": projections_path,
            "num_configs": len(canonical_configs),
            "tags": tags or [],
            "notes": notes,
            "extra_metadata": extra_metadata or {},
        }
        (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2))

        runs = self.list_runs()
        runs.append(
            SimulationRunRecord(
                run_id=run_id,
                created_at=run_metadata["created_at"],
                rankings_path=rankings_path,
                projections_path=projections_path,
                num_configs=len(canonical_configs),
                metadata={
                    "tags": tags or [],
                    "notes": notes,
                    "extra_metadata": extra_metadata or {},
                },
            )
        )
        self._write_index(runs)

        return runs[-1]

    def variance_refine(
        self,
        *,
        base_configs: List[Dict[str, Any]],
        ranges: Dict[str, tuple[float, float]],
        num_refinements: int,
        top_k_by_std: int = 3,
        integer_params: Optional[Iterable[str]] = None,
        rankings_path: Optional[str] = None,
        projections_path: Optional[str] = None,
        seed: Optional[int] = None,
        run_notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> SimulationRunRecord:
        if not base_configs:
            raise ValueError("variance_refine requires at least one base configuration")

        rankings_path = rankings_path or str(DEFAULT_RANKINGS)
        projections_path = projections_path or str(DEFAULT_PROJECTIONS)
        rng = np.random.default_rng(seed)
        integer_params = set(integer_params or [])

        results: List[Dict[str, Any]] = []

        total_expected = len(base_configs) + max(0, num_refinements)
        evaluations_done = 0

        def report_progress():
            nonlocal evaluations_done
            evaluations_done += 1
            if progress_callback:
                progress_callback(evaluations_done, total_expected)

        def evaluate_and_record(config: Dict[str, Any]) -> Dict[str, Any]:
            league = evaluate_league(
                rankings_path,
                projections_path=projections_path,
                **config,
            )
            combined_scores = league["combined_scores"]
            combined_array = np.array(list(combined_scores.values()), dtype=float)
            combined_std = float(combined_array.std(ddof=0))
            record = {
                "config": config,
                "combined_std": combined_std,
                "league": league,
            }
            results.append(record)
            report_progress()
            return record

        for cfg in base_configs:
            evaluate_and_record(cfg)

        for _ in range(num_refinements):
            results.sort(key=lambda r: r["combined_std"], reverse=True)
            candidates = results[:top_k_by_std]
            if not candidates:
                break
            parent = rng.choice(candidates)
            parent_cfg = parent["config"]

            child_cfg = dict(parent_cfg)
            for key, (lo, hi) in ranges.items():
                span = (hi - lo) * 0.2
                center = float(parent_cfg.get(key, (lo + hi) / 2))
                perturb = rng.normal(loc=center, scale=max(1e-6, span / 2))
                value = float(min(max(perturb, lo), hi))
                if key in integer_params:
                    value = int(round(value))
                child_cfg[key] = value

            evaluate_and_record(child_cfg)

        configs_to_run = [r["config"] for r in results]

        return self.run_batch(
            configs_to_run,
            rankings_path=rankings_path,
            projections_path=projections_path,
            tags=tags,
            notes=run_notes,
            extra_metadata={
                "strategy": "variance_refine",
                "base_configs": base_configs,
                "ranges": ranges,
                "num_refinements": num_refinements,
                "top_k_by_std": top_k_by_std,
            },
            progress_callback=None,
            precomputed=results,
        )

    def load_run(self, run_id: str) -> Dict[str, pd.DataFrame]:
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_id} not found")

        data: Dict[str, pd.DataFrame] = {}
        for name in [
            ("teams", "teams.parquet"),
            ("configs", "configs.parquet"),
            ("replacement_levels", "replacement_levels.parquet"),
            ("scarcity_samples", "scarcity_samples.parquet"),
        ]:
            key, filename = name
            path = run_dir / filename
            if path.exists():
                data[key] = pd.read_parquet(path)
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            data["metadata"] = pd.DataFrame([json.loads(meta_path.read_text())])
        return data
