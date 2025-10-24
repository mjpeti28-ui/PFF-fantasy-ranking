from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from context import ContextManager
from espn_client import convert_league_to_rosters, fetch_league_state_for_week
from main import evaluate_league
from power_snapshots import POWER_HISTORY_DIR, build_snapshot_metadata, iter_snapshot_metadata

LOG = logging.getLogger("backfill")


def _parse_weeks(raw: str | None, range_raw: str | None) -> List[int]:
    weeks: set[int] = set()
    if raw:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                weeks.add(int(chunk))
            except ValueError as exc:
                raise ValueError(f"Invalid week '{chunk}'") from exc
    if range_raw:
        parts = [part.strip() for part in range_raw.split("-", 1)]
        if len(parts) != 2:
            raise ValueError("Week range must use the form start-end (e.g. 1-8)")
        try:
            start, end = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Invalid week range '{range_raw}'") from exc
        if start > end:
            start, end = end, start
        for week in range(start, end + 1):
            weeks.add(week)
    ordered = sorted(week for week in weeks if week > 0)
    if not ordered:
        raise ValueError("At least one positive week must be provided.")
    return ordered


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_existing_keys() -> set[Tuple[Any, ...]]:
    existing: set[Tuple[Any, ...]] = set()
    metadata = iter_snapshot_metadata()
    for entry in metadata:
        tag_tuple = tuple(entry.get("tags") or [])
        key = (entry.get("source"), entry.get("week"), tag_tuple)
        existing.add(key)
    return existing


@dataclass
class InputResolver:
    rankings_path: Optional[str] = None
    projections_path: Optional[str] = None
    supplemental_path: Optional[str] = None
    rankings_template: Optional[str] = None
    projections_template: Optional[str] = None

    def resolve_rankings(self, week: int, default_path: str) -> str:
        if self.rankings_template:
            candidate = self.rankings_template.format(week=week)
            path = Path(candidate)
            if not path.exists():
                raise FileNotFoundError(f"Rankings template resolved to missing file: {candidate}")
            return candidate
        if self.rankings_path:
            path = Path(self.rankings_path)
            if not path.exists():
                raise FileNotFoundError(f"Rankings file not found: {self.rankings_path}")
            return self.rankings_path
        return default_path

    def resolve_projections(self, week: int, default_path: Optional[str]) -> Optional[str]:
        if self.projections_template:
            candidate = self.projections_template.format(week=week)
            path = Path(candidate)
            if not path.exists():
                raise FileNotFoundError(f"Projections template resolved to missing file: {candidate}")
            return candidate
        if self.projections_path:
            path = Path(self.projections_path)
            if not path.exists():
                raise FileNotFoundError(f"Projections file not found: {self.projections_path}")
            return self.projections_path
        return default_path


@dataclass
class BackfillOptions:
    weeks: Sequence[int]
    mode: str
    source: str
    base_tags: Tuple[str, ...] = field(default_factory=tuple)
    force: bool = False
    dry_run: bool = False
    resolver: InputResolver = field(default_factory=InputResolver)
    evaluate_overrides: Dict[str, Any] = field(default_factory=dict)


def _historical_rosters(week: int) -> Dict[str, Dict[str, List[str]]]:
    LOG.debug("Fetching ESPN data for week %s", week)
    league_state = fetch_league_state_for_week(week)
    return convert_league_to_rosters(league_state)


def _build_tags(base_tags: Iterable[str], week: int, mode: str) -> List[str]:
    tags = [tag for tag in base_tags if tag]
    tags.append(f"week:{week}")
    tags.append(f"mode:{mode}")
    return sorted(set(tags))


def _should_skip(existing: set[Tuple[Any, ...]], key: Tuple[Any, ...], force: bool) -> bool:
    if force:
        return False
    return key in existing


def run_backfill(options: BackfillOptions) -> None:
    manager = ContextManager()
    ctx = manager.get()
    existing_keys = _load_existing_keys()
    created = 0
    skipped = 0

    default_rankings = str(ctx.rankings_path)
    default_projections = str(ctx.projections_path) if ctx.projections_path else None
    supplemental_path = options.resolver.supplemental_path or (
        str(ctx.supplemental_path) if ctx.supplemental_path else None
    )

    for week in options.weeks:
        try:
            rankings_path = options.resolver.resolve_rankings(week, default_rankings)
            projections_path = options.resolver.resolve_projections(week, default_projections)
        except FileNotFoundError as exc:
            LOG.error("Week %s: %s", week, exc)
            continue

        snapshot_tags = _build_tags(options.base_tags, week, options.mode)
        key = (options.source, week, tuple(snapshot_tags))
        if _should_skip(existing_keys, key, options.force):
            LOG.info("Week %s: snapshot already exists for source=%s tags=%s (skipping)", week, options.source, snapshot_tags)
            skipped += 1
            continue

        if options.mode == "historical":
            try:
                rosters = _historical_rosters(week)
            except Exception as exc:
                LOG.error("Week %s: failed to fetch historical rosters (%s)", week, exc)
                continue
        elif options.mode == "current":
            rosters = None
        else:
            LOG.error("Week %s: unsupported mode '%s'", week, options.mode)
            continue

        snapshot_metadata = build_snapshot_metadata(
            options.source,
            ctx if options.mode == "historical" else None,
            week=week,
            tags=snapshot_tags,
            mode=options.mode,
        )

        if options.dry_run:
            LOG.info(
                "Week %s: would evaluate rankings=%s projections=%s mode=%s tags=%s",
                week,
                rankings_path,
                projections_path,
                options.mode,
                snapshot_tags,
            )
            created += 1
            continue

        eval_kwargs = dict(options.evaluate_overrides)
        if rosters is not None:
            eval_kwargs["custom_rosters"] = rosters

        evaluate_league(
            rankings_path,
            projections_path=projections_path,
            supplemental_path=supplemental_path,
            snapshot_metadata=snapshot_metadata,
            save_snapshot=True,
            **eval_kwargs,
        )
        LOG.info(
            "Week %s: snapshot saved (source=%s tags=%s rankings=%s projections=%s)",
            week,
            options.source,
            snapshot_tags,
            rankings_path,
            projections_path,
        )
        created += 1

    LOG.info("Backfill complete: %s processed, %s created, %s skipped", len(options.weeks), created, skipped)


def _parse_overrides(raw: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: '{item}'")
        overrides[key] = _coerce_value(value)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill power ranking snapshots for historical analysis.")
    parser.add_argument("--weeks", help="Comma-separated list of weeks to backfill (e.g. 1,2,3).")
    parser.add_argument("--week-range", help="Range of weeks (e.g. 1-8).")
    parser.add_argument("--mode", choices=["historical", "current"], default="historical", help="Roster mode to backfill.")
    parser.add_argument("--source", default="backfill", help="Source label stored in the snapshot metadata.")
    parser.add_argument("--tag", action="append", dest="tags", default=[], help="Additional tag to attach to each snapshot.")
    parser.add_argument("--force", action="store_true", help="Overwrite/duplicate snapshots even if one already exists for the week/source/tags combination.")
    parser.add_argument("--dry-run", action="store_true", help="Do not persist snapshots; only log intended actions.")
    parser.add_argument("--rankings-path", help="Explicit rankings file to use (overrides context defaults).")
    parser.add_argument("--projections-path", help="Explicit projections file to use (overrides context defaults).")
    parser.add_argument("--supplemental-path", help="Explicit supplemental rankings file.")
    parser.add_argument("--rankings-template", help="Template for per-week rankings files (e.g. data/rankings/week_{week}.csv).")
    parser.add_argument("--projections-template", help="Template for per-week projections files.")
    parser.add_argument("--eval-override", action="append", default=[], help="Override evaluator kwargs (key=value).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    weeks = _parse_weeks(args.weeks, args.week_range)
    overrides = _parse_overrides(args.eval_override)
    resolver = InputResolver(
        rankings_path=args.rankings_path,
        projections_path=args.projections_path,
        supplemental_path=args.supplemental_path,
        rankings_template=args.rankings_template,
        projections_template=args.projections_template,
    )
    options = BackfillOptions(
        weeks=weeks,
        mode=args.mode,
        source=args.source,
        base_tags=tuple(tag for tag in args.tags if tag),
        force=args.force,
        dry_run=args.dry_run,
        resolver=resolver,
        evaluate_overrides=overrides,
    )

    POWER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    run_backfill(options)


if __name__ == "__main__":
    main()
