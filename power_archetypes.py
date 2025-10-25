from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PowerSnapshotArchetype:
    key: str
    label: str
    sources: Tuple[str, ...]
    tags: Tuple[str, ...]
    description: str | None = None


POWER_SNAPSHOT_ARCHETYPES: Dict[str, PowerSnapshotArchetype] = {
    "balanced": PowerSnapshotArchetype(
        key="balanced",
        label="Balanced (default)",
        sources=("archetype.balanced", "archetype.balanced-current"),
        tags=("archetype:balanced",),
        description="Default blend of starter strength and bench depth.",
    ),
    "projection_sprint": PowerSnapshotArchetype(
        key="projection_sprint",
        label="Projection sprint",
        sources=("archetype.projection-sprint", "archetype.projection-sprint-current"),
        tags=("archetype:projection_sprint",),
        description="Projection-forward outlook prioritising short-term upside.",
    ),
    "depth_fortress": PowerSnapshotArchetype(
        key="depth_fortress",
        label="Depth fortress",
        sources=("archetype.depth-fortress", "archetype.depth-fortress-current"),
        tags=("archetype:depth_fortress",),
        description="Bench-heavy weighting aimed at playoff resilience.",
    ),
    "waiver_hawk": PowerSnapshotArchetype(
        key="waiver_hawk",
        label="Waiver hawk",
        sources=("archetype.waiver-hawk", "archetype.waiver-hawk-current"),
        tags=("archetype:waiver_hawk",),
        description="Aggressive waiver/streaming assumptions with shallow baselines.",
    ),
    "risk_shield": PowerSnapshotArchetype(
        key="risk_shield",
        label="Risk shield",
        sources=("archetype.risk-shield", "archetype.risk-shield-current"),
        tags=("archetype:risk_shield",),
        description="Conservative, floor-oriented settings with wider percentile clamps.",
    ),
}


def list_power_archetypes() -> List[PowerSnapshotArchetype]:
    return sorted(POWER_SNAPSHOT_ARCHETYPES.values(), key=lambda item: item.label.lower())


def resolve_archetype_filters(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    if not names:
        return [], []
    sources: List[str] = []
    tags: List[str] = []
    seen_sources = set()
    seen_tags = set()
    for name in names:
        archetype = POWER_SNAPSHOT_ARCHETYPES.get(name)
        if archetype is None:
            continue
        for src in archetype.sources:
            if src not in seen_sources:
                seen_sources.add(src)
                sources.append(src)
        for tag in archetype.tags:
            if tag not in seen_tags:
                seen_tags.add(tag)
                tags.append(tag)
    return sources, tags


def iterate_archetype_choices() -> Iterable[Tuple[str, PowerSnapshotArchetype]]:
    for key, value in POWER_SNAPSHOT_ARCHETYPES.items():
        yield key, value

