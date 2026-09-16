"""Response Priority Score.

    priority = w_d*damage + w_p*population + w_i*infrastructure + w_a*accessibility

Pure functions over counts - no imagery, no model, no I/O. All weights and thresholds come from
config/priority.yaml so they can be re-tuned live during the demo.

See PROJECT_PLAN.md section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import priority_config
from ..schemas import DamageClass, PriorityBreakdown

SEVERE_CLASSES = (DamageClass.DESTROYED, DamageClass.MAJOR)
DAMAGED_CLASSES = (DamageClass.DESTROYED, DamageClass.MAJOR, DamageClass.MINOR)


@dataclass
class WardInputs:
    """Everything the score needs about one ward."""

    ward_id: str
    ward_name: str
    damage_counts: dict[DamageClass, int]
    affected_population: int
    facility_types: list[str]
    road_disruption: float = 0.0       # 0 = clear, 1 = network cut
    distance_from_staging_km: float = 0.0


def raw_damage_score(counts: dict[DamageClass, int]) -> float:
    """Weighted building count. Un-normalized - normalization needs the whole AOI."""
    weights = priority_config()["damage_class_weights"]
    return sum(weights.get(cls.value, 0.0) * n for cls, n in counts.items())


def raw_infrastructure_score(facility_types: list[str]) -> float:
    """Summed facility importance. Un-normalized, like the damage term.

    Clipping this at 1.0 was wrong: a Kathmandu ward routinely has 150+ schools, clinics and
    bridges, so every ward saturated and the term stopped discriminating between them
    entirely. Normalizing against the AOI maximum keeps it informative.
    """
    weights = priority_config()["facility_weights"]
    return sum(weights.get(t, 0.3) for t in facility_types)


def accessibility_penalty(road_disruption: float, distance_km: float, max_distance_km: float) -> float:
    """0 = easy to reach, 1 = effectively cut off.

    Half road-network disruption, half distance from the staging area.
    """
    distance_term = distance_km / max_distance_km if max_distance_km > 0 else 0.0
    return _clip(0.5 * _clip(road_disruption) + 0.5 * _clip(distance_term))


def band_for(score: float) -> tuple[str, str]:
    """Map a score to its (name, color). Bands are evaluated top-down."""
    for band in priority_config()["bands"]:
        if score >= band["min"]:
            return band["name"], band["color"]
    return "Low", "#65a30d"


def score_wards(wards: list[WardInputs]) -> dict[str, tuple[float, PriorityBreakdown]]:
    """Score every ward in an AOI.

    Damage and population are normalized against the AOI maximum, so the score answers
    "which ward first?" rather than "how bad in absolute terms?" - which is the decision
    responders actually face.
    """
    if not wards:
        return {}

    weights = priority_config()["weights"]

    raw_damage = {w.ward_id: raw_damage_score(w.damage_counts) for w in wards}
    raw_infrastructure = {w.ward_id: raw_infrastructure_score(w.facility_types) for w in wards}
    max_damage = max(raw_damage.values()) or 1.0
    max_infrastructure = max(raw_infrastructure.values()) or 1.0
    max_population = max((w.affected_population for w in wards), default=0) or 1
    max_distance = max((w.distance_from_staging_km for w in wards), default=0.0)

    results: dict[str, tuple[float, PriorityBreakdown]] = {}
    for w in wards:
        breakdown = PriorityBreakdown(
            damage=_clip(raw_damage[w.ward_id] / max_damage),
            population=_clip(w.affected_population / max_population),
            infrastructure=_clip(raw_infrastructure[w.ward_id] / max_infrastructure),
            accessibility=accessibility_penalty(
                w.road_disruption, w.distance_from_staging_km, max_distance
            ),
        )
        score = _clip(
            weights["damage"] * breakdown.damage
            + weights["population"] * breakdown.population
            + weights["infrastructure"] * breakdown.infrastructure
            + weights["accessibility"] * breakdown.accessibility
        )
        results[w.ward_id] = (round(score, 4), breakdown)
    return results


def count_damaged(counts: dict[DamageClass, int]) -> int:
    return sum(counts.get(c, 0) for c in DAMAGED_CLASSES)


def count_severe(counts: dict[DamageClass, int]) -> int:
    return sum(counts.get(c, 0) for c in SEVERE_CLASSES)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
