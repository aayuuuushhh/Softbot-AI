"""In-memory assessment store.

SQLite/SQLAlchemy lands in H24-H40 (PROJECT_PLAN.md section 6); the interface here is
deliberately narrow so swapping the backing store touches only this file.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import priority_config
from .schemas import (
    AssessmentStatus,
    AssessmentSummary,
    BuildingProperties,
    DamageClass,
    GeometrySource,
    Feature,
    FeatureCollection,
    InferenceMode,
    PriorityBreakdown,
    WardCard,
)
from .services.score import band_for, count_damaged, count_severe, score_wards, WardInputs


@dataclass
class Building:
    id: str
    ward_id: str
    geometry: dict
    damage_class: DamageClass
    confidence: float
    reviewed_by_human: bool = False
    review_note: str | None = None


@dataclass
class Ward:
    ward_id: str
    ward_name: str
    geometry: dict | None = None
    facility_types: list[str] = field(default_factory=list)
    # People per damaged building in this ward. A proxy until the WorldPop zonal sum
    # lands - see PROJECT_PLAN.md section 4.
    people_per_building: float = 4.5
    road_disruption: float = 0.0
    distance_from_staging_km: float = 0.0


@dataclass
class Assessment:
    id: str
    status: AssessmentStatus
    mode: InferenceMode
    aoi_name: str | None
    created_at: str
    geometry_source: GeometrySource = GeometrySource.SYNTHETIC
    buildings: list[Building] = field(default_factory=list)
    wards: list[Ward] = field(default_factory=list)
    error: str | None = None

    # --- derived -----------------------------------------------------------

    def buildings_by_ward(self) -> dict[str, list[Building]]:
        grouped: dict[str, list[Building]] = {w.ward_id: [] for w in self.wards}
        for b in self.buildings:
            grouped.setdefault(b.ward_id, []).append(b)
        return grouped

    def ward_cards(self) -> list[WardCard]:
        """Aggregate buildings into the brief's ward cards, then rank by priority.

        Recomputed on every read, so a human override immediately moves the ward.
        """
        grouped = self.buildings_by_ward()

        inputs: list[WardInputs] = []
        counts_by_ward: dict[str, dict[DamageClass, int]] = {}
        population_by_ward: dict[str, int] = {}

        for ward in self.wards:
            members = grouped.get(ward.ward_id, [])
            counts = {cls: 0 for cls in DamageClass}
            counts.update(Counter(b.damage_class for b in members))
            counts_by_ward[ward.ward_id] = counts

            affected = int(round(count_damaged(counts) * ward.people_per_building))
            population_by_ward[ward.ward_id] = affected

            inputs.append(
                WardInputs(
                    ward_id=ward.ward_id,
                    ward_name=ward.ward_name,
                    damage_counts=counts,
                    affected_population=affected,
                    facility_types=ward.facility_types,
                    road_disruption=ward.road_disruption,
                    distance_from_staging_km=ward.distance_from_staging_km,
                )
            )

        scored = score_wards(inputs)
        headline_types = priority_config().get(
            "headline_facility_types", ["hospital", "bridge", "fire_station"]
        )

        cards: list[WardCard] = []
        for ward in self.wards:
            counts = counts_by_ward[ward.ward_id]
            members = grouped.get(ward.ward_id, [])
            facility_counts = Counter(ward.facility_types)
            headline = sum(facility_counts[t] for t in headline_types)
            score, breakdown = scored.get(
                ward.ward_id, (0.0, PriorityBreakdown(damage=0, population=0, infrastructure=0, accessibility=0))
            )
            name, color = band_for(score)
            cards.append(
                WardCard(
                    ward_id=ward.ward_id,
                    ward_name=ward.ward_name,
                    total_buildings=len(members),
                    damaged_buildings=count_damaged(counts),
                    severely_damaged=count_severe(counts),
                    damage_breakdown=counts,
                    affected_population=population_by_ward[ward.ward_id],
                    critical_facilities=headline,
                    facility_breakdown=dict(facility_counts.most_common()),
                    facility_types=[t for t, _ in facility_counts.most_common()],
                    priority_score=score,
                    priority_band=name,
                    priority_color=color,
                    breakdown=breakdown,
                    reviewed_count=sum(1 for b in members if b.reviewed_by_human),
                )
            )

        cards.sort(key=lambda c: c.priority_score, reverse=True)
        return cards

    def feature_collection(self, ward_id: str | None = None) -> FeatureCollection:
        features = [
            Feature(
                geometry=b.geometry,
                properties=BuildingProperties(
                    id=b.id,
                    ward_id=b.ward_id,
                    damage_class=b.damage_class,
                    confidence=b.confidence,
                    reviewed_by_human=b.reviewed_by_human,
                    review_note=b.review_note,
                ),
            )
            for b in self.buildings
            if ward_id is None or b.ward_id == ward_id
        ]
        return FeatureCollection(features=features)

    def summary(self) -> AssessmentSummary:
        cards = self.ward_cards() if self.buildings else []
        return AssessmentSummary(
            assessment_id=self.id,
            status=self.status,
            mode=self.mode,
            geometry_source=self.geometry_source,
            aoi_name=self.aoi_name,
            created_at=self.created_at,
            total_buildings=len(self.buildings),
            damaged_buildings=sum(c.damaged_buildings for c in cards),
            severely_damaged=sum(c.severely_damaged for c in cards),
            affected_population=sum(c.affected_population for c in cards),
            wards=len(self.wards),
            error=self.error,
        )


class Store:
    def __init__(self) -> None:
        self._assessments: dict[str, Assessment] = {}
        self._building_index: dict[str, tuple[str, Building]] = {}

    def create(
        self,
        mode: InferenceMode,
        aoi_name: str | None = None,
        status: AssessmentStatus = AssessmentStatus.RUNNING,
    ) -> Assessment:
        assessment = Assessment(
            id=f"asmt_{uuid.uuid4().hex[:10]}",
            status=status,
            mode=mode,
            aoi_name=aoi_name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._assessments[assessment.id] = assessment
        return assessment

    def add(self, assessment: Assessment) -> None:
        self._assessments[assessment.id] = assessment
        self.reindex(assessment)

    def reindex(self, assessment: Assessment) -> None:
        for b in assessment.buildings:
            self._building_index[b.id] = (assessment.id, b)

    def get(self, assessment_id: str) -> Assessment | None:
        return self._assessments.get(assessment_id)

    def list(self) -> list[Assessment]:
        return sorted(self._assessments.values(), key=lambda a: a.created_at, reverse=True)

    def get_building(self, building_id: str) -> tuple[Assessment, Building] | None:
        hit = self._building_index.get(building_id)
        if hit is None:
            return None
        assessment_id, building = hit
        assessment = self._assessments.get(assessment_id)
        return (assessment, building) if assessment else None


store = Store()
