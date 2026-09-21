"""Baseline planning parameters for need computation and allocation (S4/S8).

Every parameter carries where it came from. Two kinds:

* `standard`   - a published humanitarian minimum (Sphere, WHO, census).
* `assumption` - a planning placeholder with no single authoritative figure.
                 Shown as such in the UI and PDF, and the first thing the
                 research agent (agents/researcher.py) is asked to replace
                 with evidence from comparable past disasters.

CLAUDE.md: allocation "must not rely exclusively on hardcoded heuristics". These
are the starting point the agent adapts, not the answer. The agent may override a
parameter only by citing a retrieved source, and only within `bounds`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.schemas import Citation

SPHERE = Citation(
    claim="Sphere Handbook 2018: minimum standards for water (15 L/person/day), "
    "food (2,100 kcal/person/day) and covered living space (3.5 m2/person).",
    source_title="The Sphere Handbook: Humanitarian Charter and Minimum Standards (2018)",
    source_url="https://spherestandards.org/handbook/",
    published="2018",
)
WHO_EMT = Citation(
    claim="WHO Emergency Medical Team classification: a Type 1 mobile team provides "
    "outpatient emergency care for at least 50 patients per day.",
    source_title="WHO Classification and Minimum Standards for Emergency Medical Teams",
    source_url="https://extranet.who.int/emt/",
    published="2021",
)
NEPAL_CENSUS = Citation(
    claim="Nepal National Population and Housing Census 2021: national average "
    "household size 4.37 persons.",
    source_title="National Population and Housing Census 2021, National Statistics Office",
    source_url="https://censusnepal.cbs.gov.np/",
    published="2021",
)


class Parameter(BaseModel):
    """One tunable planning number, with provenance."""

    name: str
    value: float
    unit: str
    kind: Literal["standard", "assumption"]
    description: str
    bounds: tuple[float, float] = Field(description="Hard limits the agent cannot exceed")
    citations: list[Citation] = Field(default_factory=list)
    adjusted_by_agent: bool = False
    note: str | None = None


def _p(name, value, unit, kind, description, bounds, citations=()) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        unit=unit,
        kind=kind,
        description=description,
        bounds=bounds,
        citations=list(citations),
    )


BASELINE: dict[str, Parameter] = {
    p.name: p
    for p in [
        # --- population ---
        _p(
            "hh_size",
            4.37,
            "persons/household",
            "standard",
            "Average household size, used to turn destroyed buildings into displaced people. "
            "National figure - override per district where known.",
            (2.0, 9.0),
            [NEPAL_CENSUS],
        ),
        _p(
            "ground_only_extent",
            0.25,
            "share of ward",
            "assumption",
            "Share of a ward assumed affected at full severity when damage is known only from "
            "ground photos (no overhead coverage, e.g. under cloud). Scaled by severity.",
            (0.05, 1.0),
        ),
        _p(
            "injury_rate",
            0.10,
            "fraction of affected",
            "assumption",
            "Share of affected people needing medical care.",
            (0.01, 0.5),
        ),
        # --- water / food ---
        _p(
            "water_l_per_person_day",
            15.0,
            "L/person/day",
            "standard",
            "Minimum water for drinking, cooking and hygiene.",
            (7.5, 50.0),
            [SPHERE],
        ),
        _p(
            "rice_kg_per_person_day",
            0.40,
            "kg/person/day",
            "assumption",
            "Cereal component of a general ration contributing to the Sphere 2,100 kcal minimum.",
            (0.25, 0.6),
            [SPHERE],
        ),
        _p(
            "planning_horizon_days",
            7.0,
            "days",
            "assumption",
            "How many days of supply one allocation round covers.",
            (1.0, 30.0),
        ),
        _p(
            "purifier_l_per_day",
            1000.0,
            "L/day per water_purifier unit",
            "assumption",
            "Output of one water_purifier SKU. Replace with the real unit spec.",
            (100.0, 20000.0),
        ),
        # --- medical ---
        _p(
            "patients_per_doctor_day",
            50.0,
            "patients/doctor/day",
            "standard",
            "Outpatient throughput, from the WHO EMT Type 1 minimum.",
            (10.0, 150.0),
            [WHO_EMT],
        ),
        _p(
            "trauma_kits_per_injured",
            1.0,
            "kits/injured person",
            "assumption",
            "Trauma kits consumed per injured person over the planning horizon.",
            (0.1, 3.0),
        ),
        # --- search, rescue, assessment ---
        _p(
            "rescuers_per_destroyed",
            0.1,
            "personnel/destroyed building",
            "assumption",
            "Search-and-rescue personnel per destroyed structure (1 per 10).",
            (0.01, 5.0),
        ),
        _p(
            "engineers_per_damaged",
            0.04,
            "engineers/damaged building",
            "assumption",
            "Structural assessors per damaged building (1 per 25).",
            (0.005, 0.5),
        ),
        # --- routing / prioritisation ---
        _p(
            "degraded_road_factor",
            2.5,
            "cost multiplier",
            "assumption",
            "Travel-cost multiplier on a degraded road relative to an open one.",
            (1.0, 10.0),
        ),
        _p(
            "air_cost_factor",
            4.0,
            "cost multiplier",
            "assumption",
            "Cost of 1 km by helicopter relative to 1 km of open road. Air is used only "
            "when the event has air transport verified.",
            (1.0, 50.0),
        ),
        _p(
            "isolation_priority_boost",
            1.5,
            "multiplier",
            "assumption",
            "Priority multiplier for a zone with no road access - the villages that cannot "
            "call for help.",
            (1.0, 5.0),
        ),
    ]
}

# Zone severity -> priority weight. Destroyed outranks major by more than its
# score alone because collapse drives mortality in the first 72 h.
SEVERITY_PRIORITY = {"none": 0.0, "minor": 0.5, "major": 1.5, "destroyed": 3.0}


def baseline() -> dict[str, Parameter]:
    """A fresh, mutable copy of the baseline parameter set."""
    return {k: v.model_copy(deep=True) for k, v in BASELINE.items()}


def values(params: dict[str, Parameter]) -> dict[str, float]:
    return {k: p.value for k, p in params.items()}
