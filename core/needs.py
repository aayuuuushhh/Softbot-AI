"""S4: zone damage -> affected people -> per-resource need, with derivations.

Every figure carries the arithmetic that produced it, so a coordinator can ask
"why 171?" and get "38 destroyed x 4.5 persons/household", not a shrug.

Pure functions over plain dicts; `update_zone_needs` is the only DB-touching
entry point.
"""

from __future__ import annotations

import math

from core.schemas import Severity
from core.standards import Parameter, baseline


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def derivation(
    value: float,
    formula: str,
    params: dict[str, Parameter],
    used: list[str],
    extra_assumptions: list[str] | None = None,
) -> dict:
    """One explainable figure: value, the arithmetic, and where each input came from."""
    assumptions = [
        f"{params[n].name}={_fmt(params[n].value)} {params[n].unit} "
        f"({params[n].kind}{', agent-adjusted' if params[n].adjusted_by_agent else ''})"
        for n in used
    ]
    sources = sorted({c.source_url for n in used for c in params[n].citations})
    return {
        "value": value,
        "formula": formula,
        "assumptions": assumptions + (extra_assumptions or []),
        "sources": sources,
    }


def affected_people(zone: dict, params: dict[str, Parameter]) -> dict:
    """Estimate displaced / directly affected people in one zone.

    Two estimates, the larger wins (capped at population):
      buildings: destroyed structures x household size - only when the imagery
                 could resolve structures (buildings_destroyed > 0);
      area:      population x damage extent - the damaged share of the ward
                 measured from overhead imagery (S3 `damage_extent`).

    With no overhead evidence (ground photos only, e.g. under cloud), extent is
    unknown: a photo shows one building, not how much of the ward fell. The
    labelled `ground_only_extent` assumption, scaled by severity, stands in.
    """
    from core.vision.fusion import SEVERITY_SCORE

    population = int(zone.get("population", 0) or 0)
    severity = zone.get("severity", Severity.NONE.value)
    if severity == Severity.NONE.value or population <= 0:
        return derivation(0, f"severity={severity}, no affected population", params, [])

    hh = params["hh_size"].value
    destroyed = int(zone.get("buildings_destroyed", 0) or 0)
    extent = zone.get("damage_extent")

    extra: list[str] = []
    if extent is not None:
        share, share_text, share_used = (
            float(extent),
            f"{float(extent):.3f} damaged share (overhead)",
            [],
        )
        extra.append(
            "damaged share of ward taken as the affected share of population (planning assumption)"
        )
    else:
        sev_score = SEVERITY_SCORE.get(severity, 0.0)
        share = params["ground_only_extent"].value * sev_score
        share_text = (
            f"{params['ground_only_extent'].value:.2f} assumed extent x {sev_score:.2f} "
            f"({severity}) - no overhead coverage"
        )
        share_used = ["ground_only_extent"]

    by_buildings = destroyed * hh
    by_area = population * share
    if by_buildings >= by_area and destroyed > 0:
        raw, formula, used = (
            by_buildings,
            f"{destroyed} destroyed x {_fmt(hh)} persons/household",
            ["hh_size"],
        )
        extra = []
    else:
        raw, formula, used = by_area, f"{population:,} population x {share_text}", share_used

    value = min(population, math.ceil(raw))
    if value < raw:
        formula += f" = {_fmt(math.ceil(raw))}, capped at population {population:,}"
    return derivation(value, formula, params, used, extra)


def compute_zone_needs(zone: dict, params: dict[str, Parameter] | None = None) -> dict:
    """Needs for one zone, as {field: derivation} plus per-SKU demand.

    SKU demand is what the allocator matches against inventory; the named
    fields are what humans read.
    """
    params = params or baseline()
    v = {k: p.value for k, p in params.items()}
    aff = affected_people(zone, params)
    n = aff["value"]
    days = v["planning_horizon_days"]
    destroyed = int(zone.get("buildings_destroyed", 0) or 0)
    # Without a resolvable building count, approximate structures from people.
    structures = destroyed if destroyed > 0 else math.ceil(n / v["hh_size"]) if n else 0
    structure_note = (
        []
        if destroyed > 0
        else [
            "building count unresolvable from imagery; structures approximated as affected / hh_size"
        ]
    )

    injured = math.ceil(n * v["injury_rate"])
    water_l = math.ceil(n * v["water_l_per_person_day"])
    rice_kg = n * v["rice_kg_per_person_day"] * days
    doctors = math.ceil(injured / v["patients_per_doctor_day"]) if injured else 0
    rescuers = math.ceil(structures * v["rescuers_per_destroyed"]) if structures else 0
    engineers = math.ceil(structures * v["engineers_per_damaged"]) if structures else 0

    d = {
        "affected_people": aff,
        "injured": derivation(
            injured, f"{n:,} affected x {v['injury_rate']:.2f}", params, ["injury_rate"]
        ),
        "water_litres_per_day": derivation(
            water_l,
            f"{n:,} affected x {_fmt(v['water_l_per_person_day'])} L/person/day",
            params,
            ["water_l_per_person_day"],
        ),
        "food_rice_kg": derivation(
            round(rice_kg, 1),
            f"{n:,} affected x {v['rice_kg_per_person_day']:.2f} kg/day x {_fmt(days)} days",
            params,
            ["rice_kg_per_person_day", "planning_horizon_days"],
        ),
        "doctors": derivation(
            doctors,
            f"ceil({injured:,} injured / {_fmt(v['patients_per_doctor_day'])} patients/doctor/day)",
            params,
            ["patients_per_doctor_day", "injury_rate"],
        ),
        "rescue_personnel": derivation(
            rescuers,
            f"ceil({structures:,} destroyed structures x {v['rescuers_per_destroyed']})",
            params,
            ["rescuers_per_destroyed"],
            structure_note,
        ),
        "engineers": derivation(
            engineers,
            f"ceil({structures:,} damaged structures x {v['engineers_per_damaged']})",
            params,
            ["engineers_per_damaged"],
            structure_note,
        ),
    }

    sku_demand = {
        "doctor": doctors,
        "rescue_personnel": rescuers,
        "engineer": engineers,
        "rice_25kg": math.ceil(rice_kg / 25.0) if rice_kg else 0,
        "water_purifier": math.ceil(water_l / v["purifier_l_per_day"]) if water_l else 0,
        "trauma_kit": math.ceil(injured * v["trauma_kits_per_injured"]) if injured else 0,
    }
    d["sku_demand"] = {
        "rice_25kg": derivation(
            sku_demand["rice_25kg"],
            f"ceil({rice_kg:,.1f} kg / 25 kg per sack)",
            params,
            ["rice_kg_per_person_day", "planning_horizon_days"],
        ),
        "water_purifier": derivation(
            sku_demand["water_purifier"],
            f"ceil({water_l:,} L/day / {_fmt(v['purifier_l_per_day'])} L/day per unit)",
            params,
            ["purifier_l_per_day", "water_l_per_person_day"],
        ),
        "trauma_kit": derivation(
            sku_demand["trauma_kit"],
            f"ceil({injured:,} injured x {v['trauma_kits_per_injured']} kits)",
            params,
            ["trauma_kits_per_injured", "injury_rate"],
        ),
        "doctor": d["doctors"],
        "rescue_personnel": d["rescue_personnel"],
        "engineer": d["engineers"],
    }

    return {
        # Summary fields kept for the Zone.needs schema and the dashboard.
        "personnel": doctors + rescuers + engineers,
        "food_rations": math.ceil(n * days),  # person-days of food
        "water_litres": water_l,
        "medical_kits": sku_demand["trauma_kit"],
        "affected_people": n,
        "sku_demand": sku_demand,
        "derivations": d,
    }


async def update_zone_needs(
    event_id: str, params: dict[str, Parameter] | None = None
) -> list[dict]:
    """Recompute and store needs for every zone of an event (baseline parameters by default)."""
    from core import db

    database = db.get_db()
    out = []
    async for zone in database[db.ZONES].find({"event_id": event_id}):
        needs = compute_zone_needs(zone, params)
        await database[db.ZONES].update_one({"_id": zone["_id"]}, {"$set": {"needs": needs}})
        out.append({"zone_id": str(zone["_id"]), "name": zone.get("name"), **needs})
    return out
