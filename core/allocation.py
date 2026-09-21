"""S8 (deterministic core): match available stock to zone need over the road graph.

The agent (agents/allocator.py) decides the *parameters* - how much each zone
needs, how roads are weighted, how isolation is prioritised - by adapting the
baseline to cited historical evidence. This module turns a parameter set into
dispatch orders, and it is deliberately free of any LLM so its guarantees can be
property-tested:

  1. never allocates more of a (node, sku) than is available;
  2. never sends a zone more of a sku than it needs;
  3. every route passes graph_network.validate_route - no blocked road, no
     blocked node, no air unless verified;
  4. every order carries at least one citation.

Algorithm, per SKU:
  * weight each reachable zone by severity x affected people x isolation;
  * if supply covers demand, every zone gets its demand; otherwise ration by
    weighted share, capped at demand, redistributing what capped zones leave
    ("water-filling") - scarce stock is spread by need, not handed wholesale to
    the single worst zone;
  * fill each zone's share from its cheapest reachable sources first.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from core.schemas import Citation
from core.spatial.graph_network import RoadGraph, Route, best_route
from core.standards import SEVERITY_PRIORITY, SPHERE, Parameter

# Which planning parameters drive each SKU's demand - their citations travel
# with every order that ships that SKU.
SKU_PARAMETERS: dict[str, list[str]] = {
    "doctor": ["patients_per_doctor_day", "injury_rate", "hh_size"],
    "trauma_kit": ["trauma_kits_per_injured", "injury_rate", "hh_size"],
    "rice_25kg": ["rice_kg_per_person_day", "planning_horizon_days", "hh_size"],
    "water_purifier": ["water_l_per_person_day", "purifier_l_per_day", "hh_size"],
    "rescue_personnel": ["rescuers_per_destroyed", "hh_size"],
    "engineer": ["engineers_per_damaged", "hh_size"],
}

# The allocation principle itself: assistance prioritised by need alone.
NEED_BASED_PRIORITY = Citation(
    claim="Humanitarian Charter (Sphere): assistance is provided impartially and "
    "prioritised on the basis of need alone.",
    source_title=SPHERE.source_title,
    source_url=SPHERE.source_url,
    published=SPHERE.published,
)


@dataclass
class ZoneDemand:
    zone_id: str
    node_id: str
    name: str
    severity: str
    affected: int
    sku_demand: dict[str, int]


@dataclass
class Allocation:
    zone_id: str
    source_node: str
    sku: str
    quantity: int
    route: Route


@dataclass
class Unmet:
    zone_id: str
    name: str
    sku: str
    shortfall: int
    reason: str


@dataclass
class AllocationResult:
    allocations: list[Allocation] = field(default_factory=list)
    unmet: list[Unmet] = field(default_factory=list)
    zone_rank: list[dict] = field(default_factory=list)
    ignored_skus: list[str] = field(default_factory=list)


def zone_weight(z: ZoneDemand, isolated: bool, params: dict[str, Parameter]) -> float:
    w = SEVERITY_PRIORITY.get(z.severity, 0.0) * max(z.affected, 1)
    if isolated:
        w *= params["isolation_priority_boost"].value
    return w


def ration(demand: dict[str, int], weight: dict[str, float], supply: int) -> dict[str, int]:
    """Integer water-filling: weighted shares of `supply`, each capped at demand."""
    target = {z: 0 for z in demand}
    active = {z for z, d in demand.items() if d > 0}
    remaining = supply
    if sum(demand[z] for z in active) <= supply:
        return {z: demand[z] for z in demand}

    while active and remaining > 0:
        total_w = sum(weight[z] * demand[z] for z in active) or float(len(active))
        shares = {
            z: remaining * ((weight[z] * demand[z]) / total_w if total_w else 1 / len(active))
            for z in active
        }
        capped = [z for z in active if target[z] + shares[z] >= demand[z]]
        if not capped:
            for z in active:
                add = int(math.floor(shares[z]))
                target[z] += add
                remaining -= add
            break
        for z in capped:
            remaining -= demand[z] - target[z]
            target[z] = demand[z]
            active.discard(z)

    # Units lost to flooring go one at a time, highest priority first.
    order = sorted((z for z in demand if target[z] < demand[z]), key=lambda z: -weight[z])
    while remaining > 0 and order:
        progressed = False
        for z in order:
            if remaining == 0:
                break
            if target[z] < demand[z]:
                target[z] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return target


def allocate(
    zones: list[ZoneDemand],
    stock: dict[tuple[str, str], int],
    graph: RoadGraph,
    params: dict[str, Parameter],
) -> AllocationResult:
    result = AllocationResult()
    stock = {k: max(0, int(v)) for k, v in stock.items()}  # defensive copy
    cache: dict = {}
    sources = graph.supply_nodes

    # Routes from every source to every zone, computed once.
    routes: dict[str, list[tuple[str, Route]]] = {}
    for z in zones:
        options = [(s, r) for s in sources if (r := best_route(graph, s, z.node_id, cache))]
        routes[z.zone_id] = sorted(options, key=lambda sr: sr[1].cost)

    weights: dict[str, float] = {}
    for z in zones:
        opts = routes[z.zone_id]
        isolated = bool(opts) and all(r.mode == "air" for _, r in opts)
        weights[z.zone_id] = zone_weight(z, isolated, params)

    ranked = sorted(zones, key=lambda z: -weights[z.zone_id])
    for rank, z in enumerate(ranked, start=1):
        opts = routes[z.zone_id]
        result.zone_rank.append(
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "rank": rank,
                "weight": round(weights[z.zone_id], 2),
                "severity": z.severity,
                "affected": z.affected,
                "access": "unreachable"
                if not opts
                else ("air" if opts[0][1].mode == "air" else "road"),
            }
        )

    by_id = {z.zone_id: z for z in zones}
    skus = sorted({s for z in zones for s, d in z.sku_demand.items() if d > 0})
    stocked = {sku for (_, sku) in stock}
    result.ignored_skus = sorted(stocked - set(SKU_PARAMETERS))

    for sku in skus:
        demand = {z.zone_id: int(z.sku_demand.get(sku, 0)) for z in zones}
        for z in zones:
            if demand[z.zone_id] > 0 and not routes[z.zone_id]:
                result.unmet.append(
                    Unmet(
                        z.zone_id,
                        z.name,
                        sku,
                        demand[z.zone_id],
                        "no road access and air transport not verified",
                    )
                )
                demand[z.zone_id] = 0

        reachable_sources = {s for opts in routes.values() for s, _ in opts}
        supply = sum(q for (n, s), q in stock.items() if s == sku and n in reachable_sources)
        target = ration(demand, weights, supply)

        for z in ranked:
            want = target.get(z.zone_id, 0)
            for source, route in routes[z.zone_id]:
                if want <= 0:
                    break
                have = stock.get((source, sku), 0)
                take = min(want, have)
                if take > 0:
                    stock[(source, sku)] = have - take
                    want -= take
                    result.allocations.append(Allocation(z.zone_id, source, sku, take, route))
            short = demand[z.zone_id] - (target.get(z.zone_id, 0) - want)
            if short > 0:
                reason = (
                    "stock at the sources that can reach this zone is exhausted"
                    if want > 0
                    else "rationed: total stock is below total need"
                )
                result.unmet.append(Unmet(z.zone_id, by_id[z.zone_id].name, sku, short, reason))
    return result


def citations_for(skus: list[str], params: dict[str, Parameter]) -> list[Citation]:
    seen, out = set(), [NEED_BASED_PRIORITY]
    seen.add((NEED_BASED_PRIORITY.source_url, NEED_BASED_PRIORITY.claim))
    for sku in skus:
        for name in SKU_PARAMETERS.get(sku, []):
            for c in params[name].citations if name in params else []:
                key = (c.source_url, c.claim)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
    # Agent-sourced evidence first: it is what changed from the baseline.
    return sorted(out, key=lambda c: c is NEED_BASED_PRIORITY or "Sphere" in c.source_title)


def to_orders(
    result: AllocationResult,
    zones: list[ZoneDemand],
    params: dict[str, Parameter],
    sku_kind: dict[str, str],
) -> list[dict]:
    """Group allocations into DispatchOrder-shaped dicts, one per (zone, source, mode)."""
    by_zone = {z.zone_id: z for z in zones}
    rank = {r["zone_id"]: r["rank"] for r in result.zone_rank}
    n_zones = max(len(rank), 1)
    groups: dict[tuple[str, str, str], list[Allocation]] = defaultdict(list)
    for a in result.allocations:
        groups[(a.zone_id, a.source_node, a.route.mode)].append(a)

    orders = []
    for (zone_id, source, mode), allocs in groups.items():
        z = by_zone[zone_id]
        route = allocs[0].route
        items = [
            {"kind": sku_kind.get(a.sku, "medical"), "sku": a.sku, "quantity": a.quantity}
            for a in sorted(allocs, key=lambda a: a.sku)
        ]
        adjusted = sorted(
            {
                n
                for a in allocs
                for n in SKU_PARAMETERS.get(a.sku, [])
                if n in params and params[n].adjusted_by_agent
            }
        )
        uncited = sorted(
            {
                n
                for a in allocs
                for n in SKU_PARAMETERS.get(a.sku, [])
                if n in params and not params[n].citations
            }
        )
        via = (
            f"by helicopter ({route.distance_km:.1f} km direct; no open road)"
            if mode == "air"
            else f"by road, {route.distance_km:.1f} km"
            + (f" ({route.degraded_km:.1f} km degraded)" if route.degraded_km else "")
        )
        shipment = ", ".join(f"{i['quantity']} {i['sku']}" for i in items)
        rationale = (
            f"{z.name}: {z.severity}, {z.affected:,} affected, priority rank {rank[zone_id]}/{n_zones}. "
            f"Ship {shipment} {via}."
        )
        if adjusted:
            rationale += f" Quantities use agent-adjusted parameters: {', '.join(adjusted)}."
        if uncited:
            rationale += f" Planning assumptions without a source: {', '.join(uncited)}."
        priority = 1 + math.floor(9 * (rank[zone_id] - 1) / max(n_zones - 1, 1))
        orders.append(
            {
                "zone_id": zone_id,
                "from_node_id": source,
                "items": items,
                "transport_mode": mode,
                "route": route.path,
                "route_distance_km": round(route.distance_km, 2),
                "priority": int(min(max(priority, 1), 10)),
                "rationale": rationale,
                "citations": [
                    c.model_dump() for c in citations_for([a.sku for a in allocs], params)
                ],
            }
        )
    return sorted(orders, key=lambda o: (o["priority"], o["zone_id"]))


def check_invariants(
    result: AllocationResult,
    zones: list[ZoneDemand],
    stock: dict[tuple[str, str], int],
    nodes: list[dict],
    edges: list[dict],
    air_verified: bool,
) -> list[str]:
    """Re-verify properties 1-3 from scratch. Empty list = plan is sound."""
    from core.spatial.graph_network import validate_route

    problems: list[str] = []
    used: dict[tuple[str, str], int] = defaultdict(int)
    given: dict[tuple[str, str], int] = defaultdict(int)
    for a in result.allocations:
        used[(a.source_node, a.sku)] += a.quantity
        given[(a.zone_id, a.sku)] += a.quantity
        if a.quantity <= 0:
            problems.append(f"non-positive quantity {a.quantity} of {a.sku}")
        for p in validate_route(a.route, nodes, edges, air_verified):
            problems.append(f"{a.sku} to zone {a.zone_id}: {p}")
    for key, q in used.items():
        if q > stock.get(key, 0):
            problems.append(
                f"over-allocated {key[1]} at {key[0]}: {q} > available {stock.get(key, 0)}"
            )
    demand = {(z.zone_id, s): d for z in zones for s, d in z.sku_demand.items()}
    for key, q in given.items():
        if q > demand.get(key, 0):
            problems.append(f"zone {key[0]} sent {q} {key[1]}, needs only {demand.get(key, 0)}")
    return problems
