"""Reasoning layer (S4 needs, S5 graph, S8 optimiser) - pure functions, no DB.

The property tests are the CLAUDE.md section 7 requirements made executable:
never allocate more than exists, never route through a blocked road or node
unless air is verified, and every order carries a citation.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.allocation import (
    ZoneDemand,
    allocate,
    check_invariants,
    ration,
    to_orders,
)
from core.needs import compute_zone_needs
from core.schemas import DispatchOrder
from core.spatial.graph_network import (
    build_graph,
    flag_blocked_edges,
    reachability,
    road_route,
    validate_route,
)
from core.standards import BASELINE, baseline

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def node(nid, kind="zone", lon=85.3, lat=28.1, status="open", zone_id=None):
    return {
        "_id": nid,
        "name": nid,
        "kind": kind,
        "status": status,
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "zone_id": zone_id or (nid if kind == "zone" else None),
    }


def edge(eid, u, v, km=5.0, status="open", source="seed"):
    return {"_id": eid, "u": u, "v": v, "distance_km": km, "status": status, "source": source}


def zone(name, severity="destroyed", population=1000, score=0.6, destroyed=0):
    return {
        "_id": name,
        "name": name,
        "severity": severity,
        "population": population,
        "damage_score": score,
        "damage_extent": score,
        "buildings_destroyed": destroyed,
    }


# --------------------------------------------------------------------------
# S4 needs
# --------------------------------------------------------------------------


def test_undamaged_zone_needs_nothing():
    needs = compute_zone_needs(zone("z", severity="none"))
    assert needs["affected_people"] == 0
    assert all(v == 0 for v in needs["sku_demand"].values())


def test_affected_people_from_buildings_carries_its_arithmetic():
    needs = compute_zone_needs(zone("z", population=5000, score=0.01, destroyed=38))
    aff = needs["derivations"]["affected_people"]
    assert aff["value"] == 167  # ceil(38 x 4.37)
    assert "38 destroyed x 4.37 persons/household" in aff["formula"]
    assert any("hh_size=4.37" in a for a in aff["assumptions"])
    assert "https://censusnepal.cbs.gov.np/" in aff["sources"]


def test_affected_people_from_area_when_buildings_are_unresolvable():
    needs = compute_zone_needs(zone("z", population=2000, score=0.5, destroyed=0))
    assert needs["affected_people"] == 1000
    assert "damaged share (overhead)" in needs["derivations"]["affected_people"]["formula"]


def test_a_ground_photo_alone_does_not_declare_the_whole_ward_affected():
    """Regression: a ground override set the ward's score to 1.0 and needs read
    that as 100% of the population affected. One photo is one building."""
    z = {
        "severity": "destroyed",
        "population": 3200,
        "damage_score": 1.0,
        "damage_extent": None,
        "buildings_destroyed": 0,
    }
    needs = compute_zone_needs(z)
    assert needs["affected_people"] == 800  # 3200 x 0.25 assumed extent x 1.0
    formula = needs["derivations"]["affected_people"]["formula"]
    assert "no overhead coverage" in formula


def test_ground_override_keeps_the_overhead_extent():
    z = {
        "severity": "destroyed",
        "population": 3200,
        "damage_score": 1.0,
        "damage_extent": 0.3,
        "buildings_destroyed": 0,
    }
    assert compute_zone_needs(z)["affected_people"] == 960


def test_affected_people_never_exceeds_population():
    needs = compute_zone_needs(zone("z", population=100, score=0.2, destroyed=500))
    assert needs["affected_people"] == 100
    assert "capped at population" in needs["derivations"]["affected_people"]["formula"]


def test_sku_demand_follows_sphere_water_minimum():
    needs = compute_zone_needs(zone("z", population=2000, score=0.5))
    assert needs["water_litres"] == 1000 * 15
    assert needs["sku_demand"]["water_purifier"] == 15  # 15,000 L / 1,000 L per unit
    assert needs["sku_demand"]["rice_25kg"] == 112  # ceil(1000 x 0.4 x 7 / 25)


def test_changing_a_parameter_changes_the_need():
    params = baseline()
    params["patients_per_doctor_day"] = params["patients_per_doctor_day"].model_copy(
        update={"value": 25.0}
    )
    base = compute_zone_needs(zone("z", population=4000, score=0.5))
    tuned = compute_zone_needs(zone("z", population=4000, score=0.5), params)
    assert tuned["sku_demand"]["doctor"] == 2 * base["sku_demand"]["doctor"]


def test_every_baseline_standard_has_a_source_and_every_value_is_in_bounds():
    for p in BASELINE.values():
        assert p.bounds[0] <= p.value <= p.bounds[1], p.name
        if p.kind == "standard":
            assert p.citations, f"{p.name} is labelled a standard but cites nothing"


# --------------------------------------------------------------------------
# S5 graph
# --------------------------------------------------------------------------


def small_network(blocked_edge=False, blocked_node=False, degraded=False):
    nodes = [
        node("depot", "holding_center", 85.30, 28.10),
        node("a", lon=85.32, lat=28.11, status="blocked" if blocked_node else "open"),
        node("b", lon=85.34, lat=28.12),
        node("c", lon=85.36, lat=28.13),
    ]
    edges = [
        edge("e1", "depot", "a", 5, "degraded" if degraded else "open"),
        edge("e2", "a", "b", 5),
        edge("e3", "depot", "b", 20),
        edge("e4", "b", "c", 5, "blocked" if blocked_edge else "open"),
    ]
    return nodes, edges


def test_pyg_graph_holds_only_passable_roads_in_both_directions():
    nodes, edges = small_network(blocked_edge=True)
    g = build_graph(nodes, edges)
    assert g.data.num_nodes == 4
    assert g.data.edge_index.shape[1] == 2 * 3  # e4 removed, the rest bidirectional
    assert g.data.x.shape == (4, 5)


def test_shortest_route_prefers_the_cheaper_path():
    nodes, edges = small_network()
    r = road_route(build_graph(nodes, edges), "depot", "b")
    assert r.path == ["depot", "a", "b"] and r.distance_km == pytest.approx(10)


def test_degraded_roads_cost_more_and_can_change_the_route():
    nodes, edges = small_network(degraded=True)
    g = build_graph(nodes, edges, degraded_factor=5.0)
    r = road_route(g, "depot", "b")
    assert r.path == ["depot", "b"]  # 20 km open beats 5 km x5 + 5 km


def test_blocked_node_is_never_traversed():
    nodes, edges = small_network(blocked_node=True)
    g = build_graph(nodes, edges)
    r = road_route(g, "depot", "b")
    assert "a" not in r.path
    assert road_route(g, "depot", "a") is None


def test_blocked_road_makes_a_zone_unreachable_unless_air_is_verified():
    nodes, edges = small_network(blocked_edge=True)
    no_air = reachability(build_graph(nodes, edges))
    assert no_air["c"]["status"] == "unreachable"
    assert no_air["c"]["needs_air_verification"] is True
    with_air = reachability(build_graph(nodes, edges, air_verified=True))
    assert with_air["c"]["status"] == "air"
    assert with_air["c"]["best"]["path"] == ["depot", "c"]


def test_validate_route_catches_every_forbidden_route():
    nodes, edges = small_network(blocked_edge=True, blocked_node=True)
    assert validate_route({"mode": "road", "path": ["depot", "b", "c"]}, nodes, edges, False)
    assert validate_route({"mode": "road", "path": ["depot", "a", "b"]}, nodes, edges, False)
    assert validate_route({"mode": "road", "path": ["depot", "c"]}, nodes, edges, False)  # no road
    assert validate_route({"mode": "air", "path": ["depot", "c"]}, nodes, edges, False)
    assert not validate_route({"mode": "air", "path": ["depot", "c"]}, nodes, edges, True)
    assert not validate_route({"mode": "road", "path": ["depot", "b"]}, nodes, edges, False)


def _square(lon, lat, h=0.005):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - h, lat - h],
                [lon + h, lat - h],
                [lon + h, lat + h],
                [lon - h, lat + h],
                [lon - h, lat - h],
            ]
        ],
    }


def test_cv_blocks_roads_cut_between_wards_but_not_by_the_village_itself():
    nodes = [node("depot", "holding_center", 85.30, 28.10), node("v", lon=85.40, lat=28.10)]
    edges = [edge("road", "depot", "v", 10)]
    zones = [{"_id": "v", "geometry": _square(85.40, 28.10)}]
    in_village = {
        "severity": "destroyed",
        "confidence": 0.9,
        "centroid": {"coordinates": [85.40, 28.10]},
        "geometry": _square(85.40, 28.10, 0.001),
    }
    on_corridor = {
        "severity": "destroyed",
        "confidence": 0.9,
        "centroid": {"coordinates": [85.35, 28.10]},
        "geometry": _square(85.35, 28.10, 0.001),
    }
    assert flag_blocked_edges(nodes, edges, zones, [in_village]) == []
    flagged = flag_blocked_edges(nodes, edges, zones, [on_corridor])
    assert [f["edge_id"] for f in flagged] == ["road"]


def test_cv_never_overrides_an_operators_call():
    nodes = [node("depot", "holding_center", 85.30, 28.10), node("v", lon=85.40, lat=28.10)]
    edges = [edge("road", "depot", "v", 10, source="manual")]
    debris = {
        "severity": "destroyed",
        "confidence": 0.9,
        "centroid": {"coordinates": [85.35, 28.10]},
    }
    assert flag_blocked_edges(nodes, edges, [], [debris]) == []


# --------------------------------------------------------------------------
# S8 optimiser
# --------------------------------------------------------------------------


def test_rationing_spreads_scarce_stock_by_weighted_need():
    target = ration({"a": 100, "b": 100}, {"a": 3.0, "b": 1.0}, 100)
    assert sum(target.values()) == 100
    assert target["a"] == 75 and target["b"] == 25


def test_rationing_caps_at_demand_and_redistributes():
    target = ration({"a": 10, "b": 100}, {"a": 10.0, "b": 1.0}, 60)
    assert target == {"a": 10, "b": 50}


def test_rationing_gives_everyone_their_demand_when_supply_suffices():
    assert ration({"a": 5, "b": 7}, {"a": 1.0, "b": 1.0}, 100) == {"a": 5, "b": 7}


def _plan(nodes, edges, zones, stock, air=False):
    params = baseline()
    g = build_graph(nodes, edges, air_verified=air)
    result = allocate(zones, stock, g, params)
    return result, check_invariants(result, zones, stock, nodes, edges, air), params


def test_unreachable_zone_gets_nothing_and_is_reported():
    nodes, edges = small_network(blocked_edge=True)
    zones = [ZoneDemand("c", "c", "c", "destroyed", 500, {"doctor": 3})]
    result, problems, _ = _plan(nodes, edges, zones, {("depot", "doctor"): 10})
    assert not problems
    assert not result.allocations
    assert result.unmet[0].reason.startswith("no road access")


def test_air_verified_zone_is_served_by_air():
    nodes, edges = small_network(blocked_edge=True)
    zones = [ZoneDemand("c", "c", "c", "destroyed", 500, {"doctor": 3})]
    result, problems, params = _plan(nodes, edges, zones, {("depot", "doctor"): 10}, air=True)
    assert not problems
    assert [(a.route.mode, a.quantity) for a in result.allocations] == [("air", 3)]
    orders = to_orders(result, zones, params, {"doctor": "personnel"})
    assert orders[0]["transport_mode"] == "air"
    assert "helicopter" in orders[0]["rationale"]


def test_orders_validate_and_always_cite():
    nodes, edges = small_network()
    zones = [ZoneDemand("b", "b", "b", "major", 800, {"rice_25kg": 40, "doctor": 2})]
    result, problems, params = _plan(
        nodes, edges, zones, {("depot", "rice_25kg"): 100, ("depot", "doctor"): 5}
    )
    orders = to_orders(result, zones, params, {"rice_25kg": "food", "doctor": "personnel"})
    assert not problems and orders
    for o in orders:
        DispatchOrder.model_validate(o)
        assert any("spherestandards.org" in c["source_url"] for c in o["citations"])


# --- property tests -------------------------------------------------------

SKUS = ["doctor", "rice_25kg", "trauma_kit"]


@st.composite
def scenarios(draw):
    n_zones = draw(st.integers(1, 6))
    n_depots = draw(st.integers(1, 3))
    nodes = [node(f"d{i}", "staging_node", 85 + i * 0.1, 28.0) for i in range(n_depots)]
    nodes += [
        node(
            f"z{i}",
            lon=85 + i * 0.05,
            lat=28.2,
            status=draw(st.sampled_from(["open", "open", "open", "blocked"])),
        )
        for i in range(n_zones)
    ]
    ids = [n["_id"] for n in nodes]
    edges = []
    for k in range(draw(st.integers(0, 14))):
        u, v = draw(st.sampled_from(ids)), draw(st.sampled_from(ids))
        if u != v:
            edges.append(
                edge(
                    f"e{k}",
                    u,
                    v,
                    draw(st.floats(0.5, 40)),
                    draw(st.sampled_from(["open", "degraded", "blocked"])),
                )
            )
    zones = [
        ZoneDemand(
            f"z{i}",
            f"z{i}",
            f"z{i}",
            draw(st.sampled_from(["minor", "major", "destroyed"])),
            draw(st.integers(0, 5000)),
            {s: draw(st.integers(0, 60)) for s in SKUS},
        )
        for i in range(n_zones)
    ]
    stock = {(f"d{i}", s): draw(st.integers(0, 80)) for i in range(n_depots) for s in SKUS}
    air = draw(st.booleans())
    return nodes, edges, zones, stock, air


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_property_plans_never_break_the_invariants(scenario):
    nodes, edges, zones, stock, air = scenario
    result, problems, params = _plan(nodes, edges, zones, stock, air)
    assert problems == []

    # never more than exists, per (node, sku)
    used = {}
    for a in result.allocations:
        used[(a.source_node, a.sku)] = used.get((a.source_node, a.sku), 0) + a.quantity
    assert all(q <= stock[k] for k, q in used.items())

    # never through a blocked road or node; air only if verified
    for a in result.allocations:
        assert not validate_route(a.route, nodes, edges, air)
        if not air:
            assert a.route.mode == "road"

    # every order is a valid, cited DispatchOrder
    kinds = {"doctor": "personnel", "rice_25kg": "food", "trauma_kit": "medical"}
    for o in to_orders(result, zones, params, kinds):
        assert DispatchOrder.model_validate(o).citations


@settings(max_examples=200, deadline=None)
@given(
    st.dictionaries(st.sampled_from("abcdef"), st.integers(0, 500), min_size=1),
    st.integers(0, 2000),
    st.data(),
)
def test_property_rationing_is_bounded_and_uses_what_it_can(demand, supply, data):
    weight = {z: data.draw(st.floats(0.01, 100)) for z in demand}
    target = ration(demand, weight, supply)
    assert all(0 <= target[z] <= demand[z] for z in demand)
    assert sum(target.values()) == min(supply, sum(demand.values()))
