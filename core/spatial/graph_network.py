"""S5: road network as a PyTorch Geometric graph - reachability and routing.

Representation: one `torch_geometric.data.Data` per event.
  x          (N, 5)  [lon, lat, is_supply, is_zone, capacity]
  edge_index (2, 2E) every *passable* road, both directions
  edge_attr  (2E, 3) [distance_km, status_code, travel_cost]

Hard rules (CLAUDE.md section 7, "Graph Connectivity"):
  * a `blocked` road is not in edge_index at all - removed, never penalised;
  * a `blocked` node has every incident road removed, so nothing is routed
    *through* it, and it can only be delivered to by air;
  * air transport is a separate mode, available only when the event has
    `air_transport_verified`. It flies supply node -> destination directly and
    never touches the road graph.

`validate_route` re-checks a route against the raw road list, independently of
how it was produced; the allocator and the property tests both call it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

STATUS_CODE = {"open": 0, "degraded": 1, "blocked": 2}
SUPPLY_KINDS = {"holding_center", "staging_node"}


def haversine_km(a: list[float], b: list[float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


@dataclass
class Route:
    mode: Literal["road", "air"]
    path: list[str]  # node ids, origin -> destination
    distance_km: float
    cost: float
    edge_ids: list[str] = field(default_factory=list)
    degraded_km: float = 0.0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "path": self.path,
            "distance_km": round(self.distance_km, 2),
            "cost": round(self.cost, 2),
            "edge_ids": self.edge_ids,
            "degraded_km": round(self.degraded_km, 2),
        }


@dataclass
class RoadGraph:
    data: Data
    node_ids: list[str]
    index: dict[str, int]
    nodes: dict[str, dict]
    edges: list[dict]
    air_verified: bool
    air_cost_factor: float
    # Adjacency derived from data.edge_index: i -> [(j, cost, distance, edge_id, degraded)]
    adj: list[list[tuple[int, float, float, str, bool]]] = field(default_factory=list)

    @property
    def supply_nodes(self) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n.get("kind") in SUPPLY_KINDS]

    def node_blocked(self, node_id: str) -> bool:
        return self.nodes[node_id].get("status", "open") == "blocked"


def _sid(value) -> str:
    return str(value)


def build_graph(
    nodes: list[dict],
    edges: list[dict],
    *,
    degraded_factor: float = 2.5,
    air_verified: bool = False,
    air_cost_factor: float = 4.0,
) -> RoadGraph:
    """Build the passable road graph. `nodes`/`edges` are DB documents (or dicts
    with the same keys); ids may be ObjectId or str."""
    node_map = {_sid(n.get("_id", n.get("id"))): n for n in nodes}
    node_ids = list(node_map)
    index = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.tensor(
        [
            [
                n["location"]["coordinates"][0],
                n["location"]["coordinates"][1],
                float(n.get("kind") in SUPPLY_KINDS),
                float(n.get("kind") == "zone"),
                float(n.get("capacity", 0) or 0),
            ]
            for n in node_map.values()
        ],
        dtype=torch.float32,
    ).reshape(-1, 5)

    # (a, b) -> (cost, km, status_code, edge_id); parallel roads keep the cheapest.
    best: dict[tuple[int, int], tuple[float, float, int, str]] = {}
    for e in edges:
        u, v = _sid(e["u"]), _sid(e["v"])
        if u not in index or v not in index:
            continue
        status = e.get("status", "open")
        if status == "blocked":
            continue  # hard removal
        if node_map[u].get("status") == "blocked" or node_map[v].get("status") == "blocked":
            continue  # never route into or through a blocked node by road
        km = float(e["distance_km"])
        cost = km * (degraded_factor if status == "degraded" else 1.0)
        eid = _sid(e.get("_id", e.get("id")))
        for a, b in ((index[u], index[v]), (index[v], index[u])):
            if (a, b) not in best or cost < best[(a, b)][0]:
                best[(a, b)] = (cost, km, STATUS_CODE[status], eid)

    pairs = sorted(best)
    if pairs:
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(
            [[best[p][1], float(best[p][2]), best[p][0]] for p in pairs], dtype=torch.float32
        )
        edge_index, edge_attr = coalesce(
            edge_index, edge_attr, num_nodes=len(node_ids), reduce="min"
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_ids))
    adj: list[list[tuple[int, float, float, str, bool]]] = [[] for _ in node_ids]
    for k in range(edge_index.shape[1]):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        cost, km, code, eid = best[(a, b)]
        adj[a].append((b, cost, km, eid, code == STATUS_CODE["degraded"]))

    return RoadGraph(
        data=data,
        node_ids=node_ids,
        index=index,
        nodes=node_map,
        edges=list(edges),
        air_verified=air_verified,
        air_cost_factor=air_cost_factor,
        adj=adj,
    )


def shortest_paths(
    graph: RoadGraph, source: str
) -> tuple[dict[int, float], dict[int, tuple[int, str]]]:
    """Dijkstra over the PyG adjacency. Returns (cost by node index, predecessor map)."""
    start = graph.index[source]
    dist = {start: 0.0}
    prev: dict[int, tuple[int, str]] = {}
    heap = [(0.0, start)]
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist.get(i, math.inf):
            continue
        for j, cost, _km, eid, _deg in graph.adj[i]:
            nd = d + cost
            if nd < dist.get(j, math.inf):
                dist[j] = nd
                prev[j] = (i, eid)
                heapq.heappush(heap, (nd, j))
    return dist, prev


def road_route(
    graph: RoadGraph, source: str, target: str, _cache: dict | None = None
) -> Route | None:
    if source not in graph.index or target not in graph.index:
        return None
    if _cache is not None and source in _cache:
        dist, prev = _cache[source]
    else:
        dist, prev = shortest_paths(graph, source)
        if _cache is not None:
            _cache[source] = (dist, prev)
    t = graph.index[target]
    if t not in dist:
        return None
    path_idx, eids = [t], []
    while path_idx[-1] != graph.index[source]:
        p, eid = prev[path_idx[-1]]
        eids.append(eid)
        path_idx.append(p)
    path_idx.reverse()
    eids.reverse()

    km = degraded_km = 0.0
    for a, b in zip(path_idx, path_idx[1:], strict=False):
        _j, _c, dkm, _eid, deg = next(t for t in graph.adj[a] if t[0] == b)
        km += dkm
        degraded_km += dkm if deg else 0.0
    return Route("road", [graph.node_ids[i] for i in path_idx], km, dist[t], eids, degraded_km)


def air_route(graph: RoadGraph, source: str, target: str) -> Route | None:
    if not graph.air_verified:
        return None
    km = haversine_km(
        graph.nodes[source]["location"]["coordinates"],
        graph.nodes[target]["location"]["coordinates"],
    )
    return Route("air", [source, target], km, km * graph.air_cost_factor)


def best_route(
    graph: RoadGraph, source: str, target: str, _cache: dict | None = None
) -> Route | None:
    """Road if one exists, else air if verified, else None (unreachable)."""
    return road_route(graph, source, target, _cache) or air_route(graph, source, target)


def reachability(graph: RoadGraph, targets: list[str] | None = None) -> dict[str, dict]:
    """For each target node: how it can be reached from any supply node.

    status: "road" | "air" | "unreachable". `unreachable` with
    `needs_air_verification: true` is the actionable case - a helicopter
    problem, not a paperwork problem.
    """
    targets = targets or [nid for nid, n in graph.nodes.items() if n.get("kind") == "zone"]
    cache: dict = {}
    out = {}
    for t in targets:
        routes = [r for s in graph.supply_nodes if (r := road_route(graph, s, t, cache))]
        if routes:
            best = min(routes, key=lambda r: r.cost)
            out[t] = {
                "status": "road",
                "best": best.as_dict(),
                "sources": sorted({r.path[0] for r in routes}),
            }
        elif graph.air_verified and graph.supply_nodes:
            best = min((air_route(graph, s, t) for s in graph.supply_nodes), key=lambda r: r.cost)
            out[t] = {"status": "air", "best": best.as_dict(), "sources": graph.supply_nodes}
        else:
            out[t] = {
                "status": "unreachable",
                "best": None,
                "sources": [],
                "needs_air_verification": not graph.air_verified,
            }
    return out


def validate_route(
    route: dict | Route, nodes: list[dict], edges: list[dict], air_verified: bool
) -> list[str]:
    """Independent check of a route against the raw network. Returns violations.

    Deliberately does not reuse RoadGraph: a bug in graph construction must not
    be able to vouch for itself.
    """
    r = route.as_dict() if isinstance(route, Route) else route
    node_map = {_sid(n.get("_id", n.get("id"))): n for n in nodes}
    path = [str(p) for p in r["path"]]
    problems: list[str] = []
    if len(path) < 2:
        return ["route has fewer than two nodes"]
    if r["mode"] == "air":
        if not air_verified:
            problems.append("air route used but air transport is not verified for this event")
        if len(path) != 2:
            problems.append("air route must be a direct origin -> destination hop")
        return problems

    for mid in path[1:-1]:
        if node_map.get(mid, {}).get("status") == "blocked":
            problems.append(f"road route passes through blocked node {mid}")
    if node_map.get(path[-1], {}).get("status") == "blocked":
        problems.append(f"road route delivers to blocked node {path[-1]}")
    if node_map.get(path[0], {}).get("status") == "blocked":
        problems.append(f"road route departs blocked node {path[0]}")

    passable: dict[frozenset, bool] = {}
    for e in edges:
        key = frozenset((_sid(e["u"]), _sid(e["v"])))
        ok = e.get("status", "open") != "blocked"
        passable[key] = passable.get(key, False) or ok
    for a, b in zip(path, path[1:], strict=False):
        key = frozenset((a, b))
        if key not in passable:
            problems.append(f"no road between {a} and {b}")
        elif not passable[key]:
            problems.append(f"road route uses blocked road {a} - {b}")
    return problems


# --------------------------------------------------------------------------
# CV -> graph: roads cut by detected damage
# --------------------------------------------------------------------------


def flag_blocked_edges(
    nodes: list[dict],
    edges: list[dict],
    zones: list[dict],
    detections: list[dict],
    buffer_m: float = 40.0,
    severities: tuple[str, ...] = ("major", "destroyed"),
) -> list[dict]:
    """Roads whose corridor is crossed by major/destroyed damage outside any ward.

    The stretch of a road inside its endpoint wards is the "last mile" through
    the village itself - damage there is the village's damage, already counted,
    not evidence that the corridor is cut. Only the between-ward corridor is
    tested, so a damaged village does not also get its own access road blocked.

    Never touches edges an operator set by hand (`source == "manual"`) and never
    downgrades a status. Returns [{edge_id, reason}].
    """
    from shapely.geometry import LineString, shape
    from shapely.geometry import Point as SPoint
    from shapely.ops import unary_union

    node_map = {_sid(n.get("_id", n.get("id"))): n for n in nodes}
    zone_polys = unary_union([shape(z["geometry"]) for z in zones]) if zones else None
    damaged = [d for d in detections if d.get("severity") in severities]
    if not damaged:
        return []

    out = []
    for e in edges:
        if e.get("status") == "blocked" or e.get("source") == "manual":
            continue
        u, v = node_map.get(_sid(e["u"])), node_map.get(_sid(e["v"]))
        if not u or not v:
            continue
        line = LineString([u["location"]["coordinates"], v["location"]["coordinates"]])
        corridor = line.difference(zone_polys) if zone_polys is not None else line
        if corridor.is_empty:
            continue
        lat = line.centroid.y
        # Metres -> degrees; lon degrees shrink with latitude.
        buf_deg = buffer_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))
        corridor_zone = corridor.buffer(buf_deg)
        for d in damaged:
            geom = (
                shape(d["geometry"]) if d.get("geometry") else SPoint(d["centroid"]["coordinates"])
            )
            if corridor_zone.intersects(geom):
                out.append(
                    {
                        "edge_id": _sid(e.get("_id", e.get("id"))),
                        "reason": f"{d['severity']} damage (conf {d.get('confidence', 0):.2f}) "
                        f"crosses the road corridor outside the endpoint wards",
                    }
                )
                break
    return out


async def load_graph(
    event_id: str, params: dict | None = None
) -> tuple[RoadGraph, list[dict], list[dict], dict]:
    """Fetch an event's network from MongoDB and build its RoadGraph."""
    from core import db
    from core.standards import baseline

    params = params or baseline()
    database = db.get_db()
    event = await database[db.EVENTS].find_one({"_id": db.oid(event_id)})
    if event is None:
        raise ValueError(f"no such event: {event_id}")
    nodes = [n async for n in database[db.NODES].find({"event_id": event_id})]
    edges = [e async for e in database[db.EDGES].find({"event_id": event_id})]
    graph = build_graph(
        nodes,
        edges,
        degraded_factor=params["degraded_road_factor"].value,
        air_verified=bool(event.get("air_transport_verified")),
        air_cost_factor=params["air_cost_factor"].value,
    )
    return graph, nodes, edges, event


async def apply_cv_blockages(event_id: str) -> list[dict]:
    """Mark roads cut by detected damage as blocked (source="cv")."""
    from core import db

    database = db.get_db()
    nodes = [n async for n in database[db.NODES].find({"event_id": event_id})]
    edges = [e async for e in database[db.EDGES].find({"event_id": event_id})]
    zones = [z async for z in database[db.ZONES].find({"event_id": event_id})]
    detections = [d async for d in database[db.DETECTIONS].find({"event_id": event_id})]
    flagged = flag_blocked_edges(nodes, edges, zones, detections)
    for f in flagged:
        await database[db.EDGES].update_one(
            {"_id": db.oid(f["edge_id"])},
            {"$set": {"status": "blocked", "source": "cv", "blocked_reason": f["reason"]}},
        )
    return flagged
