"""Road network / routing graph endpoints (S5)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from core import db
from core.schemas import GraphEdge, GraphEdgeCreate, GraphNode, GraphNodeCreate
from core.spatial.graph_network import apply_cv_blockages, best_route, load_graph, reachability

router = APIRouter(prefix="/api", tags=["graph"])


@router.post("/events/{event_id}/nodes", response_model=GraphNode, status_code=201)
async def create_node(event_id: str, payload: GraphNodeCreate) -> GraphNode:
    doc = payload.model_dump()
    doc["event_id"] = event_id
    result = await db.get_db()[db.NODES].insert_one(doc)
    doc["_id"] = result.inserted_id
    return GraphNode.model_validate(db.doc_out(doc))


@router.post("/events/{event_id}/edges", response_model=GraphEdge, status_code=201)
async def create_edge(event_id: str, payload: GraphEdgeCreate) -> GraphEdge:
    doc = payload.model_dump()
    doc["event_id"] = event_id
    result = await db.get_db()[db.EDGES].insert_one(doc)
    doc["_id"] = result.inserted_id
    return GraphEdge.model_validate(db.doc_out(doc))


@router.get("/events/{event_id}/graph")
async def get_graph(event_id: str) -> dict:
    """Nodes and edges as GeoJSON, ready for MapLibre."""
    database = db.get_db()
    nodes = [db.doc_out(d) async for d in database[db.NODES].find({"event_id": event_id})]
    edges = [db.doc_out(d) async for d in database[db.EDGES].find({"event_id": event_id})]
    by_id = {n["_id"]: n for n in nodes}

    node_features = [
        {
            "type": "Feature",
            "geometry": n["location"],
            "properties": {
                "id": n["_id"],
                "name": n["name"],
                "kind": n["kind"],
                "capacity": n.get("capacity", 0),
                "zone_id": n.get("zone_id"),
                "status": n.get("status", "open"),
            },
        }
        for n in nodes
    ]

    edge_features = []
    for e in edges:
        u, v = by_id.get(e["u"]), by_id.get(e["v"])
        if not u or not v:
            continue
        edge_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        u["location"]["coordinates"],
                        v["location"]["coordinates"],
                    ],
                },
                "properties": {
                    "id": e["_id"],
                    "status": e["status"],
                    "distance_km": e["distance_km"],
                    "source": e.get("source"),
                    "blocked_reason": e.get("blocked_reason"),
                    "from": u["name"],
                    "to": v["name"],
                },
            }
        )

    return {
        "nodes": {"type": "FeatureCollection", "features": node_features},
        "edges": {"type": "FeatureCollection", "features": edge_features},
    }


@router.get("/events/{event_id}/graph/summary")
async def graph_summary(event_id: str) -> dict:
    """The PyG view of the network: what routing actually sees."""
    graph, nodes, edges, event = await load_graph(event_id)
    data = graph.data
    return {
        "num_nodes": int(data.num_nodes),
        "passable_directed_edges": int(data.edge_index.shape[1]),
        "roads": len(edges),
        "blocked_roads": sum(1 for e in edges if e.get("status") == "blocked"),
        "degraded_roads": sum(1 for e in edges if e.get("status") == "degraded"),
        "blocked_nodes": sum(1 for n in nodes if n.get("status") == "blocked"),
        "supply_nodes": len(graph.supply_nodes),
        "air_transport_verified": bool(event.get("air_transport_verified")),
    }


@router.get("/events/{event_id}/reachability")
async def zone_reachability(event_id: str) -> list[dict]:
    """Per zone: reachable by road, by (verified) air, or not at all."""
    graph, _nodes, _edges, _event = await load_graph(event_id)
    out = []
    for node_id, r in reachability(graph).items():
        node = graph.nodes[node_id]
        best = r["best"]
        out.append(
            {
                "zone_id": str(node["zone_id"]) if node.get("zone_id") else None,
                "node_id": node_id,
                "name": node["name"],
                **r,
                "best": best
                and {**best, "path_names": [graph.nodes[p]["name"] for p in best["path"]]},
            }
        )
    order = {"unreachable": 0, "air": 1, "road": 2}
    return sorted(out, key=lambda r: (order[r["status"]], r["name"]))


@router.get("/events/{event_id}/route")
async def route(event_id: str, from_node: str = Query(...), to_node: str = Query(...)) -> dict:
    graph, _nodes, _edges, _event = await load_graph(event_id)
    if from_node not in graph.index or to_node not in graph.index:
        raise HTTPException(404, "unknown node id")
    r = best_route(graph, from_node, to_node)
    if r is None:
        raise HTTPException(409, "no road route, and air transport is not verified for this event")
    return {**r.as_dict(), "path_names": [graph.nodes[p]["name"] for p in r.path]}


@router.post("/events/{event_id}/graph/cv-blockages")
async def cv_blockages(event_id: str) -> dict:
    """Re-derive road blockages from the latest satellite detections."""
    return {"blocked": await apply_cv_blockages(event_id)}


@router.patch("/edges/{edge_id}", response_model=GraphEdge)
async def set_edge_status(edge_id: str, status: str) -> GraphEdge:
    """Mark a road open / degraded / blocked.

    An operator's call is final: the edge becomes `source: manual`, which the CV
    pipeline never overrides. Routing treats `blocked` as a hard removal.
    """
    if status not in ("open", "degraded", "blocked"):
        raise HTTPException(422, f"invalid edge status: {status}")
    doc = await db.get_db()[db.EDGES].find_one_and_update(
        {"_id": db.oid(edge_id)},
        {"$set": {"status": status, "source": "manual", "blocked_reason": None}},
        return_document=True,
    )
    if doc is None:
        raise HTTPException(404, f"no such edge: {edge_id}")
    return GraphEdge.model_validate(db.doc_out(doc))


@router.patch("/nodes/{node_id}", response_model=GraphNode)
async def set_node_status(node_id: str, status: str) -> GraphNode:
    """Mark a node open or blocked (e.g. a village cut off by a collapsed bridge)."""
    if status not in ("open", "blocked"):
        raise HTTPException(422, f"invalid node status: {status}")
    doc = await db.get_db()[db.NODES].find_one_and_update(
        {"_id": db.oid(node_id)}, {"$set": {"status": status}}, return_document=True
    )
    if doc is None:
        raise HTTPException(404, f"no such node: {node_id}")
    return GraphNode.model_validate(db.doc_out(doc))


@router.patch("/events/{event_id}/air-transport")
async def set_air_transport(event_id: str, verified: bool) -> dict:
    """Verify (or revoke) air transport - the only switch that lets routing
    reach zones with no open road."""
    res = await db.get_db()[db.EVENTS].update_one(
        {"_id": db.oid(event_id)}, {"$set": {"air_transport_verified": verified}}
    )
    if res.matched_count == 0:
        raise HTTPException(404, f"no such event: {event_id}")
    return {"event_id": event_id, "air_transport_verified": verified}
