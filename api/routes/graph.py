"""Road network / routing graph endpoints (S5).

Scaffolded in milestone 1; PyTorch Geometric routing lands in milestone 3.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core import db
from core.schemas import GraphEdge, GraphEdgeCreate, GraphNode, GraphNodeCreate

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
                },
            }
        )

    return {
        "nodes": {"type": "FeatureCollection", "features": node_features},
        "edges": {"type": "FeatureCollection", "features": edge_features},
    }


@router.patch("/edges/{edge_id}", response_model=GraphEdge)
async def set_edge_status(edge_id: str, status: str) -> GraphEdge:
    """Mark a road open / degraded / blocked.

    The CV pipeline writes here too; routing in milestone 3 treats `blocked`
    as a hard removal, never a penalty.
    """
    if status not in ("open", "degraded", "blocked"):
        raise HTTPException(422, f"invalid edge status: {status}")
    doc = await db.get_db()[db.EDGES].find_one_and_update(
        {"_id": db.oid(edge_id)}, {"$set": {"status": status}}, return_document=True
    )
    if doc is None:
        raise HTTPException(404, f"no such edge: {edge_id}")
    return GraphEdge.model_validate(db.doc_out(doc))
