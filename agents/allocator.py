"""S7 + S8: the allocation agent.

    research (web)  ->  policy (LLM)  ->  needs + routing  ->  optimise  ->  verify  ->  persist
    agents/researcher   this module      core/needs,           core/allocation

Division of labour - the part that matters for "Traceable AI":

* The LLM never writes a dispatch order. It proposes *parameter adjustments*
  (e.g. patients per doctor per day, degraded-road cost), each one citing
  precedents the researcher retrieved. Code checks every adjustment: known
  parameter, value within hard bounds, cites >= 1 valid precedent. Anything else
  is rejected and recorded.
* The deterministic optimiser turns the accepted parameter set into orders, and
  core.allocation.check_invariants re-verifies them (stock, demand, routes).
* If the agent's plan fails verification, the baseline plan is used instead and
  the run is flagged `degraded`. If research is unavailable, the run is flagged
  `mode="baseline"` with the reason. Neither happens silently.

Every run is written to `agent_runs` in full - queries, report, precedents,
proposal, accepted/rejected adjustments, parameters, invariant checks - so any
order can be traced back to the evidence behind it.
"""

from __future__ import annotations

import logging
from collections import Counter

from pydantic import BaseModel, Field

from agents import researcher as researcher_mod
from core import db, inventory
from core.allocation import ZoneDemand, allocate, check_invariants, to_orders
from core.config import get_settings
from core.needs import compute_zone_needs
from core.schemas import DispatchOrder, DispatchStatus, StageStatus, utcnow
from core.spatial.graph_network import build_graph, reachability
from core.standards import Parameter, baseline

log = logging.getLogger(__name__)

POLICY_SYSTEM = """You set planning parameters for disaster relief allocation in Nepal. \
A deterministic optimiser turns your parameters into dispatch orders for doctors, food, \
water and medical kits, so a wrong number sends real aid to the wrong place.

Adjust a parameter only when the numbered precedents give direct, quantitative evidence \
for a better value in a situation comparable to this one. Cite the precedent ids you rely \
on. Stay inside each parameter's bounds. Where evidence is thin, conflicting, or from a \
very different context, leave the parameter unchanged - an unchanged, cited baseline is \
better than an adjustment you cannot support. Explain any unit conversion in the rationale."""


class Adjustment(BaseModel):
    parameter: str
    value: float
    precedent_ids: list[int] = Field(description="Ids of the precedents this relies on")
    rationale: str


class PolicyProposal(BaseModel):
    adjustments: list[Adjustment]
    summary: str = Field(description="Two or three sentences on what changed and why")


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


def build_context(
    event: dict,
    zones: list[dict],
    reach: dict,
    stock: dict,
    params: dict[str, Parameter],
    baseline_needs: dict[str, dict],
    edges: list[dict],
) -> dict:
    severity_counts = Counter(z.get("severity", "none") for z in zones)
    node_by_zone = {str(z["_id"]): z.get("_node_id") for z in zones}
    unreachable = sorted(
        z["name"]
        for z in zones
        if reach.get(node_by_zone[str(z["_id"])], {}).get("status") == "unreachable"
        and z.get("severity", "none") != "none"
    )
    affected = sum(n["affected_people"] for n in baseline_needs.values())
    demand: Counter = Counter()
    for n in baseline_needs.values():
        demand.update(n["sku_demand"])
    supply: Counter = Counter()
    for (_node, sku), q in stock.items():
        supply[sku] += q
    shortfalls = {
        s: {"need": demand[s], "available": supply.get(s, 0)}
        for s in sorted(demand)
        if demand[s] > supply.get(s, 0)
    }
    blocked = sum(1 for e in edges if e.get("status") == "blocked")
    situation = (
        f"{severity_counts.get('destroyed', 0)} wards destroyed, {severity_counts.get('major', 0)} major, "
        f"{severity_counts.get('minor', 0)} minor damage; ~{affected:,} people affected; "
        f"{blocked} road segments blocked; cut-off damaged wards: {', '.join(unreachable) or 'none'}; "
        f"air transport {'verified' if event.get('air_transport_verified') else 'NOT verified'}; "
        f"shortfalls vs. baseline need: {shortfalls or 'none'}."
    )
    return {
        "event_name": event.get("name", ""),
        "disaster_type": event.get("disaster_type", ""),
        "region": event.get("region", ""),
        "severity_counts": dict(sorted(severity_counts.items())),
        "unreachable_zones": unreachable,
        "situation": situation,
        "parameters": [
            {
                "name": p.name,
                "value": p.value,
                "unit": p.unit,
                "kind": p.kind,
                "description": p.description,
                "bounds": list(p.bounds),
            }
            for p in params.values()
        ],
    }


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


async def propose_policy(
    client, context: dict, research: researcher_mod.ResearchResult
) -> PolicyProposal | None:
    settings = get_settings()
    precedents = "\n".join(
        f"[{i}] ({research.parameters[i] if i < len(research.parameters) and research.parameters[i] else 'general'}) "
        f"{p.claim}" + (f" Value: {p.value}." if p.value else "") + f" Source: {p.source_title}"
        for i, p in enumerate(research.precedents)
    )
    params = "\n".join(
        f"- {p['name']} = {p['value']} {p['unit']} ({p['kind']}), bounds {p['bounds']}: {p['description']}"
        for p in context["parameters"]
    )
    response = await client.beta.messages.parse(
        model=settings.llm_model,
        max_tokens=16000,
        system=POLICY_SYSTEM,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Event: {context['event_name']} ({context['disaster_type']}, {context['region']})\n"
                    f"Situation: {context['situation']}\n\nParameters:\n{params}\n\n"
                    f"Precedents:\n{precedents}\n\nPropose adjustments."
                ),
            }
        ],
        output_format=PolicyProposal,
        betas=[researcher_mod.FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output


def apply_policy(
    params: dict[str, Parameter],
    proposal: PolicyProposal | None,
    research: researcher_mod.ResearchResult,
) -> tuple[dict[str, Parameter], list[dict], list[dict]]:
    """Validate and apply adjustments. Returns (params, accepted, rejected)."""
    accepted, rejected = [], []
    if proposal is None:
        return params, accepted, rejected
    seen: set[str] = set()
    for adj in proposal.adjustments:
        record = adj.model_dump()
        p = params.get(adj.parameter)
        valid_ids = [i for i in adj.precedent_ids if 0 <= i < len(research.precedents)]
        if p is None:
            rejected.append({**record, "reason": "unknown parameter"})
        elif adj.parameter in seen:
            rejected.append({**record, "reason": "parameter adjusted twice"})
        elif not valid_ids:
            rejected.append({**record, "reason": "cites no retrieved precedent"})
        elif len(valid_ids) != len(adj.precedent_ids):
            rejected.append({**record, "reason": "cites precedent ids that do not exist"})
        elif not p.bounds[0] <= adj.value <= p.bounds[1]:
            rejected.append({**record, "reason": f"value outside bounds {list(p.bounds)}"})
        else:
            seen.add(adj.parameter)
            cites = [research.precedents[i].as_citation() for i in valid_ids]
            params[adj.parameter] = p.model_copy(
                update={
                    "value": float(adj.value),
                    "adjusted_by_agent": True,
                    "citations": cites + p.citations,
                    "note": f"baseline {p.value} -> {adj.value}: {adj.rationale}",
                }
            )
            accepted.append({**record, "baseline": p.value})
    return params, accepted, rejected


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


async def _load(event_id: str):
    database = db.get_db()
    event = await database[db.EVENTS].find_one({"_id": db.oid(event_id)})
    if event is None:
        raise ValueError(f"no such event: {event_id}")
    zones = [z async for z in database[db.ZONES].find({"event_id": event_id})]
    nodes = [n async for n in database[db.NODES].find({"event_id": event_id})]
    edges = [e async for e in database[db.EDGES].find({"event_id": event_id})]
    zone_node = {str(n["zone_id"]): str(n["_id"]) for n in nodes if n.get("zone_id")}
    for z in zones:
        z["_node_id"] = zone_node.get(str(z["_id"]))
    return event, zones, nodes, edges


def plan_once(
    event: dict,
    zones: list[dict],
    nodes: list[dict],
    edges: list[dict],
    stock: dict,
    params: dict[str, Parameter],
    sku_kind: dict[str, str],
) -> dict:
    """Needs -> graph -> allocation -> invariant check, for one parameter set."""
    graph = build_graph(
        nodes,
        edges,
        degraded_factor=params["degraded_road_factor"].value,
        air_verified=bool(event.get("air_transport_verified")),
        air_cost_factor=params["air_cost_factor"].value,
    )
    needs = {str(z["_id"]): compute_zone_needs(z, params) for z in zones}
    demands = [
        ZoneDemand(
            str(z["_id"]),
            z["_node_id"],
            z["name"],
            z.get("severity", "none"),
            needs[str(z["_id"])]["affected_people"],
            needs[str(z["_id"])]["sku_demand"],
        )
        for z in zones
        if z.get("_node_id")
    ]
    result = allocate(demands, stock, graph, params)
    problems = check_invariants(
        result, demands, stock, nodes, edges, bool(event.get("air_transport_verified"))
    )
    orders = to_orders(result, demands, params, sku_kind)
    for o in orders:  # schema-level guarantees: >= 1 item, >= 1 valid citation
        DispatchOrder.model_validate(o)
    return {"needs": needs, "result": result, "problems": problems, "orders": orders}


async def run_allocation(event_id: str, *, client=None, research_fn=None) -> dict:
    """Plan and persist dispatch orders for an event. Returns the plan."""
    research_fn = research_fn or researcher_mod.research
    database = db.get_db()
    await database[db.EVENTS].update_one(
        {"_id": db.oid(event_id)}, {"$set": {"stages.allocation": StageStatus.RUNNING.value}}
    )
    try:
        event, zones, nodes, edges = await _load(event_id)
        stock = await inventory.available(event_id)
        sku_kind = await inventory.sku_kinds(event_id)

        base = baseline()
        base_graph = build_graph(
            nodes,
            edges,
            degraded_factor=base["degraded_road_factor"].value,
            air_verified=bool(event.get("air_transport_verified")),
        )
        reach = reachability(base_graph)
        baseline_needs = {str(z["_id"]): compute_zone_needs(z, base) for z in zones}
        context = build_context(event, zones, reach, stock, base, baseline_needs, edges)

        notes: list[str] = []
        research = await research_fn(context)
        params, accepted, rejected, proposal = baseline(), [], [], None
        if research.status in ("ok", "cached") and research.precedents:
            settings = get_settings()
            if client is not None or settings.anthropic_api_key:
                import anthropic

                try:
                    proposal = await propose_policy(
                        client or researcher_mod._client(), context, research
                    )
                except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
                    notes.append(
                        f"policy step failed ({type(exc).__name__}); baseline parameters used"
                    )
                params, accepted, rejected = apply_policy(params, proposal, research)
        elif research.status == "unavailable":
            notes.append(
                f"web research unavailable: {research.error}. Allocation uses the cited "
                f"baseline parameters only."
            )
        else:
            notes.append("web research found no usable precedents; baseline parameters used")

        plan = plan_once(event, zones, nodes, edges, stock, params, sku_kind)
        degraded, violations = False, []
        if plan["problems"]:
            degraded, violations = True, plan["problems"]
            notes.append("agent-parameter plan failed verification; deterministic baseline used")
            plan = plan_once(event, zones, nodes, edges, stock, baseline(), sku_kind)
            if plan["problems"]:
                raise RuntimeError(f"baseline plan violates invariants: {plan['problems']}")
            params, accepted = baseline(), []

        mode = "agent" if accepted else "baseline"
        result = plan["result"]
        summary = (
            f"{len(plan['orders'])} dispatch orders to "
            f"{len({o['zone_id'] for o in plan['orders']})} zones; "
            f"{len(result.unmet)} unmet needs"
            + (
                f" ({sum(1 for u in result.unmet if 'air' in u.reason)} need air access)"
                if result.unmet
                else ""
            )
            + f". Parameters: {len(accepted)} adapted from web research, "
            f"{len(rejected)} proposals rejected."
        )
        if proposal is not None and proposal.summary:
            summary += f" Agent: {proposal.summary}"

        run_doc = {
            "event_id": event_id,
            "created_at": utcnow(),
            "mode": mode,
            "degraded": degraded,
            "violations": violations,
            "notes": notes,
            "summary": summary,
            "context": context,
            "research": research.as_dict(),
            "proposal": proposal.model_dump() if proposal else None,
            "accepted": accepted,
            "rejected": rejected,
            "parameters": {k: v.model_dump() for k, v in params.items()},
            "zone_rank": result.zone_rank,
            "unmet": [u.__dict__ for u in result.unmet],
            "ignored_skus": result.ignored_skus,
        }
        run_id = str((await database[db.AGENT_RUNS].insert_one(run_doc)).inserted_id)

        # Replace the previous *proposed* plan; reserved / in-transit orders stand.
        await database[db.DISPATCHES].delete_many(
            {"event_id": event_id, "status": DispatchStatus.PROPOSED.value}
        )
        now = utcnow()
        if plan["orders"]:
            await database[db.DISPATCHES].insert_many(
                [
                    {
                        **o,
                        "event_id": event_id,
                        "run_id": run_id,
                        "status": DispatchStatus.PROPOSED.value,
                        "created_at": now,
                        "updated_at": now,
                    }
                    for o in plan["orders"]
                ]
            )
        for zone_id, needs in plan["needs"].items():
            await database[db.ZONES].update_one(
                {"_id": db.oid(zone_id)}, {"$set": {"needs": needs}}
            )
        await database[db.EVENTS].update_one(
            {"_id": db.oid(event_id)},
            {"$set": {"stages.allocation": StageStatus.DONE.value, "stages.message": None}},
        )

        return {
            "run_id": run_id,
            "mode": mode,
            "degraded": degraded,
            "violations": violations,
            "notes": notes,
            "summary": summary,
            "research_status": research.status,
            "precedents": len(research.precedents),
            "accepted": accepted,
            "rejected": rejected,
            "dispatches": plan["orders"],
            "unmet": run_doc["unmet"],
            "zone_rank": result.zone_rank,
        }
    except Exception as exc:
        await database[db.EVENTS].update_one(
            {"_id": db.oid(event_id)},
            {"$set": {"stages.allocation": StageStatus.FAILED.value, "stages.message": str(exc)}},
        )
        raise
