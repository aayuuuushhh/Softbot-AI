"""Gemini analyst — reads the ranked JSON, judges risk, and allocates resources.

The model never re-ranks: scoring.py owns the numbers. Gemini's job is to turn
those counts into a risk read per zone and a division of a finite responder
roster across zones, both of which are judgement calls the deterministic
pipeline cannot make on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.config import settings
from app.schemas import (
    DEFAULT_RESOURCES,
    BriefResponse,
    ResourceAssignment,
    ResourceUnit,
    ZoneRisk,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 2000

# How many zones the plan covers. scoring.py can rank a dozen or more zones in a
# dense scene, but the dashboard draws the top five on the damage overlay — so a
# risk panel listing sixteen zones, and a roster split sixteen ways, described
# ground the coordinator could not see and handed every zone a single team. The
# plan and the picture have to be about the same places.
MAX_PLAN_ZONES = 5


def _plan_zones(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """The zones the brief, risk read and allocation are allowed to cover."""
    zones = [z for z in (analysis.get("zones") or []) if isinstance(z, dict)]
    zones.sort(key=lambda z: int(z.get("rank", 0) or 0))
    return zones[:MAX_PLAN_ZONES]

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Never ship these to the LLM: the overlay is a full-size base64 PNG that would
# dwarf the analysis JSON in the prompt, and the mask path is a server-local
# detail the model has no use for.
_PROMPT_EXCLUDED_KEYS = frozenset({"mask_base64", "mask_path"})

# A key that is out of quota or revoked should cost us one attempt and then get
# out of the way. A 400/500 is about the request, not the key, so retrying the
# same body under a different key would just repeat the failure.
_KEY_EXHAUSTED_STATUSES = frozenset({401, 403, 429})

# Google returns 503 on free-tier model overload often enough that giving up on
# the first sweep would drop the dashboard to the stub plan several times a day.
# These are about the service, not the key, so the whole roster is worth a
# second sweep after a short pause.
_TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
_MAX_SWEEPS = 3
_SWEEP_BACKOFF_SECONDS = 2.0

# The whole retry effort has to finish inside the client's brief timeout (90s in
# frontend/src/lib/api.ts). Retrying past that point is worse than not retrying:
# the browser has already given up, so the caller sees an error instead of the
# stub plan the fallback would have produced. Budget stops the sweeps early and
# leaves room to build and send that fallback.
_TOTAL_BUDGET_SECONDS = 55.0
_PER_REQUEST_TIMEOUT_SECONDS = 25.0

RISK_LEVELS = ("critical", "high", "moderate", "low")
URGENCIES = ("immediate", "urgent", "scheduled")

SYSTEM_PROMPT = """You are a disaster response analyst supporting an emergency
operations centre. You receive a JSON object holding deterministic per-zone
building damage counts and priority scores derived from pre/post satellite
imagery, plus the responder roster currently available.

Produce three things:

1. `brief` — a situation brief in plain language for emergency coordinators,
   under 250 words. Lead with the highest-priority zones.
2. `risk_analysis` — one entry per zone in the input (the input holds only the
   highest-priority zones), judging the risk that zone
   poses. Weigh destroyed and major-damage building counts most heavily, then
   consider likely trapped occupants, structural collapse risk, and access.
3. `resource_allocation` — divide the supplied roster across the zones so that
   life-saving capacity reaches the worst-hit zones first.

Hard rules:
- Never change the ranks, priority scores, or damage counts you are given, and
  never invent damage data that is not in the input.
- Every `zone_rank` you emit must be one that appears in the input.
- Allocate no more of a resource than the roster holds. Totals per resource must
  not exceed the supplied quantity. Leaving a deliberate reserve is allowed and
  should be explained in `reserve_notes`.
- `risk_level` must be one of: critical, high, moderate, low.
- `urgency` must be one of: immediate, urgent, scheduled.
- Keep every justification to one sentence.
"""

# Gemini honours a declared response schema, which removes the usual "model
# wrapped its JSON in prose" failure mode and lets us parse without repair.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "brief": {"type": "string"},
        "risk_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zone_rank": {"type": "integer"},
                    "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
                    "risk_score": {"type": "number"},
                    "primary_hazards": {"type": "array", "items": {"type": "string"}},
                    "population_at_risk": {"type": "string"},
                    "access_notes": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                # Everything is required: Gemini leaves optional fields empty,
                # which would render as blank cells in the risk panel.
                "required": [
                    "zone_rank",
                    "risk_level",
                    "risk_score",
                    "primary_hazards",
                    "population_at_risk",
                    "access_notes",
                    "rationale",
                ],
            },
        },
        "resource_allocation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zone_rank": {"type": "integer"},
                    "resource": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit": {"type": "string"},
                    "urgency": {"type": "string", "enum": list(URGENCIES)},
                    "justification": {"type": "string"},
                },
                "required": [
                    "zone_rank",
                    "resource",
                    "quantity",
                    "unit",
                    "urgency",
                    "justification",
                ],
            },
        },
        "reserve_notes": {"type": "string"},
    },
    "required": ["brief", "risk_analysis", "resource_allocation"],
}


def _resource_list(resources: list[ResourceUnit] | None) -> list[ResourceUnit]:
    return list(resources) if resources else list(DEFAULT_RESOURCES)


def _risk_level_for(destroyed: int, major: int, total: int) -> tuple[str, float]:
    """Deterministic fallback risk read, used when no LLM answer is available."""
    if total <= 0:
        return "low", 0.0
    severe = (destroyed * 2 + major) / (total * 2)
    score = round(min(100.0, severe * 100), 1)
    if score >= 60:
        return "critical", score
    if score >= 35:
        return "high", score
    if score >= 15:
        return "moderate", score
    return "low", score


def _stub_analysis(
    analysis: dict[str, Any], resources: list[ResourceUnit]
) -> tuple[list[ZoneRisk], list[ResourceAssignment], str]:
    """Proportional allocation: each zone's share tracks its priority score.

    This is arithmetic, not judgement — it exists so the dashboard still has a
    plan to show when no key is configured or every key fails.
    """
    zones = _plan_zones(analysis)
    risks: list[ZoneRisk] = []
    for z in zones:
        bc = z.get("building_counts", {}) or {}
        destroyed = int(bc.get("destroyed", 0) or 0)
        major = int(bc.get("major", 0) or 0)
        total = sum(int(bc.get(k, 0) or 0) for k in ("none", "minor", "major", "destroyed"))
        level, score = _risk_level_for(destroyed, major, total)
        risks.append(
            ZoneRisk(
                zone_rank=int(z.get("rank", 0)),
                risk_level=level,
                risk_score=score,
                primary_hazards=["Structural collapse"] if destroyed else ["Structural damage"],
                population_at_risk=f"{total} buildings assessed in zone",
                access_notes="Access unverified — confirm routes before dispatch.",
                rationale=(
                    f"{destroyed} destroyed and {major} major-damage buildings "
                    f"of {total} assessed."
                ),
            )
        )

    weights = [max(z.get("priority_score", 0) or 0, 0) for z in zones]
    total_weight = sum(weights)
    allocation: list[ResourceAssignment] = []

    for resource in resources:
        if total_weight <= 0 or resource.quantity <= 0:
            continue

        # Largest-remainder apportionment. Flooring each zone's exact share and
        # handing the shortfall to the last zone put eight of eight teams into
        # zone 16 whenever the exact shares were all below 1 — the worst-hit
        # zones got nothing. Here the leftover units go to the zones with the
        # largest fractional claim, so rank 1 is served first.
        exact = [resource.quantity * w / total_weight for w in weights]
        shares = [int(e) for e in exact]
        leftover = resource.quantity - sum(shares)
        if leftover > 0:
            order = sorted(
                range(len(zones)),
                key=lambda i: (exact[i] - shares[i], weights[i]),
                reverse=True,
            )
            for i in order[:leftover]:
                shares[i] += 1

        for idx, zone in enumerate(zones):
            if shares[idx] <= 0:
                continue
            allocation.append(
                ResourceAssignment(
                    zone_rank=int(zone.get("rank", 0)),
                    resource=resource.name,
                    quantity=shares[idx],
                    unit=resource.unit,
                    urgency="immediate" if idx == 0 else "urgent" if idx < 3 else "scheduled",
                    justification=(
                        f"{round(100 * weights[idx] / total_weight)}% of the priority "
                        f"weight across the top {len(zones)} zones."
                    ),
                )
            )

    return risks, allocation, "Proportional split — no reserve held back."


def _stub_brief(analysis: dict[str, Any], context: str | None, source: str) -> str:
    summary = analysis.get("summary", {})
    zones = analysis.get("zones", [])[:3]
    reason = (
        "Gemini did not answer, so this is the deterministic fallback"
        if source == "gemini-fallback"
        else "stub — set GEMINI_API_KEYS for live analysis"
    )
    lines = [
        f"SITUATION BRIEF ({reason})",
        "",
    ]
    if context:
        lines.append(f"Context: {context}")
        lines.append("")
    lines.append(
        f"Overall: {summary.get('total_buildings', 0)} buildings assessed. "
        f"Destroyed: {summary.get('destroyed_pct', 0)}%, "
        f"Major: {summary.get('major_pct', 0)}%, "
        f"Minor: {summary.get('minor_pct', 0)}%."
    )
    lines.append("")
    lines.append("Priority zones (pre-ranked by ML):")
    for z in zones:
        bc = z.get("building_counts", {})
        lines.append(
            f"  Zone #{z.get('rank')}: score {z.get('priority_score')} — "
            f"destroyed={bc.get('destroyed', 0)}, major={bc.get('major', 0)}, "
            f"minor={bc.get('minor', 0)}, undamaged={bc.get('none', 0)} buildings"
        )
    lines.append("")
    lines.append(
        "Recommendation: Deploy assessment teams to highest-scored zones first "
        "while verifying access routes and secondary hazards."
    )
    return "\n".join(lines)


def _stub_response(
    analysis: dict[str, Any],
    context: str | None,
    resources: list[ResourceUnit],
    source: str,
) -> BriefResponse:
    risks, allocation, reserve = _stub_analysis(analysis, resources)
    return BriefResponse(
        brief=_stub_brief(analysis, context, source),
        source=source,
        risk_analysis=risks,
        resource_allocation=allocation,
        reserve_notes=reserve,
    )


def _rank_key(item: dict[str, Any]) -> int:
    """Sort key for raw model output, tolerant of the junk the parsers drop.

    An unparseable zone_rank sorts last rather than raising: the loops below
    are what decide to discard it, and sorting must not pre-empt that.
    """
    try:
        return int(item.get("zone_rank"))
    except (TypeError, ValueError):
        return 1 << 30


def _clamp_allocation(
    raw: list[dict[str, Any]],
    resources: list[ResourceUnit],
    valid_ranks: set[int],
) -> list[ResourceAssignment]:
    """Drop unknown zones and hold the plan to the roster we actually have.

    The schema constrains shape, not arithmetic: a model can still hand out
    twelve of the eight teams that exist. Over-allocation is worse than a short
    plan here, because a coordinator would be dispatching units that do not
    exist, so anything past a resource's quantity is trimmed.

    Rank order decides who gets trimmed. The model emits assignments in
    whatever order it pleases, so consuming the budget in that order let an
    arbitrary mid-table zone spend the last team and left rank 1 short. Working
    worst-hit zone first means the trimming falls on the zones that can best
    afford to wait, and the returned plan reads in the order it is dispatched.
    """
    raw = sorted(raw, key=_rank_key)
    budget = {r.name.strip().lower(): r.quantity for r in resources}
    units = {r.name.strip().lower(): r.unit for r in resources}
    spent: dict[str, int] = {}
    out: list[ResourceAssignment] = []

    for item in raw:
        try:
            rank = int(item.get("zone_rank"))
            quantity = int(item.get("quantity", 0))
        except (TypeError, ValueError):
            continue
        if rank not in valid_ranks or quantity <= 0:
            continue

        name = str(item.get("resource", "")).strip()
        if not name:
            continue
        key = name.lower()

        if key in budget:
            remaining = budget[key] - spent.get(key, 0)
            if remaining <= 0:
                continue
            quantity = min(quantity, remaining)
            spent[key] = spent.get(key, 0) + quantity

        urgency = str(item.get("urgency", "")).strip().lower()
        out.append(
            ResourceAssignment(
                zone_rank=rank,
                resource=name,
                quantity=quantity,
                unit=str(item.get("unit") or units.get(key, "")),
                urgency=urgency if urgency in URGENCIES else "scheduled",
                justification=str(item.get("justification", "")).strip(),
            )
        )
    return out


def _parse_risks(raw: list[dict[str, Any]], valid_ranks: set[int]) -> list[ZoneRisk]:
    out: list[ZoneRisk] = []
    for item in raw:
        try:
            rank = int(item.get("zone_rank"))
            score = float(item.get("risk_score", 0))
        except (TypeError, ValueError):
            continue
        if rank not in valid_ranks:
            continue
        level = str(item.get("risk_level", "")).strip().lower()
        hazards = item.get("primary_hazards") or []
        out.append(
            ZoneRisk(
                zone_rank=rank,
                risk_level=level if level in RISK_LEVELS else "moderate",
                risk_score=min(max(score, 0.0), 100.0),
                primary_hazards=[str(h) for h in hazards][:6],
                population_at_risk=str(item.get("population_at_risk", "")).strip(),
                access_notes=str(item.get("access_notes", "")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    out.sort(key=lambda r: r.zone_rank)
    return out


async def _call_gemini(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Try each configured model, and within it each key. Parsed JSON, or None.

    A key rejected for quota is dropped for the rest of that model's attempt; a
    transient service error only ends that sweep, and the keys are tried again.
    When every key is exhausted on a model, the next model gets the same keys:
    the free tier counts its daily quota per model, so "all keys 429" usually
    means this model is spent for today, not that the account is.
    """
    deadline = time.monotonic() + _TOTAL_BUDGET_SECONDS

    async with httpx.AsyncClient(timeout=_PER_REQUEST_TIMEOUT_SECONDS) as client:
        for model in settings.gemini_model_list:
            if time.monotonic() >= deadline:
                break
            result = await _call_gemini_model(client, model, payload, deadline)
            if result is not None:
                return result
    return None


async def _call_gemini_model(
    client: httpx.AsyncClient,
    model: str,
    payload: dict[str, Any],
    deadline: float,
) -> dict[str, Any] | None:
    """One model, every key, with the sweep/backoff policy above."""
    url = f"{_API_ROOT}/{model}:generateContent"
    keys = settings.gemini_key_list

    live_keys = list(keys)
    for sweep in range(_MAX_SWEEPS):
        if not live_keys or time.monotonic() >= deadline:
            break
        saw_transient = False
        for index, key in enumerate(list(live_keys), start=1):
            if time.monotonic() >= deadline:
                logger.warning("Gemini retry budget spent; serving the stub plan")
                return None
            try:
                resp = await client.post(
                    url, json=payload, headers={"x-goog-api-key": key}
                )
                if resp.status_code in _KEY_EXHAUSTED_STATUSES:
                    logger.warning(
                        "Gemini key %d/%d rejected on %s (HTTP %d); "
                        "trying the next key",
                        index,
                        len(keys),
                        model,
                        resp.status_code,
                    )
                    live_keys.remove(key)
                    continue
                if resp.status_code in _TRANSIENT_STATUSES:
                    logger.warning(
                        "Gemini transient error on %s (HTTP %d) on key %d/%d",
                        model,
                        resp.status_code,
                        index,
                        len(keys),
                    )
                    saw_transient = True
                    continue
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                # A thinking model can emit reasoning parts alongside the
                # answer; the JSON body is the last part carrying text.
                text = next(
                    (p["text"] for p in reversed(parts) if p.get("text")), ""
                ).strip()
                if not text:
                    logger.warning("Gemini returned an empty candidate")
                    continue
                return json.loads(text)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                logger.warning(
                    "Gemini call failed on %s with key %d: %s", model, index, exc
                )
                saw_transient = True
                continue

        if not saw_transient:
            break
        if sweep < _MAX_SWEEPS - 1:
            backoff = _SWEEP_BACKOFF_SECONDS * (sweep + 1)
            if time.monotonic() + backoff >= deadline:
                break
            await asyncio.sleep(backoff)
    return None


async def generate_brief(
    analysis: dict[str, Any],
    context: str | None = None,
    resources: list[ResourceUnit] | None = None,
) -> BriefResponse:
    roster = _resource_list(resources)

    if not settings.gemini_key_list:
        return _stub_response(analysis, context, roster, "stub")

    if context is not None and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    plan_zones = _plan_zones(analysis)
    prompt_analysis = {k: v for k, v in analysis.items() if k not in _PROMPT_EXCLUDED_KEYS}
    # The model plans for the zones on screen, not every zone scoring.py ranked.
    prompt_analysis["zones"] = plan_zones
    user_content = json.dumps(
        {
            "analysis": prompt_analysis,
            "context": context,
            "available_resources": [r.model_dump() for r in roster],
        },
        indent=2,
    )

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }

    data = await _call_gemini(payload)
    if data is None:
        return _stub_response(analysis, context, roster, "gemini-fallback")

    brief_text = str(data.get("brief", "")).strip()
    if not brief_text:
        return _stub_response(analysis, context, roster, "gemini-fallback")

    valid_ranks = {int(z["rank"]) for z in plan_zones if "rank" in z}
    return BriefResponse(
        brief=brief_text,
        source="gemini",
        risk_analysis=_parse_risks(data.get("risk_analysis") or [], valid_ranks),
        resource_allocation=_clamp_allocation(
            data.get("resource_allocation") or [], roster, valid_ranks
        ),
        reserve_notes=str(data.get("reserve_notes", "")).strip(),
    )
