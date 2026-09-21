"""Agent layer (S7/S8) with a fake Anthropic client - no network, no API spend.

What is being pinned down is the part that must hold whatever the model says:
citations only ever point at URLs the search tool returned, and parameter
adjustments are accepted only when cited and in bounds.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from agents import researcher
from agents.allocator import Adjustment, PolicyProposal, apply_policy
from core.needs import compute_zone_needs
from core.schemas import Precedent
from core.standards import baseline

GOOD_URL = "https://reliefweb.int/report/nepal/gorkha-2015-health-response"
OTHER_URL = "https://www.who.int/emergencies/emt"


def search_response(stop="end_turn"):
    return NS(
        stop_reason=stop,
        content=[
            NS(
                type="server_tool_use",
                input={"query": "Gorkha 2015 emergency medical teams patients per day"},
            ),
            NS(
                type="web_search_tool_result",
                content=[
                    NS(
                        url=GOOD_URL,
                        title="Nepal: Gorkha earthquake health response",
                        page_age="2015-06-01",
                    ),
                    NS(url=OTHER_URL, title="WHO EMT initiative", page_age=None),
                ],
            ),
            NS(
                type="text",
                text="Teams in Gorkha saw about 30 trauma patients per doctor per day.",
                citations=[NS(url=GOOD_URL, title="Nepal: Gorkha earthquake health response")],
            ),
        ],
    )


class FakeMessages:
    def __init__(self, create_responses, parsed):
        self._create = list(create_responses)
        self._parsed = parsed
        self.create_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._create.pop(0)

    async def parse(self, **kwargs):
        return NS(stop_reason="end_turn", parsed_output=self._parsed)


class FakeClient:
    def __init__(self, create_responses, parsed):
        self.beta = NS(messages=FakeMessages(create_responses, parsed))


CONTEXT = {
    "event_name": "pytest flood",
    "disaster_type": "flood",
    "region": "Rasuwa",
    "severity_counts": {"destroyed": 2},
    "unreachable_zones": ["Haku"],
    "situation": "2 wards destroyed",
    "parameters": [
        {
            "name": p.name,
            "value": p.value,
            "unit": p.unit,
            "kind": p.kind,
            "description": p.description,
            "bounds": list(p.bounds),
        }
        for p in baseline().values()
    ],
}


def test_findings_citing_unretrieved_sources_are_dropped():
    sources = {GOOD_URL: {"title": "t", "published": None}}
    findings = [
        researcher.ExtractedFinding(
            claim="real", parameter="patients_per_doctor_day", value="30", source_index=0
        ),
        researcher.ExtractedFinding(claim="invented", parameter=None, value=None, source_index=7),
    ]
    kept, params, dropped = researcher.to_precedents(findings, sources, {"patients_per_doctor_day"})
    assert [p.claim for p in kept] == ["real"]
    assert params == ["patients_per_doctor_day"]
    assert dropped == 1


async def test_search_collects_only_urls_the_tool_returned():
    client = FakeClient([search_response()], None)
    report, sources, queries = await researcher._search(client, CONTEXT)
    assert set(sources) == {GOOD_URL, OTHER_URL}
    assert queries == ["Gorkha 2015 emergency medical teams patients per day"]
    assert "30 trauma patients" in report
    call = client.beta.messages.create_calls[0]
    assert call["tools"][0]["type"] == "web_search_20260209"
    assert call["fallbacks"] == "default"


async def test_search_resumes_a_paused_server_tool_turn():
    client = FakeClient([search_response("pause_turn"), search_response()], None)
    await researcher._search(client, CONTEXT)
    calls = client.beta.messages.create_calls
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "assistant"  # paused turn sent back


def _research(n=2):
    precedents = [
        Precedent(claim=f"finding {i}", source_title="Gorkha report", source_url=GOOD_URL)
        for i in range(n)
    ]
    return researcher.ResearchResult(
        status="ok", precedents=precedents, parameters=["patients_per_doctor_day"] * n
    )


def test_cited_in_bounds_adjustment_is_accepted_and_carries_its_evidence():
    proposal = PolicyProposal(
        summary="s",
        adjustments=[
            Adjustment(
                parameter="patients_per_doctor_day", value=30, precedent_ids=[0], rationale="Gorkha"
            ),
        ],
    )
    params, accepted, rejected = apply_policy(baseline(), proposal, _research())
    p = params["patients_per_doctor_day"]
    assert p.value == 30 and p.adjusted_by_agent
    assert p.citations[0].source_url == GOOD_URL
    assert accepted and not rejected


@pytest.mark.parametrize(
    "adj, reason",
    [
        (
            Adjustment(parameter="made_up", value=1, precedent_ids=[0], rationale=""),
            "unknown parameter",
        ),
        (
            Adjustment(parameter="injury_rate", value=0.2, precedent_ids=[], rationale=""),
            "cites no retrieved",
        ),
        (
            Adjustment(parameter="injury_rate", value=0.2, precedent_ids=[0, 9], rationale=""),
            "do not exist",
        ),
        (
            Adjustment(parameter="injury_rate", value=0.99, precedent_ids=[0], rationale=""),
            "outside bounds",
        ),
    ],
)
def test_bad_adjustments_are_rejected_with_a_reason(adj, reason):
    params, accepted, rejected = apply_policy(
        baseline(), PolicyProposal(summary="", adjustments=[adj]), _research()
    )
    assert not accepted
    assert reason in rejected[0]["reason"]
    assert params["injury_rate"].value == baseline()["injury_rate"].value


def test_accepted_adjustment_flows_through_to_needs_and_derivations():
    proposal = PolicyProposal(
        summary="",
        adjustments=[
            Adjustment(
                parameter="patients_per_doctor_day", value=25, precedent_ids=[1], rationale="x"
            )
        ],
    )
    params, _, _ = apply_policy(baseline(), proposal, _research())
    z = {
        "severity": "destroyed",
        "population": 4000,
        "damage_extent": 0.5,
        "buildings_destroyed": 0,
    }
    needs = compute_zone_needs(z, params)
    assert needs["sku_demand"]["doctor"] == 8  # ceil(200 injured / 25)
    assert GOOD_URL in needs["derivations"]["doctors"]["sources"]
    assert any("agent-adjusted" in a for a in needs["derivations"]["doctors"]["assumptions"])


async def test_research_without_a_key_degrades_loudly(monkeypatch):
    from core.config import get_settings
    from tests.conftest import MONGO_UP

    if not MONGO_UP:
        pytest.skip("research consults the MongoDB cache first")
    monkeypatch.setattr(get_settings(), "anthropic_api_key", "")
    result = await researcher.research({**CONTEXT, "region": "pytest-nokey"}, use_cache=False)
    assert result.status == "unavailable"
    assert "ANTHROPIC_API_KEY" in result.error
