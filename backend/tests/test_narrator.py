from __future__ import annotations

import json

import httpx
import pytest

from app.schemas import ResourceUnit
from app.services import narrator

_ANALYSIS = {
    "summary": {"total_buildings": 4, "destroyed_pct": 25.0, "major_pct": 25.0, "minor_pct": 50.0},
    "zones": [
        {
            "rank": 1,
            "priority_score": 87.5,
            "building_counts": {"none": 0, "minor": 1, "major": 1, "destroyed": 2},
        }
    ],
    "mask_base64": "A" * 50_000,
    "mask_path": "/srv/outputs/deadbeef/damage_mask.png",
}

_GOOD_ANSWER = {
    "brief": "Deploy to zone 1.",
    "risk_analysis": [
        {
            "zone_rank": 1,
            "risk_level": "critical",
            "risk_score": 88.0,
            "primary_hazards": ["Structural collapse"],
            "rationale": "Two destroyed buildings.",
        }
    ],
    "resource_allocation": [
        {
            "zone_rank": 1,
            "resource": "Search & rescue teams",
            "quantity": 3,
            "unit": "teams",
            "urgency": "immediate",
            "justification": "Highest priority score.",
        }
    ],
    "reserve_notes": "Holding two teams in reserve.",
}


def _gemini_response(url: str, answer: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": json.dumps(answer)}]}}]},
        request=httpx.Request("POST", url),
    )


def _install_client(monkeypatch, handler) -> None:
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            return handler(url, json, headers)

    monkeypatch.setattr(narrator.httpx, "AsyncClient", lambda **kw: _FakeClient())


@pytest.fixture
def captured_payload(monkeypatch):
    """Intercept the Gemini request and hand back its JSON body."""
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")
    captured: dict = {}

    def handler(url, body, headers):
        captured.update(body)
        return _gemini_response(url, _GOOD_ANSWER)

    _install_client(monkeypatch, handler)
    return captured


def _prompt(captured: dict) -> str:
    return captured["contents"][0]["parts"][0]["text"]


@pytest.mark.anyio
async def test_mask_base64_never_reaches_the_llm(captured_payload):
    result = await narrator.generate_brief(dict(_ANALYSIS))

    assert result.source == "gemini"
    prompt = _prompt(captured_payload)
    assert "mask_base64" not in prompt
    assert "mask_path" not in prompt
    assert "AAAA" not in prompt
    # The zone data the brief is supposed to narrate must survive the filter.
    assert "priority_score" in prompt


@pytest.mark.anyio
async def test_risk_and_allocation_are_parsed(captured_payload):
    result = await narrator.generate_brief(dict(_ANALYSIS))

    assert [r.zone_rank for r in result.risk_analysis] == [1]
    assert result.risk_analysis[0].risk_level == "critical"
    assert [a.quantity for a in result.resource_allocation] == [3]
    assert result.reserve_notes == "Holding two teams in reserve."


@pytest.mark.anyio
async def test_roster_is_sent_to_the_llm(captured_payload):
    await narrator.generate_brief(
        dict(_ANALYSIS), resources=[ResourceUnit(name="Boats", quantity=2, unit="boats")]
    )
    sent = json.loads(_prompt(captured_payload))
    assert sent["available_resources"] == [
        {"name": "Boats", "quantity": 2, "unit": "boats"}
    ]


@pytest.mark.anyio
async def test_allocation_cannot_exceed_the_roster(monkeypatch):
    """A model that hands out units we do not have would send teams nowhere."""
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")
    greedy = dict(_GOOD_ANSWER)
    greedy["resource_allocation"] = [
        {"zone_rank": 1, "resource": "Boats", "quantity": 99, "urgency": "immediate"}
    ]
    _install_client(monkeypatch, lambda url, body, headers: _gemini_response(url, greedy))

    result = await narrator.generate_brief(
        dict(_ANALYSIS), resources=[ResourceUnit(name="Boats", quantity=2, unit="boats")]
    )
    assert [a.quantity for a in result.resource_allocation] == [2]


@pytest.mark.anyio
async def test_unknown_zone_ranks_are_dropped(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")
    hallucinated = dict(_GOOD_ANSWER)
    hallucinated["risk_analysis"] = [
        {"zone_rank": 9, "risk_level": "critical", "risk_score": 50.0, "rationale": "x"}
    ]
    _install_client(
        monkeypatch, lambda url, body, headers: _gemini_response(url, hallucinated)
    )

    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert result.risk_analysis == []


@pytest.mark.anyio
async def test_generate_brief_does_not_mutate_callers_analysis(captured_payload):
    analysis = dict(_ANALYSIS)
    await narrator.generate_brief(analysis)
    assert "mask_base64" in analysis


@pytest.mark.anyio
async def test_context_is_truncated(captured_payload):
    await narrator.generate_brief(dict(_ANALYSIS), context="x" * 5000)

    sent = json.loads(_prompt(captured_payload))
    assert len(sent["context"]) == narrator.MAX_CONTEXT_CHARS


@pytest.mark.anyio
async def test_a_spent_key_falls_through_to_the_next(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "spent-key, good-key")
    seen: list[str] = []

    def handler(url, body, headers):
        key = headers["x-goog-api-key"]
        seen.append(key)
        if key == "spent-key":
            return httpx.Response(429, json={}, request=httpx.Request("POST", url))
        return _gemini_response(url, _GOOD_ANSWER)

    _install_client(monkeypatch, handler)

    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert seen == ["spent-key", "good-key"]
    assert result.source == "gemini"


@pytest.mark.anyio
async def test_every_key_failing_falls_back_to_stub(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "a,b")
    _install_client(
        monkeypatch,
        lambda url, body, headers: httpx.Response(
            429, json={}, request=httpx.Request("POST", url)
        ),
    )

    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert result.source == "gemini-fallback"
    assert "Zone #1" in result.brief
    # The stub still has to produce a usable plan, not an empty panel.
    assert result.resource_allocation


@pytest.mark.anyio
async def test_network_failure_falls_back_to_stub(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            raise httpx.ConnectError("gemini unreachable")

    monkeypatch.setattr(narrator.httpx, "AsyncClient", lambda **kw: _FailingClient())
    # A dead network is retried across every sweep of every fallback model, so
    # the real backoff would put ~18s of sleeping in the suite for a path that
    # is about the fallback, not the pacing.
    monkeypatch.setattr(narrator, "_SWEEP_BACKOFF_SECONDS", 0.0)

    result = await narrator.generate_brief(dict(_ANALYSIS))

    assert result.source == "gemini-fallback"
    assert "Zone #1" in result.brief


@pytest.mark.anyio
async def test_empty_candidate_falls_back_to_stub(monkeypatch):
    """Thinking models can burn the whole budget and return no text part."""
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")
    _install_client(
        monkeypatch,
        lambda url, body, headers: httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": ""}]}}]},
            request=httpx.Request("POST", url),
        ),
    )

    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert result.source == "gemini-fallback"


@pytest.mark.anyio
async def test_missing_api_key_serves_stub(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "")
    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert result.source == "stub"


_SIXTEEN_ZONES = {
    "summary": {"total_buildings": 160, "destroyed_pct": 10.0, "major_pct": 20.0, "minor_pct": 30.0},
    "zones": [
        {
            "rank": rank,
            # Scores cluster tightly, as they do on real imagery, so every
            # zone's exact share of 8 teams is well under 1.
            "priority_score": 57.0 - rank * 0.3,
            "building_counts": {"none": 5, "minor": 3, "major": 1, "destroyed": 1},
        }
        for rank in range(1, 17)
    ],
}


@pytest.mark.anyio
async def test_stub_allocation_serves_the_worst_zones_first(monkeypatch):
    """Two failures this guards against, in order of discovery.

    First: flooring each zone's exact share dumped every unit on the last zone.
    With 16 zones and 8 teams each exact share is ~0.5, so flooring gave every
    zone 0 and the remainder-to-last rule handed all 8 teams to zone 16 — the
    least damaged one. Largest-remainder apportionment fixed that.

    Then: spreading the roster over all 16 zones bought each of them a single
    team, on a dashboard that only ever draws five. The plan now covers the
    top MAX_PLAN_ZONES zones, so the worst zone gets a force worth dispatching.
    """
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "")
    roster = [ResourceUnit(name="Search & rescue teams", quantity=8, unit="teams")]

    result = await narrator.generate_brief(dict(_SIXTEEN_ZONES), resources=roster)

    teams = {a.zone_rank: a.quantity for a in result.resource_allocation}
    assert sum(teams.values()) == 8, "the whole roster must be committed"
    assert set(teams) == set(
        range(1, narrator.MAX_PLAN_ZONES + 1)
    ), f"only the zones on screen get a plan: {teams}"
    assert teams[1] == max(teams.values()), f"rank 1 is never out-served: {teams}"
    assert {r.zone_rank for r in result.risk_analysis} == set(teams), (
        "the risk read must cover the same zones as the plan"
    )


@pytest.mark.anyio
async def test_fallback_brief_does_not_blame_a_missing_key(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "a-real-key")
    _install_client(
        monkeypatch,
        lambda url, body, headers: httpx.Response(
            503, json={}, request=httpx.Request("POST", url)
        ),
    )
    monkeypatch.setattr(narrator, "_TOTAL_BUDGET_SECONDS", 0.0)

    result = await narrator.generate_brief(dict(_ANALYSIS))
    assert result.source == "gemini-fallback"
    assert "GEMINI_API_KEYS" not in result.brief


@pytest.mark.anyio
async def test_scarce_units_reach_the_worst_hit_zone_first(monkeypatch):
    """Rank order, not the model's output order, decides who gets trimmed.

    The panel promises "worst-hit zones first". A model listing zone 3 before
    zone 1 used to spend the whole roster on zone 3 and leave rank 1 with
    nothing, because the budget was consumed in the order the JSON arrived.
    """
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")

    analysis = dict(_ANALYSIS)
    analysis["zones"] = [
        {
            "rank": rank,
            "priority_score": score,
            "building_counts": {"none": 0, "minor": 1, "major": 1, "destroyed": 2},
        }
        for rank, score in ((1, 87.5), (2, 60.0), (3, 20.0))
    ]

    # Deliberately worst-last, and asking for four of the two boats that exist.
    out_of_order = dict(_GOOD_ANSWER)
    out_of_order["resource_allocation"] = [
        {"zone_rank": 3, "resource": "Boats", "quantity": 2, "urgency": "scheduled"},
        {"zone_rank": 1, "resource": "Boats", "quantity": 2, "urgency": "immediate"},
    ]
    _install_client(
        monkeypatch, lambda url, body, headers: _gemini_response(url, out_of_order)
    )

    result = await narrator.generate_brief(
        analysis, resources=[ResourceUnit(name="Boats", quantity=2, unit="boats")]
    )

    # Rank 1 takes both boats; rank 3 is trimmed away entirely, and the plan
    # reads in dispatch order.
    assert [(a.zone_rank, a.quantity) for a in result.resource_allocation] == [(1, 2)]


@pytest.mark.anyio
async def test_allocation_is_returned_in_rank_order(monkeypatch):
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "test-key")

    analysis = dict(_ANALYSIS)
    analysis["zones"] = [
        {
            "rank": rank,
            "priority_score": 50.0,
            "building_counts": {"none": 0, "minor": 1, "major": 1, "destroyed": 2},
        }
        for rank in (1, 2, 3)
    ]

    shuffled = dict(_GOOD_ANSWER)
    shuffled["resource_allocation"] = [
        {"zone_rank": 2, "resource": "Boats", "quantity": 1, "urgency": "urgent"},
        {"zone_rank": 3, "resource": "Boats", "quantity": 1, "urgency": "scheduled"},
        {"zone_rank": 1, "resource": "Boats", "quantity": 1, "urgency": "immediate"},
    ]
    _install_client(
        monkeypatch, lambda url, body, headers: _gemini_response(url, shuffled)
    )

    result = await narrator.generate_brief(
        analysis, resources=[ResourceUnit(name="Boats", quantity=9, unit="boats")]
    )
    assert [a.zone_rank for a in result.resource_allocation] == [1, 2, 3]


@pytest.mark.anyio
async def test_a_model_out_of_daily_quota_falls_through_to_the_next_model(monkeypatch):
    """Free-tier quota is per model, not per account.

    gemini-3.6-flash allows 20 requests a day; once that is spent every key
    answers 429 on that model while the same keys still work on another. Before
    this fallback the dashboard spent the rest of the day on the deterministic
    stub plan, with an AI brief that said Gemini did not answer.
    """
    monkeypatch.setattr(narrator.settings, "gemini_api_keys", "k1,k2")
    monkeypatch.setattr(narrator.settings, "gemini_model", "primary-model")
    monkeypatch.setattr(narrator.settings, "gemini_fallback_models", "backup-model")
    tried: list[str] = []

    def handler(url, body, headers):
        model = url.rsplit("/", 1)[-1].split(":")[0]
        tried.append(model)
        if model == "primary-model":
            return httpx.Response(429, json={}, request=httpx.Request("POST", url))
        return _gemini_response(url, _GOOD_ANSWER)

    _install_client(monkeypatch, handler)

    result = await narrator.generate_brief(dict(_ANALYSIS))

    assert result.source == "gemini"
    # Both keys are spent on the primary before the backup is tried at all.
    assert tried == ["primary-model", "primary-model", "backup-model"]
