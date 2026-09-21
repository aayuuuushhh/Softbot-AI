"""The schemas carry two of CLAUDE.md's non-negotiables structurally:
a dispatch cannot exist without a citation, and a citation cannot exist
without a real URL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.schemas import (
    SEVERITY_ORDER,
    Citation,
    DispatchItem,
    DispatchOrder,
    Point,
    ResourceKind,
    Severity,
)


def _citation() -> Citation:
    return Citation(
        claim="Isolated flood zones needed 1 doctor per 450 affected people.",
        source_title="Nepal Earthquake 2015 Health Cluster Report",
        source_url="https://reliefweb.int/report/nepal/example",
    )


def _items() -> list[DispatchItem]:
    return [DispatchItem(kind=ResourceKind.MEDICAL, sku="trauma_kit", quantity=20)]


def test_dispatch_requires_at_least_one_citation():
    with pytest.raises(ValidationError, match="citations"):
        DispatchOrder(
            zone_id="a" * 24,
            from_node_id="b" * 24,
            items=_items(),
            rationale="because",
            citations=[],
        )


def test_dispatch_accepts_a_cited_order():
    order = DispatchOrder(
        zone_id="a" * 24,
        from_node_id="b" * 24,
        items=_items(),
        rationale="Zone is cut off and has the highest destroyed-structure count.",
        citations=[_citation()],
    )
    assert order.citations[0].source_url.startswith("https://")
    assert order.transport_mode == "road"


def test_dispatch_requires_at_least_one_item():
    with pytest.raises(ValidationError, match="items"):
        DispatchOrder(
            zone_id="a" * 24,
            from_node_id="b" * 24,
            items=[],
            rationale="empty",
            citations=[_citation()],
        )


@pytest.mark.parametrize("bad", ["reliefweb.int/report", "ftp://x.y", "", "see report"])
def test_citation_rejects_non_http_sources(bad):
    with pytest.raises(ValidationError):
        Citation(claim="c", source_title="t", source_url=bad)


def test_point_requires_lon_lat_pair():
    assert Point(coordinates=[85.3, 28.1]).coordinates == [85.3, 28.1]
    with pytest.raises(ValidationError):
        Point(coordinates=[85.3])


def test_severity_order_is_monotonic():
    ordered = [Severity.NONE, Severity.MINOR, Severity.MAJOR, Severity.DESTROYED]
    values = [SEVERITY_ORDER[s.value] for s in ordered]
    assert values == sorted(values) == [0, 1, 2, 3]


def test_precedent_converts_to_citation():
    from core.schemas import Precedent

    p = Precedent(
        claim="Food distribution ran through ward-level hubs.",
        metric="rations_per_household",
        value="1 sack / 2 weeks",
        source_title="Sindhupalchok 2014 Landslide Response Review",
        source_url="https://example.org/review",
    )
    c = p.as_citation()
    assert c.source_url == p.source_url
    assert c.claim == p.claim
