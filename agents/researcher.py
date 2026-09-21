"""S7: web research for historical disaster-response precedents.

Two Claude calls, split on purpose:

  1. search  - Claude with the server-side web_search tool investigates how
               comparable past responses (Gorkha 2015, Jajarkot 2023,
               Sindhupalchok 2014, ...) sized and routed relief. We record every
               URL the search tool actually returned.
  2. extract - a structured-output call turns that report into Precedent
               records, each naming one of the retrieved URLs.

Any extracted finding whose URL was not in the retrieved set is dropped. The
model therefore cannot cite a source it did not actually fetch: citation
integrity is enforced in code, not requested in a prompt.

Results are cached in MongoDB (research_cache, TTL index) keyed on the research
context, so repeated allocation runs for one event cost one search.

With no API key, or on any API failure, research returns status="unavailable"
and the allocator proceeds on the cited baseline - loudly, never silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from core.config import get_settings
from core.schemas import Precedent, utcnow

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_PAUSE_RESUMES = 4

SEARCH_SYSTEM = """You research historical disaster-response logistics for a relief \
allocation system used by district emergency operations in Nepal. Your findings set \
planning numbers that decide where doctors, food and water go, so only report what a \
source actually states. Prefer after-action reports, government PDNAs, WHO / IFRC / \
OCHA / WFP / Sphere documents and peer-reviewed studies over news. For every figure, \
name the source and the event it comes from. If you cannot find evidence for a \
parameter, say so rather than estimating."""


class ExtractedFinding(BaseModel):
    claim: str = Field(description="One sentence stating what the source establishes")
    parameter: str | None = Field(
        default=None, description="Planning parameter this informs, from the list given, or null"
    )
    value: str | None = Field(default=None, description="The number the source gives, with units")
    source_index: int = Field(description="Index into the numbered source list")
    event: str | None = Field(default=None, description="Disaster the evidence comes from")


class ExtractedFindings(BaseModel):
    findings: list[ExtractedFinding]


@dataclass
class ResearchResult:
    status: str  # "ok" | "cached" | "unavailable"
    precedents: list[Precedent] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    report: str = ""
    error: str | None = None
    # parameter name for each precedent (parallel to precedents), may be None
    parameters: list[str | None] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "queries": self.queries,
            "report": self.report,
            "precedents": [
                {
                    "id": i,
                    "parameter": self.parameters[i] if i < len(self.parameters) else None,
                    **p.model_dump(),
                }
                for i, p in enumerate(self.precedents)
            ],
        }


def context_key(context: dict) -> str:
    """Cache key: what the research depends on, not incidental detail."""
    key = {
        k: context.get(k)
        for k in ("disaster_type", "region", "severity_counts", "unreachable_zones", "parameters")
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def search_prompt(context: dict) -> str:
    params = "\n".join(
        f"- {p['name']} (currently {p['value']} {p['unit']}, {p['kind']}): {p['description']}"
        for p in context["parameters"]
    )
    return f"""Current event: {context["event_name"]} - {context["disaster_type"]} in \
{context["region"]}.
Situation: {context["situation"]}

Find evidence from comparable past disasters - especially Nepal (2015 Gorkha earthquake, \
2023 Jajarkot earthquake, 2014 Sindhupalchok/Jure landslide, monsoon floods) and other \
mountainous or road-cut responses - that bears on these planning parameters:

{params}

Also look for lessons on prioritising cut-off communities and on air versus road delivery \
when roads are blocked. Report each finding with its figure, the event, and the source."""


def _client():
    import anthropic

    settings = get_settings()
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)


async def _search(client, context: dict) -> tuple[str, dict[str, dict], list[str]]:
    """Run the web-search turn. Returns (report text, url -> source meta, queries)."""
    settings = get_settings()
    messages = [{"role": "user", "content": search_prompt(context)}]
    sources: dict[str, dict] = {}
    queries: list[str] = []
    text_parts: list[str] = []

    for _ in range(MAX_PAUSE_RESUMES + 1):
        response = await client.beta.messages.create(
            model=settings.llm_model,
            max_tokens=16000,
            system=SEARCH_SYSTEM,
            thinking={"type": "adaptive"},
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("research request declined by the model")
        for block in response.content:
            if block.type == "server_tool_use" and getattr(block, "input", None):
                q = block.input.get("query") if isinstance(block.input, dict) else None
                if q:
                    queries.append(q)
            elif block.type == "web_search_tool_result":
                content = block.content
                if isinstance(content, list):  # an error result is a single object
                    for r in content:
                        if getattr(r, "url", None):
                            sources.setdefault(
                                r.url, {"title": r.title or r.url, "published": r.page_age}
                            )
            elif block.type == "text":
                text_parts.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url:
                        sources.setdefault(
                            url, {"title": getattr(c, "title", None) or url, "published": None}
                        )
        if response.stop_reason != "pause_turn":
            break
        # Resume the server-side loop: resend with the paused assistant turn.
        messages = [*messages, {"role": "assistant", "content": response.content}]
    return "\n".join(text_parts).strip(), sources, queries


async def _extract(
    client, context: dict, report: str, sources: dict[str, dict]
) -> list[ExtractedFinding]:
    settings = get_settings()
    urls = list(sources)
    numbered = "\n".join(f"[{i}] {sources[u]['title']} - {u}" for i, u in enumerate(urls))
    names = ", ".join(p["name"] for p in context["parameters"])
    response = await client.beta.messages.parse(
        model=settings.llm_model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research report:\n\n{report}\n\nNumbered sources:\n{numbered}\n\n"
                    f"Extract every distinct, sourced finding from the report. Each must cite "
                    f"exactly one source by its index. Set `parameter` to one of [{names}] when "
                    f"the finding gives a figure for it, else null. Omit anything the report "
                    f"does not attribute to a listed source."
                ),
            }
        ],
        output_format=ExtractedFindings,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return []
    return response.parsed_output.findings


def to_precedents(
    findings: list[ExtractedFinding], sources: dict[str, dict], allowed_parameters: set[str]
) -> tuple[list[Precedent], list[str | None], int]:
    """Keep only findings that point at a retrieved source. Returns (kept, params, dropped)."""
    urls = list(sources)
    kept, params, dropped = [], [], 0
    for f in findings:
        if not 0 <= f.source_index < len(urls):
            dropped += 1
            continue
        url = urls[f.source_index]
        if not url.startswith(("http://", "https://")):
            dropped += 1
            continue
        claim = f.claim if not f.event or f.event in f.claim else f"{f.claim} ({f.event})"
        kept.append(
            Precedent(
                claim=claim,
                metric=f.parameter,
                value=f.value,
                source_title=sources[url]["title"],
                source_url=url,
                published=sources[url].get("published"),
            )
        )
        params.append(f.parameter if f.parameter in allowed_parameters else None)
    return kept, params, dropped


async def research(context: dict, *, use_cache: bool = True, client=None) -> ResearchResult:
    """Find cited precedents for the planning parameters in `context`."""
    from core import db

    settings = get_settings()
    key = context_key(context)
    cache = db.get_db()[db.RESEARCH_CACHE]
    if use_cache:
        hit = await cache.find_one({"query_hash": key})
        if hit:
            result = ResearchResult(
                status="cached",
                precedents=[Precedent.model_validate(p) for p in hit["precedents"]],
                parameters=hit.get("parameters", []),
                queries=hit.get("queries", []),
                report=hit.get("report", ""),
            )
            return result

    if client is None and not settings.anthropic_api_key:
        return ResearchResult(
            status="unavailable", error="ANTHROPIC_API_KEY is not set - web research skipped"
        )

    import anthropic

    try:
        client = client or _client()
        report, sources, queries = await _search(client, context)
        if not sources:
            return ResearchResult(
                status="unavailable",
                report=report,
                queries=queries,
                error="web search returned no sources",
            )
        findings = await _extract(client, context, report, sources)
    except (anthropic.APIConnectionError, anthropic.APIStatusError, RuntimeError) as exc:
        log.warning("research failed: %s", exc)
        return ResearchResult(status="unavailable", error=f"{type(exc).__name__}: {exc}")

    allowed = {p["name"] for p in context["parameters"]}
    precedents, params, dropped = to_precedents(findings, sources, allowed)
    if dropped:
        log.warning("dropped %d findings citing sources that were never retrieved", dropped)
    result = ResearchResult(
        status="ok", precedents=precedents, parameters=params, queries=queries, report=report
    )
    await cache.update_one(
        {"query_hash": key},
        {
            "$set": {
                "query_hash": key,
                "created_at": utcnow(),
                "queries": queries,
                "report": report,
                "precedents": [p.model_dump() for p in precedents],
                "parameters": params,
            }
        },
        upsert=True,
    )
    return result
