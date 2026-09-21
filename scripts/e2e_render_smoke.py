#!/usr/bin/env python3
"""Full API E2E against the deployed DisasterIQ backend using D:\\AMD\\data\\demo."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://disasteriq-backend.onrender.com"
ORIGIN = "https://disasteriq.vercel.app"
DEMO = Path(r"D:\AMD\data\demo\images")

results: list[tuple[str, str, str]] = []


def header(headers: dict[str, str], name: str) -> str:
    lower = name.lower()
    for key, value in headers.items():
        if key.lower() == lower:
            return value
    return ""


def record(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


def request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 90,
) -> tuple[int, dict[str, str], bytes]:
    hdrs = {"Origin": ORIGIN, **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, {k: v for k, v in resp.headers.items()}, body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, {k: v for k, v in exc.headers.items()}, body


def encode_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----DisasterIQE2E7a3f9"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(value.encode())
    for name, (filename, content, ctype) in files.items():
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
        )
        lines.append(f"Content-Type: {ctype}".encode())
        lines.append(b"")
        lines.append(content)
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    # 1. Health + CORS
    t0 = time.perf_counter()
    status, headers, body = request("GET", f"{API}/health", timeout=90)
    ms = (time.perf_counter() - t0) * 1000
    aca = header(headers, "Access-Control-Allow-Origin")
    try:
        health = json.loads(body.decode())
    except json.JSONDecodeError:
        health = {}
    record(
        "health+cors",
        status == 200 and aca == ORIGIN and health.get("status") == "ok",
        f"status={status} aca={aca!r} body={body[:120]!r} ({ms:.0f}ms)",
    )

    # OPTIONS
    status, headers, _ = request(
        "OPTIONS",
        f"{API}/health",
        headers={
            "Access-Control-Request-Method": "GET",
        },
        timeout=30,
    )
    record(
        "cors_options",
        status in (200, 204) and header(headers, "Access-Control-Allow-Origin") == ORIGIN,
        f"status={status} aca={header(headers, 'Access-Control-Allow-Origin')!r}",
    )

    # 2. Demo pairs
    status, _, body = request("GET", f"{API}/demo/pairs", timeout=60)
    try:
        pairs = json.loads(body.decode()) if status == 200 else []
    except json.JSONDecodeError:
        pairs = []
    record(
        "demo_pairs",
        status == 200 and isinstance(pairs, list) and len(pairs) >= 1,
        f"status={status} count={len(pairs) if isinstance(pairs, list) else 'n/a'}",
    )

    demo_id = "river-valley-flood_01"
    if isinstance(pairs, list) and pairs:
        ids = {p["id"] for p in pairs}
        if demo_id not in ids:
            demo_id = pairs[0]["id"]

    # 3. Analyze demo (form field)
    form_body = f"demo_pair_id={demo_id}".encode()
    t0 = time.perf_counter()
    status, _, body = request(
        "POST",
        f"{API}/analyze",
        data=form_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=300,
    )
    ms = (time.perf_counter() - t0) * 1000
    try:
        analysis = json.loads(body.decode()) if status == 200 else {}
    except json.JSONDecodeError:
        analysis = {}
    zones = analysis.get("zones") or []
    record(
        "analyze_demo",
        status == 200
        and bool(zones)
        and bool(analysis.get("mask_base64"))
        and analysis.get("inference_mode")
        in ("stub-groundtruth", "stub-heuristic", "docker", "pytorch"),
        f"status={status} zones={len(zones)} mode={analysis.get('inference_mode')} "
        f"geo_available={analysis.get('geo_available')} geo_mode={analysis.get('geo_mode')} "
        f"({ms:.0f}ms)",
    )

    # 4. Brief
    brief: dict = {}
    if analysis:
        payload = json.dumps({"analysis": analysis, "context": "E2E test brief"}).encode()
        t0 = time.perf_counter()
        status, _, body = request(
            "POST",
            f"{API}/brief",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        ms = (time.perf_counter() - t0) * 1000
        try:
            brief = json.loads(body.decode()) if status == 200 else {}
        except json.JSONDecodeError:
            brief = {}
        record(
            "brief",
            status == 200
            and bool(brief.get("brief"))
            and brief.get("source") in ("stub", "gemini", "gemini-fallback"),
            f"status={status} source={brief.get('source')} len={len(brief.get('brief') or '')} ({ms:.0f}ms)",
        )
    else:
        record("brief", False, "skipped — no analysis")

    # 5. PDF
    if analysis and brief.get("brief"):
        payload = json.dumps({"analysis": analysis, "brief": brief["brief"]}).encode()
        t0 = time.perf_counter()
        status, headers, body = request(
            "POST",
            f"{API}/report/pdf",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        ms = (time.perf_counter() - t0) * 1000
        ctype = headers.get("Content-Type", "")
        record(
            "report_pdf",
            status == 200 and ctype.startswith("application/pdf") and len(body) > 500,
            f"status={status} ctype={ctype} bytes={len(body)} ({ms:.0f}ms)",
        )
    else:
        record("report_pdf", False, "skipped — no analysis/brief")

    # 6. Upload
    pre = DEMO / "river-valley-flood_01_pre_disaster.png"
    post = DEMO / "river-valley-flood_01_post_disaster.png"
    if pre.is_file() and post.is_file():
        form, ctype = encode_multipart(
            {},
            {
                "pre_image": ("pre.png", pre.read_bytes(), "image/png"),
                "post_image": ("post.png", post.read_bytes(), "image/png"),
            },
        )
        t0 = time.perf_counter()
        status, _, body = request(
            "POST",
            f"{API}/analyze",
            data=form,
            headers={"Content-Type": ctype},
            timeout=300,
        )
        ms = (time.perf_counter() - t0) * 1000
        try:
            up = json.loads(body.decode()) if status == 200 else {}
        except json.JSONDecodeError:
            up = {}
        up_zones = up.get("zones") or []
        has_centroids = bool(up_zones) and all(
            isinstance(z.get("centroid_lat"), (int, float))
            and isinstance(z.get("centroid_lng"), (int, float))
            for z in up_zones
        )
        record(
            "analyze_upload",
            status == 200 and bool(up_zones) and bool(up.get("mask_base64")),
            f"status={status} zones={len(up_zones)} geo_mode={up.get('geo_mode')} "
            f"geo_available={up.get('geo_available')} centroids={has_centroids} "
            f"image_size={up.get('image_size')} ({ms:.0f}ms)",
        )
        if up:
            payload = json.dumps({"analysis": up, "context": None}).encode()
            status, _, body = request(
                "POST",
                f"{API}/brief",
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            try:
                b2 = json.loads(body.decode()) if status == 200 else {}
            except json.JSONDecodeError:
                b2 = {}
            record(
                "brief_upload",
                status == 200 and bool(b2.get("brief")),
                f"status={status} source={b2.get('source')}",
            )
        else:
            record("brief_upload", False, "skipped")
    else:
        record("analyze_upload", False, f"missing files under {DEMO}")
        record("brief_upload", False, "skipped")

    # 7. Demo image
    img_name = "river-valley-flood_01_post_disaster.png"
    status, headers, body = request("GET", f"{API}/demo/images/{img_name}", timeout=60)
    ctype = headers.get("Content-Type", "")
    record(
        "demo_image",
        status == 200 and len(body) > 1000 and "image" in ctype,
        f"status={status} ctype={ctype} bytes={len(body)}",
    )

    # 8. Bad MIME
    form, ctype = encode_multipart(
        {},
        {
            "pre_image": ("pre.pdf", b"%PDF-1.4", "application/pdf"),
            "post_image": ("post.png", b"\x89PNG\r\n\x1a\nxxxx", "image/png"),
        },
    )
    status, _, body = request(
        "POST",
        f"{API}/analyze",
        data=form,
        headers={"Content-Type": ctype},
        timeout=60,
    )
    try:
        detail = json.loads(body.decode()).get("detail", body[:160])
    except json.JSONDecodeError:
        detail = body[:160]
    record("reject_bad_mime", status == 400, f"status={status} detail={detail!r}")

    # Second demo
    form_body = b"demo_pair_id=river-valley-flood_02"
    t0 = time.perf_counter()
    status, _, body = request(
        "POST",
        f"{API}/analyze",
        data=form_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=300,
    )
    ms = (time.perf_counter() - t0) * 1000
    try:
        eq = json.loads(body.decode()) if status == 200 else {}
    except json.JSONDecodeError:
        eq = {}
    record(
        "analyze_demo_earthquake",
        status == 200 and bool(eq.get("zones")),
        f"status={status} zones={len(eq.get('zones') or [])} "
        f"geo_available={eq.get('geo_available')} geo_mode={eq.get('geo_mode')} ({ms:.0f}ms)",
    )

    print("\n=== SUMMARY ===")
    fails = [x for x in results if x[1] == "FAIL"]
    for name, st, detail in results:
        print(f"{st:4}  {name}: {detail}")
    print(f"\n{len(results) - len(fails)}/{len(results)} passed, {len(fails)} failed")

    # Residual issue notes for the operator
    print("\n=== RESIDUAL / OBSERVATIONS ===")
    if analysis.get("geo_available") is False and analysis.get("geo_mode") != "image":
        print(
            "- HIGH: Deployed backend may not yet include geo_mode=image "
            "(Render may still be on an older commit)."
        )
    if analysis and analysis.get("geo_available") is True:
        print("- Demo flood pair returned WGS84 geo — map should use OSM.")
    print(
        "- UI: hard-refresh Vercel after deploy 6c0f488 so connect harden is active."
    )
    print(
        "- Render free tier cold starts can still show Connecting… for up to ~90s."
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
