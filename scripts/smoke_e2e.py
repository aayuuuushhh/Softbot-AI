#!/usr/bin/env python3
"""End-to-end smoke test of the perception pipeline, over real HTTP.

Starts its own uvicorn on a free port so it can never be fooled by a stale
server left over from an earlier run — a failure mode that looks exactly like
a code bug.

    python scripts/smoke_e2e.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "data" / "demo"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{f' - {detail}' if detail else ''}")
    if not condition:
        failures.append(label)
    return condition


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    if not (DEMO / "pre.tif").exists():
        print("missing demo data - run: python scripts/make_synthetic_pair.py --out data/demo")
        return 1
    truth = json.loads((DEMO / "truth.json").read_text())
    expected = {c["zone"]: c["severity"] for c in truth["clusters"]}

    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        client = httpx.Client(base_url=base, timeout=120.0)
        for _ in range(60):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            print("server never came up")
            return 1

        print("\n--- health ---")
        health = client.get("/health").json()
        check("server is ours", health.get("version") == "0.1.0", str(health.get("version")))
        check("mongodb connected", health["mongodb"] is True)
        check("cuda available", health["cuda"] is True, health.get("gpu") or "")

        print("\n--- demo event ---")
        events = client.get("/api/events").json()
        event = next((e for e in events if e["name"].startswith("Rasuwa")), None)
        if not check("seeded demo event exists", event is not None,
                     "run scripts/seed_demo.py --reset"):
            return 1
        eid = event["id"]

        print("\n--- S1 + S3: analyze pre/post pair ---")
        with (DEMO / "pre.tif").open("rb") as pre, (DEMO / "post.tif").open("rb") as post:
            resp = client.post(
                f"/api/events/{eid}/analyze",
                files={"pre": ("pre.tif", pre, "image/tiff"),
                       "post": ("post.tif", post, "image/tiff")},
            )
        check("analyze returns 200", resp.status_code == 200, resp.text[:160])
        if resp.status_code != 200:
            return 1
        result = resp.json()
        check("detections found", result["detections"] > 0, f"{result['detections']} regions")
        print(f"        backend={result['backend']} cloud={result['cloud_fraction']} "
              f"gsd={result['gsd_m2_per_px']} m2/px")

        print("\n--- zone severity vs ground truth ---")
        damage = client.get(f"/api/events/{eid}/damage").json()
        matched = 0
        for feature in damage["zones"]["features"]:
            p = feature["properties"]
            want = expected.get(p["name"])
            hit = p["severity"] == want
            matched += hit
            print(f"  {'ok  ' if hit else 'FAIL'}  {p['name']:14} "
                  f"truth={want:10} fused={p['severity']:10} "
                  f"score={p['damage_score']:.3f} by={p['decided_by']}")
        check(f"all {len(expected)} zones classified correctly",
              matched == len(expected), f"{matched}/{len(expected)}")

        print("\n--- S2: ground uploads ---")
        for photo in truth["ground_photos"][:4]:
            with (DEMO / photo["file"]).open("rb") as fh:
                r = client.post(
                    "/api/upload/ground",
                    data={"event_id": eid, "lat": photo["lat"], "lon": photo["lon"],
                          "reporter": "smoke_test"},
                    files={"image": (Path(photo["file"]).name, fh, "image/jpeg")},
                )
            if not check(f"upload {photo['zone']}", r.status_code == 201, r.text[:120]):
                continue
            body = r.json()
            damaged = body["severity"] in ("major", "destroyed")
            check(f"  {photo['zone']} classified {body['severity']}",
                  damaged == (photo["expected"] == "destroyed"),
                  f"expected {photo['expected']}, conf={body['confidence']:.2f}")
            check(f"  {photo['zone']} attached to a zone", body["zone_id"] is not None)

        print("\n--- artifacts + stage bookkeeping ---")
        overlay = client.get(f"/api/events/{eid}/overlay.png")
        check("overlay.png served", overlay.status_code == 200,
              f"{len(overlay.content) // 1024} KiB")
        check("overlay is a PNG", overlay.content[:4] == b"\x89PNG")

        stages = client.get(f"/api/events/{eid}").json()["stages"]
        for stage in ("overhead", "ground", "fusion"):
            check(f"stage {stage} == done", stages[stage] == "done", stages[stage])

        reports = client.get(f"/api/events/{eid}/ground-reports").json()
        check("ground reports persisted", isinstance(reports, list) and len(reports) >= 4,
              f"{len(reports) if isinstance(reports, list) else reports}")

        print("\n--- unbuilt milestones still honest ---")
        check("allocate -> 501",
              client.post(f"/api/events/{eid}/allocate").status_code == 501)
        check("report.pdf -> 501",
              client.get(f"/api/events/{eid}/report.pdf").status_code == 501)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
