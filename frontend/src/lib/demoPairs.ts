// Bundled fallback for the demo-pair list, mirroring GET /demo/pairs on the
// deployed backend (same ids, same order).
//
// Why it exists: the dropdown used to stay empty until a /health probe
// succeeded, so during a free-tier cold start (or any backend restart) there
// was nothing to click and the demo was dead in the water. The pair list is
// static content shipped in this repo's data/demo — there is no reason to
// gate it on a network round-trip. The live list still replaces this the
// moment the backend answers, so a changed demo set wins once it's reachable;
// analyze/image requests to a waking Render host are held by its proxy until
// boot completes, so a demo started from this fallback still works.
import type { DemoPair } from "./api";

export const FALLBACK_DEMO_PAIRS: DemoPair[] = [
  {
    id: "river-valley-flood_01",
    disaster_type: "flood",
    pre_image: "river-valley-flood_01_pre_disaster.png",
    post_image: "river-valley-flood_01_post_disaster.png",
  },
  {
    id: "river-valley-flood_02",
    disaster_type: "flood",
    pre_image: "river-valley-flood_02_pre_disaster.png",
    post_image: "river-valley-flood_02_post_disaster.png",
  },
  {
    id: "river-valley-flood_03",
    disaster_type: "flood",
    pre_image: "river-valley-flood_03_pre_disaster.png",
    post_image: "river-valley-flood_03_post_disaster.png",
  },
];
