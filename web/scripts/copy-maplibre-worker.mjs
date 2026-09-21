// MapLibre 6 loads its web worker as a separate ES module resolved at runtime
// (new URL(name, import.meta.url) with a variable name), so no bundler can see
// it. Serve the worker and the chunk it imports from /public/maplibre instead;
// components/MapView.tsx points setWorkerUrl() there. Runs on postinstall so the
// copy always matches the installed version.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "maplibre-gl", "dist");
const dst = join(root, "public", "maplibre");
mkdirSync(dst, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) {
  copyFileSync(join(src, f), join(dst, f));
}
console.log(`maplibre worker copied to ${dst}`);
