// Coordinate formatting and external map deep links.
//
// Zone centroids come from the backend's affine pixel -> lat/lng fit over xBD
// building label correspondences, so they are real ground coordinates. These
// links let a coordinator jump from a ranked zone straight to the actual place.
// Deep links only — no Maps JS API, so there is no API key to leak or bill.

/** Six decimals ~= 0.1 m, well past what the affine fit resolves. */
export function formatLatLng(lat: number, lng: number): string {
  return `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
}

/**
 * Five decimals (~1 m) for the narrow zone-table column, where the six-decimal
 * form wraps onto a second line. A zone centroid is the middle of a grid cell
 * hundreds of metres across, so the dropped digit carries no real information.
 */
export function formatLatLngCompact(lat: number, lng: number): string {
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

/** Google Maps, dropped on the exact point. */
export function googleMapsUrl(lat: number, lng: number): string {
  return `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
}

/**
 * Google Earth web, camera over the point.
 * Format: @lat,lng,<elevation>a,<range>d,<tilt>y — 800 m out, looking straight
 * down, which frames a zone rather than a whole city.
 */
export function googleEarthUrl(lat: number, lng: number): string {
  return `https://earth.google.com/web/@${lat},${lng},0a,800d,35y,0h,0t,0r`;
}
