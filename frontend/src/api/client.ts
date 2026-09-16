import type {
  AssessmentSummary,
  BuildingCollection,
  BuildingProperties,
  DamageClass,
  Legend,
  WardBoundaryCollection,
  WardCard,
} from '../types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`${response.status} ${response.statusText} - ${detail.slice(0, 200)}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),

  legend: () => request<Legend>('/api/legend'),

  listAssessments: () => request<AssessmentSummary[]>('/api/assessments'),

  createDemo: () => request<AssessmentSummary>('/api/assessments/demo', { method: 'POST' }),

  wards: (id: string) => request<WardCard[]>(`/api/assessments/${id}/wards`),

  wardBoundaries: (id: string) =>
    request<WardBoundaryCollection>(`/api/assessments/${id}/ward-boundaries`),

  buildings: (id: string, wardId?: string) =>
    request<BuildingCollection>(
      `/api/assessments/${id}/buildings${wardId ? `?ward_id=${wardId}` : ''}`,
    ),

  /** Human-in-the-loop correction. Returns the re-scored ward card. */
  overrideBuilding: (buildingId: string, damageClass: DamageClass, note?: string) =>
    request<WardCard>(`/api/buildings/${buildingId}`, {
      method: 'PATCH',
      body: JSON.stringify({ damage_class: damageClass, note: note ?? null }),
    }),

  building: (buildingId: string) => request<BuildingProperties>(`/api/buildings/${buildingId}`),

  exportUrl: (id: string, layer: 'buildings' | 'wards', format: 'geojson' | 'csv') =>
    `/api/assessments/${id}/export?layer=${layer}&format=${format}`,
}
