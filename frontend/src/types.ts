// Mirrors backend/app/schemas.py - the contract agreed in hour 1 of the sprint.

export type DamageClass = 'no-damage' | 'minor-damage' | 'major-damage' | 'destroyed'

export interface BuildingProperties {
  id: string
  ward_id: string
  damage_class: DamageClass
  confidence: number
  reviewed_by_human: boolean
  review_note: string | null
}

export interface PriorityBreakdown {
  damage: number
  population: number
  infrastructure: number
  accessibility: number
}

export interface WardCard {
  ward_id: string
  ward_name: string
  total_buildings: number
  damaged_buildings: number
  severely_damaged: number
  damage_breakdown: Record<DamageClass, number>
  affected_population: number
  critical_facilities: number
  facility_breakdown: Record<string, number>
  facility_types: string[]
  priority_score: number
  priority_band: string
  priority_color: string
  breakdown: PriorityBreakdown
  reviewed_count: number
}

export interface AssessmentSummary {
  assessment_id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  mode: 'model' | 'heuristic' | 'mock'
  geometry_source: 'osm' | 'synthetic'
  aoi_name: string | null
  created_at: string
  total_buildings: number
  damaged_buildings: number
  severely_damaged: number
  affected_population: number
  wards: number
  error: string | null
}

export interface LegendEntry {
  key: string
  label: string
  color: string
}

export interface Legend {
  damage: LegendEntry[]
  priority: LegendEntry[]
  critical_infrastructure: LegendEntry
}

export type BuildingFeature = GeoJSON.Feature<GeoJSON.Polygon, BuildingProperties>
export type BuildingCollection = GeoJSON.FeatureCollection<GeoJSON.Polygon, BuildingProperties>

export interface WardBoundaryProperties {
  ward_id: string
  ward_name: string
  priority_score: number
  priority_band: string
  priority_color: string
}
export type WardBoundaryCollection = GeoJSON.FeatureCollection<GeoJSON.Polygon, WardBoundaryProperties>
