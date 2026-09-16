import { useCallback, useEffect, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import { api } from './api/client'
import { DamageMap } from './components/DamageMap'
import { Legend } from './components/Legend'
import { ReviewDrawer } from './components/ReviewDrawer'
import { WardPanel } from './components/WardPanel'
import type {
  AssessmentSummary,
  BuildingCollection,
  BuildingFeature,
  DamageClass,
  Legend as LegendData,
  WardBoundaryCollection,
  WardCard,
} from './types'

export default function App() {
  const [assessment, setAssessment] = useState<AssessmentSummary | null>(null)
  const [wards, setWards] = useState<WardCard[]>([])
  const [buildings, setBuildings] = useState<BuildingCollection | null>(null)
  const [boundaries, setBoundaries] = useState<WardBoundaryCollection | null>(null)
  const [legend, setLegend] = useState<LegendData | null>(null)

  const [selectedWard, setSelectedWard] = useState<string | null>(null)
  const [selectedBuilding, setSelectedBuilding] = useState<BuildingFeature | null>(null)
  const [showUndamaged, setShowUndamaged] = useState(false)

  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const damageColors = Object.fromEntries(
    (legend?.damage ?? []).map((entry) => [entry.key, entry.color]),
  )

  const loadAssessment = useCallback(async (id: string) => {
    const [wardCards, wardBoundaries, buildingCollection] = await Promise.all([
      api.wards(id),
      api.wardBoundaries(id),
      api.buildings(id),
    ])
    setWards(wardCards)
    setBoundaries(wardBoundaries)
    setBuildings(buildingCollection)
  }, [])

  // Boot: pick up an existing assessment, otherwise generate the demo AOI.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        setLoading(true)
        const legendData = await api.legend()
        const existing = await api.listAssessments()
        const summary = existing[0] ?? (await api.createDemo())
        if (cancelled) return
        setLegend(legendData)
        setAssessment(summary)
        await loadAssessment(summary.assessment_id)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [loadAssessment])

  const runNewAssessment = async () => {
    try {
      setLoading(true)
      setError(null)
      setSelectedWard(null)
      setSelectedBuilding(null)
      const summary = await api.createDemo()
      setAssessment(summary)
      await loadAssessment(summary.assessment_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  /** A human corrects the AI. The ward card and the map update in place. */
  const handleOverride = async (buildingId: string, damageClass: DamageClass, note: string) => {
    if (!assessment) return
    try {
      setSaving(true)
      const updatedWard = await api.overrideBuilding(buildingId, damageClass, note || undefined)

      setWards((current) =>
        current
          .map((w) => (w.ward_id === updatedWard.ward_id ? updatedWard : w))
          .sort((a, b) => b.priority_score - a.priority_score),
      )

      setBuildings((current) => {
        if (!current) return current
        return {
          ...current,
          features: current.features.map((f) =>
            f.properties.id === buildingId
              ? {
                  ...f,
                  properties: {
                    ...f.properties,
                    damage_class: damageClass,
                    confidence: 1,
                    reviewed_by_human: true,
                    review_note: note || null,
                  },
                }
              : f,
          ),
        }
      })

      // Ward polygons carry the priority color, so they need the fresh score too.
      const refreshed = await api.wardBoundaries(assessment.assessment_id)
      setBoundaries(refreshed)
      setSelectedBuilding(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const totals = wards.reduce(
    (acc, w) => ({
      damaged: acc.damaged + w.damaged_buildings,
      severe: acc.severe + w.severely_damaged,
      population: acc.population + w.affected_population,
      reviewed: acc.reviewed + w.reviewed_count,
    }),
    { damaged: 0, severe: 0, population: 0, reviewed: 0 },
  )

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>Disaster Damage Assessment</h1>
            <p>
              {assessment?.aoi_name ?? 'No area loaded'}
              {assessment && (
                <>
                  <span className={`mode mode-${assessment.geometry_source}`}>
                    {assessment.geometry_source === 'osm' ? 'OSM + WorldPop' : 'synthetic map'}
                  </span>
                  <span className={`mode mode-${assessment.mode}`}>
                    {assessment.mode === 'mock' ? 'simulated damage' : assessment.mode}
                  </span>
                </>
              )}
            </p>
          </div>
        </div>

        <div className="totals">
          <Stat value={totals.damaged} label="damaged" />
          <Stat value={totals.severe} label="severe" accent="#dc2626" />
          <Stat value={totals.population.toLocaleString()} label="people affected" />
          <Stat value={totals.reviewed} label="human-reviewed" accent="#38bdf8" />
        </div>

        <div className="actions">
          {assessment && (
            <>
              <a
                className="ghost"
                href={api.exportUrl(assessment.assessment_id, 'wards', 'csv')}
                download
              >
                Export wards
              </a>
              <a
                className="ghost"
                href={api.exportUrl(assessment.assessment_id, 'buildings', 'geojson')}
                download
              >
                Export GeoJSON
              </a>
            </>
          )}
          <button className="primary" onClick={runNewAssessment} disabled={loading}>
            {loading ? 'Running…' : 'Run assessment'}
          </button>
        </div>
      </header>

      {error && (
        <div className="banner error">
          {error}
          <button className="icon" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}

      {assessment?.mode === 'mock' && (
        <div className="banner warn">
          {assessment.geometry_source === 'osm' ? (
            <>
              Real wards, buildings and facilities (OpenStreetMap) with census-anchored
              population (WorldPop). <strong>Damage is simulated</strong> from a shaking field —
              no imagery has been classified yet.
            </>
          ) : (
            <>Synthetic demo map — no real geometry or imagery. Results are illustrative.</>
          )}
        </div>
      )}

      <main>
        <section className="map-wrap">
          <DamageMap
            buildings={buildings}
            boundaries={boundaries}
            damageColors={damageColors}
            selectedWard={selectedWard}
            selectedBuilding={selectedBuilding?.properties.id ?? null}
            showUndamaged={showUndamaged}
            onSelectWard={setSelectedWard}
            onSelectBuilding={setSelectedBuilding}
          />
          <Legend
            legend={legend}
            showUndamaged={showUndamaged}
            onToggleUndamaged={setShowUndamaged}
          />
          {selectedWard && (
            <button className="clear-filter" onClick={() => setSelectedWard(null)}>
              Showing {wards.find((w) => w.ward_id === selectedWard)?.ward_name} — show all
            </button>
          )}
        </section>

        <aside className="sidebar">
          <h2>Response priority</h2>
          <p className="sub">Where to send teams first. Click a ward to focus the map.</p>
          {loading ? (
            <p className="empty">Loading assessment…</p>
          ) : (
            <WardPanel wards={wards} selectedWard={selectedWard} onSelectWard={setSelectedWard} />
          )}
        </aside>
      </main>

      <ReviewDrawer
        building={selectedBuilding}
        legend={legend}
        busy={saving}
        onClose={() => setSelectedBuilding(null)}
        onOverride={handleOverride}
      />
    </div>
  )
}

function Stat({
  value,
  label,
  accent,
}: {
  value: number | string
  label: string
  accent?: string
}) {
  return (
    <div className="stat">
      <strong style={accent ? { color: accent } : undefined}>{value}</strong>
      <span>{label}</span>
    </div>
  )
}
