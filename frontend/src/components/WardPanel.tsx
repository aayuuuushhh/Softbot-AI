import type { WardCard } from '../types'

interface Props {
  wards: WardCard[]
  selectedWard: string | null
  onSelectWard: (wardId: string | null) => void
}

const FACILITY_LABELS: Record<string, [string, string]> = {
  hospital: ['hospital', 'hospitals'],
  bridge: ['bridge', 'bridges'],
  fire_station: ['fire station', 'fire stations'],
  school: ['school', 'schools'],
  police: ['police post', 'police posts'],
}

// Life-safety infrastructure first - that is the order a dispatcher reads in.
const FACILITY_ORDER = ['hospital', 'bridge', 'fire_station', 'school', 'police']

function facilityLabel(type: string, count: number): string {
  const [singular, plural] = FACILITY_LABELS[type] ?? [type, type]
  return `${count} ${count === 1 ? singular : plural}`
}

/**
 * The brief's ward card:
 *   Ward 8: 127 potentially damaged buildings / 34 severely damaged /
 *   Estimated affected population: 2,400 / 3 critical facilities nearby / Priority: Very High
 */
const HEADLINE = new Set(['hospital', 'bridge', 'fire_station'])

export function WardPanel({ wards, selectedWard, onSelectWard }: Props) {
  if (!wards.length) {
    return <p className="empty">No wards assessed yet.</p>
  }

  return (
    <div className="ward-list">
      {wards.map((ward, index) => {
        const isSelected = ward.ward_id === selectedWard
        return (
          <button
            key={ward.ward_id}
            className={`ward-card${isSelected ? ' selected' : ''}`}
            style={{ borderLeftColor: ward.priority_color }}
            onClick={() => onSelectWard(isSelected ? null : ward.ward_id)}
          >
            <div className="ward-head">
              <span className="rank">#{index + 1}</span>
              <span className="ward-name">{ward.ward_name}</span>
              <span className="band" style={{ background: ward.priority_color }}>
                {ward.priority_band}
              </span>
            </div>

            <div className="ward-stats">
              <div>
                <strong>{ward.damaged_buildings}</strong>
                <span>potentially damaged</span>
              </div>
              <div>
                <strong>{ward.severely_damaged}</strong>
                <span>severely damaged</span>
              </div>
              <div>
                <strong>{ward.affected_population.toLocaleString()}</strong>
                <span>affected population</span>
              </div>
              <div>
                <strong>{ward.critical_facilities}</strong>
                <span>critical facilities</span>
              </div>
            </div>

            {Object.keys(ward.facility_breakdown).length > 0 && (
              <div className="facilities">
                {FACILITY_ORDER.filter((t) => ward.facility_breakdown[t]).map((t) => (
                  <span className={`chip${HEADLINE.has(t) ? ' headline' : ''}`} key={t}>
                    {facilityLabel(t, ward.facility_breakdown[t])}
                  </span>
                ))}
              </div>
            )}

            {/* Why this ward ranks here - responders should not have to trust a bare number. */}
            <div className="breakdown" title="Contribution to the priority score">
              <Bar label="Damage" value={ward.breakdown.damage} />
              <Bar label="Population" value={ward.breakdown.population} />
              <Bar label="Infra" value={ward.breakdown.infrastructure} />
              <Bar label="Access" value={ward.breakdown.accessibility} />
            </div>

            {ward.reviewed_count > 0 && (
              <div className="reviewed-note">
                {ward.reviewed_count} building{ward.reviewed_count > 1 ? 's' : ''} reviewed by a
                human
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <span className="bar-track">
        <span className="bar-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
    </div>
  )
}
