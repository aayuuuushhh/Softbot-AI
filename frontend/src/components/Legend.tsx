import type { Legend as LegendData } from '../types'

interface Props {
  legend: LegendData | null
  showUndamaged: boolean
  onToggleUndamaged: (value: boolean) => void
}

export function Legend({ legend, showUndamaged, onToggleUndamaged }: Props) {
  if (!legend) return null
  return (
    <div className="legend">
      <div className="legend-title">Damage</div>
      {legend.damage.map((entry) => (
        <div className="legend-row" key={entry.key}>
          <span className="swatch" style={{ background: entry.color }} />
          {entry.label}
        </div>
      ))}
      <div className="legend-row">
        <span className="swatch" style={{ background: legend.critical_infrastructure.color }} />
        {legend.critical_infrastructure.label}
      </div>
      <label className="legend-toggle">
        <input
          type="checkbox"
          checked={showUndamaged}
          onChange={(e) => onToggleUndamaged(e.target.checked)}
        />
        Show undamaged
      </label>
    </div>
  )
}
