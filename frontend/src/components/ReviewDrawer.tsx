import { useEffect, useState } from 'react'
import type { BuildingFeature, DamageClass, Legend } from '../types'

interface Props {
  building: BuildingFeature | null
  legend: Legend | null
  busy: boolean
  onClose: () => void
  onOverride: (buildingId: string, damageClass: DamageClass, note: string) => void
}

const ORDER: DamageClass[] = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

/**
 * Human-in-the-loop review. An official inspects what the AI said about one building and
 * corrects it; the ward re-scores immediately. The AI does the first pass, people decide.
 */
export function ReviewDrawer({ building, legend, busy, onClose, onOverride }: Props) {
  const [choice, setChoice] = useState<DamageClass | null>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    setChoice(building?.properties.damage_class ?? null)
    setNote(building?.properties.review_note ?? '')
  }, [building])

  if (!building) return null

  const p = building.properties
  const labels = new Map(legend?.damage.map((e) => [e.key, e]) ?? [])
  const changed = choice !== null && choice !== p.damage_class
  const lowConfidence = p.confidence < 0.7 && !p.reviewed_by_human

  return (
    <aside className="drawer">
      <header>
        <div>
          <h3>{p.id}</h3>
          <span className="muted">{p.ward_id}</span>
        </div>
        <button className="icon" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>

      <div className="drawer-body">
        <div className="ai-verdict">
          <span className="muted">AI assessment</span>
          <div className="verdict-row">
            <span
              className="swatch"
              style={{ background: labels.get(p.damage_class)?.color ?? '#94a3b8' }}
            />
            <strong>{labels.get(p.damage_class)?.label ?? p.damage_class}</strong>
            <span className={`confidence${lowConfidence ? ' low' : ''}`}>
              {(p.confidence * 100).toFixed(0)}% confident
            </span>
          </div>
          {lowConfidence && (
            <p className="hint">Low confidence — worth a human look before dispatch.</p>
          )}
          {p.reviewed_by_human && <p className="hint reviewed">Already reviewed by a human.</p>}
        </div>

        <label className="field-label">Official assessment</label>
        <div className="choices">
          {ORDER.map((cls) => (
            <button
              key={cls}
              className={`choice${choice === cls ? ' active' : ''}`}
              style={{ borderColor: labels.get(cls)?.color ?? '#475569' }}
              onClick={() => setChoice(cls)}
            >
              <span className="swatch" style={{ background: labels.get(cls)?.color }} />
              {labels.get(cls)?.label ?? cls}
            </button>
          ))}
        </div>

        <label className="field-label" htmlFor="note">
          Note (optional)
        </label>
        <textarea
          id="note"
          value={note}
          placeholder="e.g. Verified collapsed by ward officer"
          onChange={(e) => setNote(e.target.value)}
        />

        <button
          className="primary"
          disabled={!changed || busy}
          onClick={() => choice && onOverride(p.id, choice, note)}
        >
          {busy ? 'Saving…' : changed ? 'Confirm correction & re-score ward' : 'No change to save'}
        </button>
      </div>
    </aside>
  )
}
