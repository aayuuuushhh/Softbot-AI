import { useEffect, useMemo } from 'react'
import { GeoJSON, MapContainer, TileLayer, useMap } from 'react-leaflet'
import type { Layer, PathOptions } from 'leaflet'
import L from 'leaflet'
import type {
  BuildingCollection,
  BuildingFeature,
  DamageClass,
  WardBoundaryCollection,
} from '../types'

const KATHMANDU: [number, number] = [27.727, 85.326]

interface Props {
  buildings: BuildingCollection | null
  boundaries: WardBoundaryCollection | null
  damageColors: Record<string, string>
  selectedWard: string | null
  selectedBuilding: string | null
  showUndamaged: boolean
  onSelectWard: (wardId: string | null) => void
  onSelectBuilding: (feature: BuildingFeature) => void
}

/** Pans and zooms when the responder picks a ward from the priority list. */
function FlyToWard({
  boundaries,
  wardId,
}: {
  boundaries: WardBoundaryCollection | null
  wardId: string | null
}) {
  const map = useMap()
  useEffect(() => {
    if (!boundaries) return
    if (!wardId) {
      map.flyToBounds(L.geoJSON(boundaries).getBounds(), { padding: [30, 30], duration: 0.6 })
      return
    }
    const feature = boundaries.features.find((f) => f.properties.ward_id === wardId)
    if (feature) {
      map.flyToBounds(L.geoJSON(feature).getBounds(), { padding: [60, 60], duration: 0.6 })
    }
  }, [wardId, boundaries, map])
  return null
}

export function DamageMap({
  buildings,
  boundaries,
  damageColors,
  selectedWard,
  selectedBuilding,
  showUndamaged,
  onSelectWard,
  onSelectBuilding,
}: Props) {
  const visible = useMemo<BuildingCollection | null>(() => {
    if (!buildings) return null
    const features = buildings.features.filter((f) => {
      if (!showUndamaged && f.properties.damage_class === 'no-damage') return false
      if (selectedWard && f.properties.ward_id !== selectedWard) return false
      return true
    })
    return { type: 'FeatureCollection', features }
  }, [buildings, showUndamaged, selectedWard])

  const buildingStyle = (feature?: BuildingFeature): PathOptions => {
    const props = feature?.properties
    const damage = (props?.damage_class ?? 'no-damage') as DamageClass
    const isSelected = props?.id === selectedBuilding
    return {
      color: props?.reviewed_by_human ? '#f8fafc' : damageColors[damage] ?? '#94a3b8',
      weight: isSelected ? 3 : props?.reviewed_by_human ? 1.6 : 0.8,
      fillColor: damageColors[damage] ?? '#94a3b8',
      fillOpacity: damage === 'no-damage' ? 0.25 : 0.75,
    }
  }

  const boundaryStyle = (feature?: GeoJSON.Feature): PathOptions => {
    const props = feature?.properties as WardBoundaryCollection['features'][0]['properties']
    const isSelected = props?.ward_id === selectedWard
    return {
      color: props?.priority_color ?? '#64748b',
      weight: isSelected ? 3 : 1.5,
      fillColor: props?.priority_color ?? '#64748b',
      fillOpacity: isSelected ? 0.16 : 0.07,
      dashArray: isSelected ? undefined : '4 4',
    }
  }

  const onEachBuilding = (feature: BuildingFeature, layer: Layer) => {
    const p = feature.properties
    layer.bindTooltip(
      `<b>${p.id}</b><br/>${p.damage_class}<br/>confidence ${(p.confidence * 100).toFixed(0)}%` +
        (p.reviewed_by_human ? '<br/><i>reviewed by a human</i>' : ''),
      { sticky: true },
    )
    layer.on('click', (event) => {
      L.DomEvent.stopPropagation(event as unknown as Event)
      onSelectBuilding(feature)
    })
  }

  const onEachBoundary = (feature: GeoJSON.Feature, layer: Layer) => {
    const p = feature.properties as WardBoundaryCollection['features'][0]['properties']
    layer.bindTooltip(`<b>${p.ward_name}</b><br/>Priority: ${p.priority_band}`, { sticky: true })
    layer.on('click', () => onSelectWard(p.ward_id === selectedWard ? null : p.ward_id))
  }

  // Leaflet caches layer styles, so the GeoJSON layers are keyed on everything that
  // changes their appearance. Cheap at demo scale and avoids stale colors after a review.
  const buildingsKey = `b-${visible?.features.length ?? 0}-${selectedBuilding}-${selectedWard}`
  const boundariesKey = `w-${boundaries?.features.map((f) => f.properties.priority_score).join()}-${selectedWard}`

  return (
    <MapContainer center={KATHMANDU} zoom={13} className="map" zoomControl={false}>
      {/* Esri World Imagery: free, no API key, and satellite is the right backdrop for a
          tool whose whole input is satellite imagery. */}
      <TileLayer
        attribution="Imagery &copy; Esri, Maxar, Earthstar Geographics"
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        maxZoom={19}
      />
      {/* Place names and roads on top, so responders can orient themselves. */}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        maxZoom={19}
      />
      {boundaries && (
        <GeoJSON
          key={boundariesKey}
          data={boundaries}
          style={boundaryStyle}
          onEachFeature={onEachBoundary}
        />
      )}
      {visible && (
        <GeoJSON
          key={buildingsKey}
          data={visible}
          style={buildingStyle as never}
          onEachFeature={onEachBuilding as never}
        />
      )}
      <FlyToWard boundaries={boundaries} wardId={selectedWard} />
    </MapContainer>
  )
}
