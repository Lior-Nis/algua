import type { ReactNode } from 'react'

export default function MetricTile({
  label,
  value,
  color,
}: {
  label: string
  value: ReactNode
  color?: string
}) {
  return (
    <div className="metric-tile">
      <div className="metric-value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="metric-label">{label}</div>
    </div>
  )
}
