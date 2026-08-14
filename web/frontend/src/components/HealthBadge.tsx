const HEALTH_COLORS: Record<string, string> = {
  ok: 'var(--green)',
  idle: 'var(--text-dim)',
  stale: 'var(--amber)',
  drift: 'var(--violet)',
  halted: 'var(--red)',
}

export function healthColor(health: string): string {
  return HEALTH_COLORS[health] ?? 'var(--text-dim)'
}

/**
 * `muted` renders the verdict factually but WITHOUT alarm color, for rows where health
 * is not an alert. fleet_alert only alerts on operational stages (live/paper/
 * forward_tested): a benched or retired strategy is `idle` because nothing ticks it, and
 * even a lingering kill-switch on it is deliberately quiet ("a benched strategy can never
 * wedge the watchdog permanently red"). Painting those red contradicts the gate.
 */
export default function HealthBadge({
  health,
  muted = false,
}: {
  health: string
  muted?: boolean
}) {
  return (
    <span
      className="health-badge"
      style={{ color: muted ? 'var(--text-dim)' : healthColor(health) }}
      title={muted ? `${health} — not watched at this stage (no operator loop)` : undefined}
    >
      {health}
    </span>
  )
}
