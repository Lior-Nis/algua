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
 * Renders the health verdict with its alarm color. The `muted` opt-out (rendering a verdict
 * factually but without alarm color, for a stage `fleet_alert` never watches — a benched/retired
 * strategy is `idle` because nothing ticks it) was removed (fix round 2, deferred item promoted):
 * it had no consumer, and the semantic it protected — a benched strategy can never wedge the
 * watchdog permanently red — is enforced server-side, in `fleet status`'s own stage-aware
 * verdict, not by a client-side color override.
 */
export default function HealthBadge({ health }: { health: string }) {
  return (
    <span className="health-badge" style={{ color: healthColor(health) }}>
      {health}
    </span>
  )
}
