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

export default function HealthBadge({ health }: { health: string }) {
  return (
    <span className="health-badge" style={{ color: healthColor(health) }}>
      {health}
    </span>
  )
}
