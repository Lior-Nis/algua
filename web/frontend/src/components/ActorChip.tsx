const ACTOR_COLORS: Record<string, string> = {
  agent: 'var(--electric)',
  human: 'var(--amber)',
  system: 'var(--text-dim)',
}

/** Who did it. `human` is the scarce one — an agent action is routine, a human action is a
 * decision, and the audit feed is far more readable when the two are separable at a glance. */
export default function ActorChip({ actor }: { actor: string }) {
  return (
    <span className="stage-chip" style={{ color: ACTOR_COLORS[actor] ?? 'var(--text-dim)' }}>
      {actor}
    </span>
  )
}
