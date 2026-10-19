export type EvidenceState = 'loading' | 'failed' | 'stale' | 'incomplete' | 'empty'

/** Compact semantic replacement for a chart when there is no honest plot to draw. */
export default function EvidenceStatus({
  title,
  state,
  detail,
  onRetry,
}: {
  title: string
  state: EvidenceState
  detail: string
  onRetry?: () => void
}) {
  return (
    <div
      className="evidence-status"
      data-testid="evidence-status"
      data-state={state}
      role={state === 'failed' ? 'alert' : 'status'}
    >
      <span className="evidence-status-title">{title}</span>
      <span className="evidence-status-state">{state}</span>
      <span>{detail}</span>
      {onRetry !== undefined && (
        <button type="button" className="evidence-retry" onClick={onRetry}>retry</button>
      )}
    </div>
  )
}
