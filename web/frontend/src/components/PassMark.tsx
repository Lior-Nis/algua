/** Shared pass/fail verdict mark. Originated in `StrategyDetail.tsx` (the gate bullet card and
 * per-check rows) and is reused verbatim by the strategy board rather than re-implementing the
 * same branching a second time. This remains the superset because gate checks use `advisory`.
 */
export default function PassMark({
  passed,
  advisory = false,
}: {
  passed: number | boolean | null | undefined
  advisory?: boolean
}) {
  // A missing/malformed verdict is UNKNOWN data, not a real failed check — never fabricate one.
  if (passed !== true && passed !== 1 && passed !== false && passed !== 0) {
    return (
      <span className="pass-mark" style={{ color: 'var(--text-dim)' }}>
        unknown
      </span>
    )
  }
  const ok = passed === true || passed === 1
  // An ADVISORY check has no veto power, so a failed one is a warning inside a passing
  // gate — gold, never the red of a breached binding floor.
  const color = advisory
    ? ok
      ? 'var(--text-dim)'
      : 'var(--amber)'
    : ok
      ? 'var(--green)'
      : 'var(--red)'
  return (
    <span className="pass-mark" style={{ color }}>
      {ok ? 'pass' : 'fail'}
    </span>
  )
}
