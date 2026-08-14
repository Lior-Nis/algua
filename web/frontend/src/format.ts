/** UTC-only timestamp/number formatting (reviewed: all domain timestamps render UTC —
 * browser-local would shift after-close ticks across dates). */

const pad = (n: number) => String(n).padStart(2, '0')

/** Parse an ISO-ish timestamp as UTC. A string with no zone info is ASSUMED UTC
 * (the CLI writes aware-UTC, but tick_ts writers accept arbitrary strings). */
export function parseUtc(ts: string): Date | null {
  let s = ts.trim()
  if (!/(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(s)) s = `${s.replace(' ', 'T')}Z`
  const d = new Date(s)
  return Number.isNaN(d.getTime()) ? null : d
}

/** "2026-08-09" (UTC) — falls back to the raw string when unparseable. */
export function utcDate(ts: string | null | undefined): string {
  if (!ts) return '—'
  const d = parseUtc(ts)
  if (!d) return ts
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
}

/** "14:30:05" (UTC). */
export function utcTime(ts: string | null | undefined): string {
  if (!ts) return '—'
  const d = parseUtc(ts)
  if (!d) return ts
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`
}

/** "2026-08-09 14:30:05Z". */
export function utcDateTime(ts: string | null | undefined): string {
  if (!ts) return '—'
  const d = parseUtc(ts)
  if (!d) return ts
  return `${utcDate(ts)} ${utcTime(ts)}Z`
}

/** Fraction → "12.0%" (sign preserved: -0.03 → "-3.0%"). */
export function pct(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return `${(x * 100).toFixed(1)}%`
}

/** Compact numeric display for tiles/tables. */
export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  if (Number.isInteger(x)) return x.toLocaleString('en-US')
  return x.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}
