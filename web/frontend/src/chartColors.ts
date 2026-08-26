/** Shared `getComputedStyle` token reader for the uPlot-driven charts (`EquityChart`,
 * `ReturnOverlay`) — the only two views in this slice that configure canvas series colors in
 * JS rather than CSS classes, since uPlot takes its series `stroke` as a plain string at
 * construction time, not a stylesheet rule. Kept in one place so a token name or fallback value
 * drifts once, not twice between the two call sites. */
export function cssColor(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}
