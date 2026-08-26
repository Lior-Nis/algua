/** Colour follows the ENTITY, never its rank or role (the dataviz non-negotiable).
 *
 * The defect this locks down: `--series-focus` (Electric) meant "this strategy" in the trial
 * cloud and "the deflated bar" in the deflation strip one card-section away, while this
 * strategy's own diamond in the strip was amber. Three referents, two marks, one card.
 *
 * Read out of `theme.css` rather than off a computed style — no CSS is loaded into the test DOM
 * (the same technique, and the same reason, as `TrialDistribution.advisorycolor.test.tsx`). */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const THEME_CSS_PATH = path.resolve(__dirname, '../theme.css')

function ruleBodyFor(css: string, selector: string): string {
  const pattern = new RegExp(`${selector.replace(/[.]/g, '\\.')}\\s*\\{([^}]*)\\}`)
  const match = css.match(pattern)
  if (match === null) throw new Error(`selector not found in theme.css: ${selector}`)
  return match[1]
}

describe('the trial-distribution card paints one identity per entity', () => {
  it('this strategy is --series-focus in BOTH marks — the cloud diamond and the strip diamond', () => {
    const css = readFileSync(THEME_CSS_PATH, 'utf-8')
    expect(ruleBodyFor(css, '.trial-dist-own-cloud-marker')).toMatch(/--series-focus\b/)
    expect(ruleBodyFor(css, '.trial-dist-own-strip-marker')).toMatch(/--series-focus\b/)
  })

  it('the deflated bar is CHROME, not the brand signal — a threshold is a reference the marks ' +
    'are judged against, not data about a strategy', () => {
    const css = readFileSync(THEME_CSS_PATH, 'utf-8')
    const body = ruleBodyFor(css, '.trial-dist-threshold')
    expect(body).not.toMatch(/--series-focus\b/)
    // The same token `.scatter-diagonal` (y=x) and `.sparkline-baseline` (zero) already use.
    expect(body).toMatch(/--slate\b/)
    expect(ruleBodyFor(css, '.scatter-diagonal')).toMatch(/--slate\b/)
  })

  it("the bar's LABEL is text, so it takes a text token — --slate is 3.62:1 on Obsidian, " +
    'below the AA floor for small text', () => {
    const css = readFileSync(THEME_CSS_PATH, 'utf-8')
    const body = ruleBodyFor(css, '.trial-dist-threshold-label')
    expect(body).not.toMatch(/--slate\b/)
    expect(body).toMatch(/--text-dim\b/)
  })
})
