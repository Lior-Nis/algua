/** The deflation strip's marker draws its color from `theme.css` (CSS classes, not an inline
 * style prop), unlike `PassMark`'s inline `style.color` — so this mirrors `PassMark.test.tsx`'s
 * assertion ("an advisory failure reads amber, never the red of a breached binding floor") by
 * reading the actual rule out of the stylesheet, rather than asserting a computed style jsdom
 * never applies (no CSS is loaded into the test DOM). */
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

describe('.trial-dist-marker-fail (the deflation-strip advisory marker)', () => {
  it('never renders in --red — the underlying holdout_sharpe check is ALWAYS advisory, so a ' +
    'fail here can never be a breached binding floor', () => {
    const css = readFileSync(THEME_CSS_PATH, 'utf-8')
    const body = ruleBodyFor(css, '.trial-dist-marker-fail')
    expect(body).not.toMatch(/--red\b/)
  })

  it('renders in --amber, matching GateBulletCard.bullet-fill-advisory-fail and ' +
    "PassMark's advisory-fail branch", () => {
    const css = readFileSync(THEME_CSS_PATH, 'utf-8')
    const body = ruleBodyFor(css, '.trial-dist-marker-fail')
    expect(body).toMatch(/--amber\b/)
  })
})
