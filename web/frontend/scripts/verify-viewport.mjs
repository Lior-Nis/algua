/** Spec §5.1 — the /money bug was a 483px 5-column table sizing its scroll parent instead of
 * scrolling inside it, so the PAGE grew to 500px at a 390px viewport. This turns that class of
 * defect into a test: no route may overflow horizontally at 390px.
 *
 * Serves the DEMO build, so every route has rich data — an empty screen cannot overflow and
 * would make this check vacuous. THAT CLAIM IS ITSELF ASSERTED BELOW (FIX 6, 2026-08-27 fix
 * wave): before measuring overflow on a route, this checks one known fixture string actually
 * landed in the page. Without it, an unfixtured endpoint (demoJSON throws -> the screen renders
 * its error panel, which is narrow and never overflows) would pass this whole script silently —
 * exactly the "new screen's overflow never measured" failure the header comment already warned
 * about but nothing enforced. Mirrors the same idea as the word-budget test's "actually renders
 * fixture data" block. */
import { createServer } from 'node:http'
import { readFileSync, existsSync } from 'node:fs'
import { extname, join, normalize } from 'node:path'
import { chromium } from 'playwright'

const ROUTES = ['/', '/fleet', '/money', '/research', '/s/liquid10_adj_momentum']

// One string per route that only appears if the real fixture data rendered — picked from
// src/fixtures/steady-state.ts (kept as literals here rather than imported: this script runs
// as plain Node against the BUILT demo bundle, not through the TS/Vitest module graph).
const ROUTE_CONTENT = {
  '/': 'cross_horizon_low_vol', // TRIAGE.items[0].title after the worst-first sort (FIX 3/5)
  '/fleet': 'liquid10_adj_momentum', // a real "every strategy" row (FLEET.data.rows)
  '/money': 'liquid10_adj_momentum', // BOOK.data.slices[0].strategy
  '/research': 'liquid10_adj_momentum', // RUNS.data.runs[0].strategy_name
  '/s/liquid10_adj_momentum': 'Cross-sectional momentum over the liquid10 universe', // registry.description
}
const DIR = 'dist-demo'
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
                '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
                '.woff2': 'font/woff2', '.woff': 'font/woff', '.webmanifest': 'application/manifest+json' }

const server = createServer((req, res) => {
  const url = req.url.split('?')[0]
  const candidate = join(DIR, normalize(url).replace(/^(\.\.[/\\])+/, ''))
  const file = existsSync(candidate) && extname(candidate) ? candidate : join(DIR, 'index.html')
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' })
  res.end(readFileSync(file))
})

await new Promise((r) => server.listen(0, '127.0.0.1', r))
const base = `http://127.0.0.1:${server.address().port}`

const browser = await chromium.launch({ channel: 'chrome' })
const page = await browser.newPage({ viewport: { width: 390, height: 844 } })
const failures = []

for (const route of ROUTES) {
  await page.goto(base + route, { waitUntil: 'networkidle' })
  await page.waitForTimeout(600)

  const expected = ROUTE_CONTENT[route]
  const bodyText = await page.evaluate(() => document.body.innerText)
  if (expected === undefined || !bodyText.includes(expected)) {
    failures.push(
      `${route}: expected fixture text "${expected}" not found in the rendered page — the ` +
        'route likely errored/emptied instead of rendering real data, which would make the ' +
        'overflow check below vacuous',
    )
    console.log(`FAIL ${route} missing expected content "${expected}"`)
    continue
  }

  const m = await page.evaluate(() => {
    const vw = window.innerWidth
    // An element clipped by an ancestor's own horizontal scrollport (overflow-x: auto/scroll/
    // hidden) is the INTENDED "scrolls inside its own container" pattern, not page overflow —
    // its bounding rect can legitimately extend past the viewport without the PAGE growing.
    // Walking the ancestor chain (not just the immediate parent) catches a clip several levels up.
    function clippedByScrollAncestor(el) {
      for (let node = el.parentElement; node && node !== document.documentElement; node = node.parentElement) {
        const ox = getComputedStyle(node).overflowX
        if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') return true
      }
      return false
    }
    const offenders = []
    document.querySelectorAll('*').forEach((el) => {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.right > vw + 1 && !clippedByScrollAncestor(el)) {
        offenders.push(`${el.tagName}.${String(el.className).slice(0, 40)} right=${Math.round(r.right)}`)
      }
    })
    return { vw, scrollWidth: document.documentElement.scrollWidth, offenders: offenders.slice(0, 5) }
  })
  // scrollWidth is the authoritative signal for "did the PAGE overflow" (the actual /money
  // symptom: the viewport itself widens/zooms out). offenders names the culprit when it does.
  const overflow = m.scrollWidth > m.vw
  console.log(`${overflow ? 'FAIL' : 'ok  '} ${route} vw=${m.vw} scrollWidth=${m.scrollWidth}`)
  if (overflow) failures.push(`${route}: scrollWidth=${m.scrollWidth} vw=${m.vw}\n    ${m.offenders.join('\n    ')}`)
}

await browser.close()
server.close()

if (failures.length) {
  console.error(`\nHorizontal overflow at 390px:\n  ${failures.join('\n  ')}`)
  process.exit(1)
}
console.log(`\nok: ${ROUTES.length} routes, no horizontal overflow at 390px`)
