/** Spec §6 guard 1: the PRODUCTION build must never bundle fixtures.
 *
 * This greps the built output for the fixture sentinel rather than trusting that Rollup
 * tree-shook the dynamic import. If this ever fails, the fix is a build-time alias that
 * removes the module from the graph — NOT deleting this check. */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const SENTINEL = 'ALGUA_DEMO_FIXTURE_a7f3e1'

function walk(dir) {
  return readdirSync(dir).flatMap((entry) => {
    const p = join(dir, entry)
    return statSync(p).isDirectory() ? walk(p) : [p]
  })
}

const [, , mode = 'prod', dir = 'dist'] = process.argv
const files = walk(dir)
const hits = files.filter((f) => readFileSync(f, 'utf8').includes(SENTINEL))

if (mode === 'prod' && hits.length > 0) {
  console.error(`FAIL: production build carries fixture data:\n  ${hits.join('\n  ')}`)
  process.exit(1)
}
if (mode === 'demo' && hits.length === 0) {
  console.error(`FAIL: demo build in ${dir} contains NO fixture data — it would render empty.`)
  process.exit(1)
}
console.log(`ok: ${mode} build in ${dir} (${files.length} files, ${hits.length} with fixtures)`)
