/** Every `var(--token)` a component references MUST be defined in theme.css.
 *
 * An undefined custom property is not an error in CSS — the declaration is simply
 * invalid at computed-value time and the element silently inherits instead. So a
 * renamed token (the brand rebrand moved `--gold` -> `--amber` and `--cyan` ->
 * `--electric`) leaves behind code that still "looks" styled, still passes a test
 * asserting the style ATTRIBUTE string, and renders with no color at all. This is
 * the only cheap way to catch it.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { expect, it } from 'vitest'

// vitest runs with cwd = web/frontend; import.meta.url is rewritten by the transform.
const SRC = join(process.cwd(), 'src')

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return /\.(ts|tsx|css)$/.test(entry) && !entry.endsWith('.test.ts') &&
      !entry.endsWith('.test.tsx')
      ? [path]
      : []
  })
}

/** Tokens defined at the top of a `:root { ... }` block in theme.css. */
function definedTokens(): Set<string> {
  const css = readFileSync(join(SRC, 'theme.css'), 'utf8')
  return new Set([...css.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((m) => m[1]))
}

it('every referenced CSS custom property is defined in theme.css', () => {
  const defined = definedTokens()
  // Sanity: the regex actually found the palette, so an empty set can't pass this vacuously.
  expect(defined.has('--electric')).toBe(true)

  const dangling: string[] = []
  for (const file of sourceFiles(SRC)) {
    const text = readFileSync(file, 'utf8')
    for (const m of text.matchAll(/var\(\s*(--[\w-]+)\s*(?:[,)])/g)) {
      if (!defined.has(m[1])) dangling.push(`${file.replace(SRC, '')} -> ${m[1]}`)
    }
    // EquityChart reads tokens through getComputedStyle by name, not via var().
    for (const m of text.matchAll(/cssColor\(\s*'(--[\w-]+)'/g)) {
      if (!defined.has(m[1])) dangling.push(`${file.replace(SRC, '')} -> ${m[1]} (cssColor)`)
    }
  }
  expect(dangling).toEqual([])
})
