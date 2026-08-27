/** Demo transport (spec §6). `DEMO` is a build-time constant: Vite statically replaces
 * `import.meta.env.VITE_ALGUA_DEMO`, so in a production build this folds to `false`, the
 * branch in `getJSON` is dead, and the dynamic `import('./fixtures')` below is never reached.
 * The dynamic form matters — it keeps the fixture module out of the main graph entirely
 * rather than relying on tree-shaking to remove a static import.
 *
 * `verify-demo-build.mjs` PROVES the production bundle is fixture-free; do not downgrade
 * that check to a comment. */
export const DEMO: boolean = import.meta.env.VITE_ALGUA_DEMO === '1'

export async function demoJSON<T>(url: string): Promise<T> {
  const { resolveFixture } = await import('./fixtures')
  const payload = resolveFixture(url)
  if (payload === undefined) {
    throw new Error(
      `demo build: no fixture for ${url} — add it to src/fixtures/index.ts. ` +
        'Failing loudly is deliberate: a fabricated empty payload is indistinguishable ' +
        'from a real empty state.',
    )
  }
  return payload as T
}
