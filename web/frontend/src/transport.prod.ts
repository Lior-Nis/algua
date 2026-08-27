/** Production stand-in for `src/transport.ts`, swapped in by `vite.config.ts`'s
 * `resolve.alias` whenever `VITE_ALGUA_DEMO !== '1'`.
 *
 * Why this exists: Rollup creates a code-split chunk for a dynamic `import('./fixtures')` at
 * BUILD time, before the `if (DEMO) ...` branch that guards it is dead-code-eliminated. So a
 * production build compiled straight from `transport.ts` still emits a `fixtures-*.js` chunk
 * containing the fixture sentinel — confirmed by running `verify-demo-build.mjs prod dist`,
 * which failed against that build. Aliasing the whole module out for non-demo builds removes
 * `./fixtures` from the production module graph entirely, which tree-shaking alone could not
 * do. Keep this file's exported shape identical to `transport.ts`. */
export const DEMO = false

export async function demoJSON<T>(_url: string): Promise<T> {
  throw new Error('demoJSON is unreachable in a production build (DEMO is always false)')
}
