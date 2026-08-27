/** Production stand-in for `src/transport.ts`, swapped in by `vite.config.ts`'s
 * `resolve.alias` whenever `VITE_ALGUA_DEMO !== '1'`.
 *
 * Why this exists: Rollup creates a code-split chunk for a dynamic `import('./fixtures')` at
 * BUILD time, before the `if (DEMO) ...` branch that guards it is dead-code-eliminated. So a
 * production build compiled straight from `transport.ts` still emits a `fixtures-*.js` chunk
 * containing the fixture sentinel — confirmed by running `verify-demo-build.mjs prod dist`,
 * which failed against that build. Aliasing the whole module out for non-demo builds removes
 * `./fixtures` from the production module graph entirely, which tree-shaking alone could not
 * do. Keep this file's exported shape identical to `transport.ts`.
 *
 * IMPORTANT — this is a build-time ALIAS TARGET, not an ordinarily-imported module: `tsc -b` /
 * `tsc --noEmit` always resolve `./transport` imports to the real `transport.ts` (there is no
 * `tsconfig` `paths` entry pointing here), so nothing typechecks this file against the module
 * it silently replaces in every production build. `transport.prod.test.ts` — which imports
 * this file directly by path and diffs its export keys against `transport.ts`'s — is the ONLY
 * thing holding the two in sync. If you change either file's exported surface, update the
 * other and re-run that test; the compiler will not catch drift here. */
export const DEMO = false

export async function demoJSON<T>(_url: string): Promise<T> {
  throw new Error('demoJSON is unreachable in a production build (DEMO is always false)')
}
