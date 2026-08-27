import { describe, expect, it } from 'vitest'
import * as prod from './transport.prod'
import * as real from './transport'

// `transport.prod.ts` is a build-time ALIAS TARGET (vite.config.ts swaps it in for
// `./transport` in every production build) — nothing else exercises it, and `tsc` never
// typechecks it against the module it replaces (no tsconfig `paths` entry redirects
// `./transport` imports there). This file is the only thing that can catch the two drifting
// apart, so import both explicitly by path rather than relying on the alias.
describe('transport.prod (production alias target)', () => {
  it('DEMO is false — the assertion whose failure would break production', () => {
    expect(prod.DEMO).toBe(false)
  })

  it('demoJSON throws when called, explaining it should be unreachable', async () => {
    await expect(prod.demoJSON('/api/anything')).rejects.toThrow(/unreachable/i)
  })

  it('exports the same surface as transport.ts, so an export added to one and not the ' +
    'other fails here instead of failing a build nobody runs locally', () => {
    expect(Object.keys(prod).sort()).toEqual(Object.keys(real).sort())
  })
})
