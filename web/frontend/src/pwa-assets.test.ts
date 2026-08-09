import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// The manifest in vite.config.ts references these by absolute URL; they must
// exist in public/ or installs get broken icons.
describe('pwa assets', () => {
  it.each([
    'icon-192.png',
    'icon-512.png',
    'icon-maskable-512.png',
    'apple-touch-icon.png',
    'favicon.svg',
  ])('public/%s exists', (name) => {
    // vitest runs with cwd = web/frontend (where vite.config.ts lives).
    expect(existsSync(join(process.cwd(), 'public', name))).toBe(true)
  })
})
