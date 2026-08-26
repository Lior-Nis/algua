import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import PassMark from './PassMark'

afterEach(cleanup)

it('renders "unknown" (dim) for a genuinely NULL/undefined verdict — never fabricates one', () => {
  render(<PassMark passed={null} />)
  const el = screen.getByText('unknown')
  expect(el.getAttribute('style')).toContain('var(--text-dim)')
})

it('renders "pass" (green) for a truthy verdict, accepting both boolean and 0/1', () => {
  render(<PassMark passed={true} />)
  expect(screen.getByText('pass').getAttribute('style')).toContain('var(--green)')
  cleanup()
  render(<PassMark passed={1} />)
  expect(screen.getByText('pass').getAttribute('style')).toContain('var(--green)')
})

it('renders "fail" (red) for a binding failure', () => {
  render(<PassMark passed={false} />)
  expect(screen.getByText('fail').getAttribute('style')).toContain('var(--red)')
})

it('an advisory failure reads amber, never the red of a breached binding floor', () => {
  render(<PassMark passed={0} advisory />)
  expect(screen.getByText('fail').getAttribute('style')).toContain('var(--amber)')
})

it('an advisory pass reads dim, not the green reserved for a binding pass', () => {
  render(<PassMark passed={1} advisory />)
  expect(screen.getByText('pass').getAttribute('style')).toContain('var(--text-dim)')
})
