import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import ChartFrame from './ChartFrame'

afterEach(cleanup)

it('renders the empty label and no svg/canvas when isEmpty, with no layout shift', () => {
  const { container: emptyContainer } = render(
    <ChartFrame title="equity" isEmpty emptyLabel="awaiting tick history" height={220}>
      <svg />
    </ChartFrame>,
  )
  expect(screen.getByText(/awaiting tick history/i)).toBeTruthy()
  expect(emptyContainer.querySelector('svg')).toBeNull()
  expect(emptyContainer.querySelector('canvas')).toBeNull()

  // Same height whether empty or populated — no layout shift when data arrives.
  const emptyFrame = emptyContainer.querySelector('.chart-frame-body') as HTMLElement | null
  expect(emptyFrame).not.toBeNull()
  const emptyHeight = emptyFrame!.style.height
  expect(emptyHeight).toBe('220px')

  cleanup()

  const { container: populatedContainer } = render(
    <ChartFrame title="equity" isEmpty={false} emptyLabel="awaiting tick history" height={220}>
      <svg data-testid="the-chart" />
    </ChartFrame>,
  )
  expect(screen.getByTestId('the-chart')).toBeTruthy()

  const populatedFrame = populatedContainer.querySelector('.chart-frame-body') as HTMLElement | null
  expect(populatedFrame).not.toBeNull()
  expect(populatedFrame!.style.height).toBe(emptyHeight)
})

it('renders the title', () => {
  render(
    <ChartFrame title="drawdown" isEmpty emptyLabel="no data" height={160}>
      <svg />
    </ChartFrame>,
  )
  expect(screen.getByText('drawdown')).toBeTruthy()
})
