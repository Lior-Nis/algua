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

it('DEFAULT behaviour is unaffected by the variableHeight opt-out: still fixed across empty -> populated', () => {
  // No `variableHeight` prop passed — this is the guarantee every existing chart consumer
  // (ScatterISOOS, the sparkline, the equity chart) relies on and must never erode silently.
  const { container: emptyContainer } = render(
    <ChartFrame title="gate checks" isEmpty emptyLabel="no checks yet" height={300}>
      <svg />
    </ChartFrame>,
  )
  const emptyFrame = emptyContainer.querySelector('.chart-frame-body') as HTMLElement
  expect(emptyFrame.style.height).toBe('300px')
  cleanup()

  const { container: populatedContainer } = render(
    <ChartFrame title="gate checks" isEmpty={false} emptyLabel="no checks yet" height={300}>
      <svg data-testid="the-chart" />
    </ChartFrame>,
  )
  const populatedFrame = populatedContainer.querySelector('.chart-frame-body') as HTMLElement
  expect(populatedFrame.style.height).toBe('300px')
})

it('variableHeight: the empty state still gets a stable height, but the populated body is left unconstrained', () => {
  const { container: emptyContainer } = render(
    <ChartFrame title="gate checks" isEmpty emptyLabel="no checks yet" height={120} variableHeight>
      <svg />
    </ChartFrame>,
  )
  const emptyFrame = emptyContainer.querySelector('.chart-frame-body') as HTMLElement
  // The honest-empty box still occupies a stable, non-collapsing size.
  expect(emptyFrame.style.height).toBe('120px')
  cleanup()

  const { container: populatedContainer } = render(
    <ChartFrame title="gate checks" isEmpty={false} emptyLabel="no checks yet" height={120} variableHeight>
      <div data-testid="rows">many rows</div>
    </ChartFrame>,
  )
  const populatedFrame = populatedContainer.querySelector('.chart-frame-body') as HTMLElement
  // NOT pinned to the empty-state height — a variable-length list sizes to its own content.
  expect(populatedFrame.style.height).toBe('')
})
