import { createContext, useContext, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { formatTimestamp } from './api'

/** Screens push the envelope's fetched_at into the header slot through this. */
const FetchedAtContext = createContext<(iso: string | null) => void>(() => {})

export function useSetFetchedAt() {
  return useContext(FetchedAtContext)
}

const SOON_TABS = ['Funnel', 'Activity', 'Ideas'] as const

export default function App() {
  const [fetchedAt, setFetchedAt] = useState<string | null>(null)

  return (
    <FetchedAtContext.Provider value={setFetchedAt}>
      <div className="shell">
        <header className="topbar">
          <span className="wordmark">algua monitor</span>
          {fetchedAt !== null && (
            <span className="fetched-at">fetched {formatTimestamp(fetchedAt)}</span>
          )}
        </header>
        <main className="content">
          <Outlet />
        </main>
        <nav className="tabbar">
          <div className="tabbar-inner">
            <NavLink
              to="/"
              end
              className={({ isActive }) => (isActive ? 'tab active' : 'tab')}
            >
              <span>Home</span>
            </NavLink>
            {SOON_TABS.map((label) => (
              <span key={label} className="tab disabled" aria-disabled="true">
                <span>{label}</span>
                <span className="soon-tag">soon</span>
              </span>
            ))}
          </div>
        </nav>
      </div>
    </FetchedAtContext.Provider>
  )
}
