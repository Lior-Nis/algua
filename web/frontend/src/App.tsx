import { createContext, useContext, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { formatTimestamp } from './api'
import Logo from './components/Logo'

/** Screens push the envelope's fetched_at into the header slot through this. */
const FetchedAtContext = createContext<(iso: string | null) => void>(() => {})

export function useSetFetchedAt() {
  return useContext(FetchedAtContext)
}

// One tab per job the dashboard does, plus the fleet detail. Activity folded into Now (the
// audit trail answers "what happened while I was asleep", which is a Now question) and Ideas
// folded into Research. Four is the practical ceiling for a phone tab bar.
const TABS = [
  { to: '/', label: 'Now', end: true },
  { to: '/fleet', label: 'Fleet', end: false },
  { to: '/money', label: 'Money', end: false },
  { to: '/research', label: 'Research', end: false },
]

export default function App() {
  const [fetchedAt, setFetchedAt] = useState<string | null>(null)
  const location = useLocation()
  const navigate = useNavigate()
  const isDetail = location.pathname.startsWith('/s/')

  return (
    <FetchedAtContext.Provider value={setFetchedAt}>
      <div className="shell">
        <header className="topbar">
          {isDetail ? (
            <button
              type="button"
              className="back-btn"
              onClick={() => {
                if (window.history.length > 1) navigate(-1)
                else navigate('/funnel')
              }}
            >
              ‹ back
            </button>
          ) : (
            <span className="brand">
              <Logo />
              <span className="brand-descriptor">monitor</span>
            </span>
          )}
          {fetchedAt !== null && (
            <span className="fetched-at">fetched {formatTimestamp(fetchedAt)}</span>
          )}
        </header>
        <main className="content">
          <Outlet />
        </main>
        <nav className="tabbar">
          <div className="tabbar-inner">
            {TABS.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) => (isActive ? 'tab active' : 'tab')}
              >
                <span>{label}</span>
              </NavLink>
            ))}
          </div>
        </nav>
      </div>
    </FetchedAtContext.Provider>
  )
}
