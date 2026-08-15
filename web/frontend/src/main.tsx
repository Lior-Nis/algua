// Brand typography (docs/brand/README.md): Inter for product language, IBM Plex
// Mono for code and data. Weights limited to the approved set.
import '@fontsource/inter/300.css'
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import './theme.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { registerSW } from 'virtual:pwa-register'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import App from './App'
import Fleet from './screens/Fleet'
import Money from './screens/Money'
import Now from './screens/Now'
import Research from './screens/Research'
import StrategyDetail from './screens/StrategyDetail'

// PROD only — a dev service worker would fight the Vite proxy. With registerType
// 'autoUpdate' the virtual module owns activation AND the post-update reload; a second
// manual controllerchange listener would be a duplicate reload path.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  registerSW({ immediate: true })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Now />} />
          <Route path="fleet" element={<Fleet />} />
          <Route path="money" element={<Money />} />
          <Route path="research" element={<Research />} />
          <Route path="s/:name" element={<StrategyDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
