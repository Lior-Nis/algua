import '@fontsource/martian-mono/400.css'
import '@fontsource/martian-mono/600.css'
import '@fontsource/martian-mono/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
import '@fontsource/jetbrains-mono/600.css'
import './theme.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { registerSW } from 'virtual:pwa-register'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import App from './App'
import Activity from './screens/Activity'
import Funnel from './screens/Funnel'
import Home from './screens/Home'
import Ideas from './screens/Ideas'
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
          <Route index element={<Home />} />
          <Route path="funnel" element={<Funnel />} />
          <Route path="s/:name" element={<StrategyDetail />} />
          <Route path="activity" element={<Activity />} />
          <Route path="ideas" element={<Ideas />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
