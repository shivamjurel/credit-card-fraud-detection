import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Predict from './pages/Predict'
import Result from './pages/Result'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import About from './pages/About'

/** Initial theme: saved choice, otherwise the operating-system setting. */
function getInitialTheme() {
  try {
    const saved = localStorage.getItem('creditguard-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* storage unavailable - fall through */
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('creditguard-theme', theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  // Start every new page at the top
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))

  return (
    <div className="app-shell">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="main-area">
        <Navbar onMenuClick={() => setSidebarOpen(true)} theme={theme} onToggleTheme={toggleTheme} />

        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/result" element={<Result />} />
            <Route path="/result/:id" element={<Result />} />
            <Route path="/dashboard" element={<Dashboard theme={theme} />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
