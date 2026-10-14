import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Menu, Moon, Sun } from 'lucide-react'
import { getHealth } from '../services/api'

const PAGE_TITLES = {
  '/': 'Home',
  '/predict': 'Check Transaction',
  '/result': 'Transaction Analysis',
  '/dashboard': 'Dashboard',
  '/history': 'Prediction History',
  '/about': 'About the Project',
}

/** Shows whether the backend is reachable and the ML model is loaded. */
function ApiStatus() {
  const [status, setStatus] = useState({ state: 'checking', text: 'Checking API…' })

  useEffect(() => {
    let cancelled = false

    const check = async () => {
      try {
        const health = await getHealth()
        if (cancelled) return
        setStatus(
          health.model_loaded
            ? { state: 'online', text: 'API online' }
            : { state: 'warning', text: 'Model not loaded' },
        )
      } catch {
        if (!cancelled) setStatus({ state: 'offline', text: 'API offline' })
      }
    }

    check()
    const intervalId = setInterval(check, 15000)
    return () => {
      cancelled = true
      clearInterval(intervalId)
    }
  }, [])

  return (
    <span className="api-status" title={status.text}>
      <span className={`status-dot ${status.state}`} />
      <span className="api-status-text">{status.text}</span>
    </span>
  )
}

function Navbar({ onMenuClick, theme, onToggleTheme }) {
  const { pathname } = useLocation()
  const baseRoute = '/' + (pathname.split('/')[1] || '')
  const title = PAGE_TITLES[baseRoute] || 'CreditGuard'

  return (
    <header className="topbar">
      <button className="icon-button menu-button" onClick={onMenuClick} aria-label="Open menu">
        <Menu size={20} />
      </button>
      <div className="topbar-title">{title}</div>

      <div className="topbar-actions">
        <ApiStatus />
        <button
          className="icon-button"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  )
}

export default Navbar
