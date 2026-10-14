import { NavLink } from 'react-router-dom'
import { Home, ScanSearch, LayoutDashboard, History, Info, ShieldCheck } from 'lucide-react'

export const NAV_ITEMS = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/predict', label: 'Predict', icon: ScanSearch },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/history', label: 'History', icon: History },
  { to: '/about', label: 'About', icon: Info },
]

function Sidebar({ isOpen, onClose }) {
  return (
    <>
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <NavLink to="/" className="brand" onClick={onClose}>
          <span className="brand-mark">
            <ShieldCheck size={20} />
          </span>
          <span>
            <div className="brand-name">CreditGuard</div>
            <div className="brand-tag">Fraud Detection · ML Demo</div>
          </span>
        </NavLink>

        <nav className="nav-list">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={onClose}
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          Academic ML demonstration.
          <br />
          Not for real financial decisions.
        </div>
      </aside>
      <div className={`sidebar-backdrop ${isOpen ? 'open' : ''}`} onClick={onClose} />
    </>
  )
}

export default Sidebar
