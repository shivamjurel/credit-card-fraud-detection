import { Link } from 'react-router-dom'
import { Inbox, ScanSearch } from 'lucide-react'

function EmptyState({ title = 'No transactions yet.', message = 'Analyze a transaction to see it here.' }) {
  return (
    <div className="empty-state">
      <Inbox size={40} />
      <h3>{title}</h3>
      <p>{message}</p>
      <Link to="/predict" className="btn btn-primary" style={{ marginTop: 8 }}>
        <ScanSearch size={16} /> Check a Transaction
      </Link>
    </div>
  )
}

export default EmptyState
