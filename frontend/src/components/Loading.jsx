import { Loader2 } from 'lucide-react'

function Loading({ message = 'Loading…' }) {
  return (
    <div className="loading" role="status">
      <Loader2 size={28} className="spinner" />
      <span>{message}</span>
    </div>
  )
}

export default Loading
