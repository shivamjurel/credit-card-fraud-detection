import { AlertCircle, RotateCcw } from 'lucide-react'

/** Red alert box with an optional "Try again" button. */
function ErrorMessage({ title = 'Something went wrong', message, onRetry }) {
  if (!message) return null

  return (
    <div className="alert alert-error" role="alert">
      <AlertCircle size={20} />
      <div style={{ flex: 1 }}>
        <div className="alert-title">{title}</div>
        <div>{message}</div>
      </div>
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry}>
          <RotateCcw size={14} /> Try again
        </button>
      )}
    </div>
  )
}

export default ErrorMessage
