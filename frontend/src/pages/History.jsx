import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Trash2, RefreshCw, Loader2, Radio } from 'lucide-react'
import { getTransactions, deleteTransaction, getErrorMessage } from '../services/api'
import {
  DATASETS, datasetShort, formatAmount, formatCategory, formatDateTime, formatPercent, predictionTone, riskTone,
} from '../utils/format'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import EmptyState from '../components/EmptyState'

const PREDICTION_FILTERS = ['All', 'Fraud', 'Legitimate']

function History() {
  const navigate = useNavigate()
  const [transactions, setTransactions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [deletingId, setDeletingId] = useState(null)
  const [predictionFilter, setPredictionFilter] = useState('All')
  const [datasetFilter, setDatasetFilter] = useState('')

  const loadTransactions = async (dataset = datasetFilter) => {
    setIsLoading(true)
    setError('')
    try {
      setTransactions(await getTransactions({ dataset: dataset || undefined }))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadTransactions()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const changeDataset = (dataset) => {
    setDatasetFilter(dataset)
    loadTransactions(dataset)
  }

  const handleDelete = async (id) => {
    if (!window.confirm(`Delete transaction #${id} from history?`)) return
    setDeletingId(id)
    setError('')
    try {
      await deleteTransaction(id)
      setTransactions((current) => current.filter((transaction) => transaction.id !== id))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setDeletingId(null)
    }
  }

  const visibleTransactions =
    predictionFilter === 'All'
      ? transactions
      : transactions.filter((transaction) => transaction.prediction === predictionFilter)

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Prediction History</h1>
          <p>Every transaction scored by the Spark models (website + real-time stream), stored in SQLite.</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => loadTransactions()} disabled={isLoading}>
          <RefreshCw size={15} className={isLoading ? 'spinner' : ''} /> Refresh
        </button>
      </div>

      <div className="btn-row">
        <div className="segmented" role="tablist" aria-label="Dataset filter">
          {[['', 'All datasets'], ...Object.entries(DATASETS).map(([key, info]) => [key, info.short])].map(
            ([key, label]) => (
              <button key={key || 'all'} className={datasetFilter === key ? 'active' : ''} onClick={() => changeDataset(key)}>
                {label}
              </button>
            ),
          )}
        </div>
        <div className="segmented" role="tablist" aria-label="Prediction filter">
          {PREDICTION_FILTERS.map((option) => (
            <button key={option} className={predictionFilter === option ? 'active' : ''} onClick={() => setPredictionFilter(option)}>
              {option}
            </button>
          ))}
        </div>
      </div>

      <ErrorMessage
        title="Could not load history"
        message={error}
        onRetry={transactions.length === 0 ? () => loadTransactions() : undefined}
      />

      {isLoading && transactions.length === 0 ? (
        <Loading message="Loading transactions…" />
      ) : (
        !error && (
          <div className="card" style={{ padding: 0 }}>
            {transactions.length === 0 ? (
              <EmptyState />
            ) : visibleTransactions.length === 0 ? (
              <div className="empty-state">No {predictionFilter.toLowerCase()} transactions in this view.</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Dataset</th>
                      <th>Amount</th>
                      <th>Category</th>
                      <th>Prediction</th>
                      <th>Fraud Score</th>
                      <th>Risk</th>
                      <th>Date</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTransactions.map((transaction) => (
                      <tr key={transaction.id}>
                        <td>
                          #{transaction.id}
                          {transaction.source === 'stream' && (
                            <span title="Received from the real-time stream" style={{ marginLeft: 6, color: 'var(--accent)' }}>
                              <Radio size={13} style={{ verticalAlign: -2 }} />
                            </span>
                          )}
                        </td>
                        <td><span className="tag">{datasetShort(transaction.dataset)}</span></td>
                        <td>{formatAmount(transaction.amount, transaction.dataset)}</td>
                        <td>{transaction.category ? formatCategory(transaction.category) : '—'}</td>
                        <td>
                          <span className={`badge ${predictionTone(transaction.prediction)}`}>{transaction.prediction}</span>
                        </td>
                        <td>{formatPercent(transaction.fraud_probability, 0)}</td>
                        <td>
                          <span className={`badge ${riskTone(transaction.risk_level)}`}>{transaction.risk_level}</span>
                        </td>
                        <td>{formatDateTime(transaction.created_at)}</td>
                        <td>
                          <div className="table-actions">
                            <button className="btn btn-secondary btn-sm" onClick={() => navigate(`/result/${transaction.id}`)}>
                              <Eye size={14} /> View
                            </button>
                            <button
                              className="btn btn-danger-ghost btn-sm"
                              onClick={() => handleDelete(transaction.id)}
                              disabled={deletingId === transaction.id}
                            >
                              {deletingId === transaction.id ? <Loader2 size={14} className="spinner" /> : <Trash2 size={14} />}
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )
      )}
      <p className="field-hint">
        <Radio size={12} style={{ verticalAlign: -1 }} /> = scored by the Spark Structured Streaming job.
      </p>
    </div>
  )
}

export default History
