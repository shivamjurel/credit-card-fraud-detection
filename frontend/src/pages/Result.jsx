import { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { ScanSearch, History as HistoryIcon, SearchX } from 'lucide-react'
import { getTransaction, getErrorMessage } from '../services/api'
import {
  DATASETS, DAY_NAMES, PCA_FEATURES, formatAmount, formatCategory, formatDateTime, formatPercent, predictionTone, riskTone,
} from '../utils/format'
import PredictionCard from '../components/PredictionCard'
import ProbabilityMeter from '../components/ProbabilityMeter'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import Disclaimer from '../components/Disclaimer'

const SPARKOV_FEATURE_LABELS = [
  ['category', 'Merchant category', (v) => formatCategory(v)],
  ['hour', 'Hour of day', (v) => `${v}:00`],
  ['day_of_week', 'Day of week', (v) => DAY_NAMES[Number(v) - 1]],
  ['gender', 'Gender', (v) => (v === 'F' ? 'Female' : 'Male')],
  ['age', 'Age', (v) => `${v} years`],
  ['city_population', 'City population', (v) => Number(v).toLocaleString()],
  ['distance_km', 'Distance to merchant', (v) => `${Number(v).toFixed(1)} km`],
  ['txn_count_24h', 'Transactions in last 24 h', (v) => v],
  ['amount_last_24h', 'Spent in last 24 h', (v) => formatAmount(v, 'sparkov')],
  ['hours_since_last', 'Hours since previous', (v) => Number(v).toFixed(2)],
  ['card_avg_amount', "Card's usual amount", (v) => formatAmount(v, 'sparkov')],
  ['merchant', 'Merchant', (v) => v],
  ['city', 'City', (v) => v],
  ['state', 'State', (v) => v],
]

function SparkovFeatures({ features }) {
  const ratio = features.card_avg_amount ? features.amount / features.card_avg_amount : null
  return (
    <>
      <dl className="detail-list feature-detail-list">
        {SPARKOV_FEATURE_LABELS.filter(([key]) => features[key] !== undefined && features[key] !== null).map(
          ([key, label, format]) => (
            <div className="detail-item" key={key}>
              <dt>{label}</dt>
              <dd>{format(features[key])}</dd>
            </div>
          ),
        )}
      </dl>
      {ratio !== null && (
        <p className="field-hint" style={{ marginTop: 12 }}>
          This amount is <strong>{ratio.toFixed(1)}×</strong> the card&apos;s usual amount - one of the strongest fraud
          signals the model learned.
        </p>
      )}
    </>
  )
}

function UlbFeatures({ features }) {
  return (
    <div className="feature-chips">
      {PCA_FEATURES.map((feature) => (
        <div className="feature-chip" key={feature}>
          <b>{feature.toUpperCase()}</b>
          <span>{Number(features[feature]).toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Shows the analysis of one transaction.
 * - Coming from the Predict page: the result is passed in router state (no extra API call).
 * - Coming from History ("View") or a page refresh: fetched with GET /transactions/{id}.
 */
function Result() {
  const { id } = useParams()
  const { state } = useLocation()
  const passedResult = state?.result && String(state.result.transaction_id) === id ? state : null

  const [record, setRecord] = useState(() =>
    passedResult
      ? { id: passedResult.result.transaction_id, source: 'web', ...passedResult.result, features: passedResult.input }
      : null,
  )
  const [isLoading, setIsLoading] = useState(!passedResult && Boolean(id))
  const [error, setError] = useState('')

  const loadRecord = async () => {
    setIsLoading(true)
    setError('')
    try {
      setRecord(await getTransaction(id))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!passedResult && id) loadRecord()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  if (!id) {
    return (
      <div className="page">
        <div className="card empty-state">
          <SearchX size={40} />
          <h3>No transaction analyzed yet</h3>
          <p>Submit a transaction on the Predict page to see its analysis here.</p>
          <Link to="/predict" className="btn btn-primary">
            <ScanSearch size={16} /> Check a Transaction
          </Link>
        </div>
      </div>
    )
  }

  if (isLoading) return <div className="page"><Loading message="Loading transaction analysis…" /></div>

  if (error) {
    return (
      <div className="page">
        <ErrorMessage title="Could not load transaction" message={error} onRetry={loadRecord} />
        <div>
          <Link to="/history" className="btn btn-secondary">
            <HistoryIcon size={16} /> Back to History
          </Link>
        </div>
      </div>
    )
  }

  if (!record) return null
  const dataset = record.dataset || 'ulb'

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Transaction Analysis</h1>
          <p>
            Transaction #{record.id} · analyzed {formatDateTime(record.created_at)}
            {record.source === 'stream' && ' · received from the real-time stream'}
          </p>
        </div>
        <div className="btn-row">
          <span className="tag">{DATASETS[dataset]?.label}</span>
          {record.model_name && <span className="tag">PySpark {record.model_name}</span>}
        </div>
      </div>

      <PredictionCard
        prediction={record.prediction}
        fraudProbability={record.fraud_probability}
        riskLevel={record.risk_level}
      />

      <div className="grid grid-2">
        <section className="card">
          <div className="card-title">Fraud Score</div>
          <div className="card-subtitle">Output of the Spark ML model, shown against the model&apos;s risk zones.</div>
          <ProbabilityMeter
            probability={record.fraud_probability}
            riskLevel={record.risk_level}
            thresholds={record.risk_thresholds}
          />
        </section>

        <section className="card">
          <div className="card-title">Summary</div>
          <div className="card-subtitle">Key values stored in the database for this prediction.</div>
          <dl className="detail-list">
            <div className="detail-item">
              <dt>Transaction Amount</dt>
              <dd>{formatAmount(record.amount, dataset)}</dd>
            </div>
            <div className="detail-item">
              <dt>Prediction</dt>
              <dd>
                <span className={`badge ${predictionTone(record.prediction)}`}>{record.prediction}</span>
              </dd>
            </div>
            <div className="detail-item">
              <dt>Fraud Score</dt>
              <dd>{formatPercent(record.fraud_probability)}</dd>
            </div>
            <div className="detail-item">
              <dt>Risk Level</dt>
              <dd>
                <span className={`badge ${riskTone(record.risk_level)}`}>{record.risk_level.toUpperCase()}</span>
              </dd>
            </div>
          </dl>
          <p className="field-hint" style={{ marginTop: 14 }}>
            High = at or above the model&apos;s tuned fraud threshold; Medium = at least half of it. These are
            demonstration settings, not real banking rules.
          </p>
        </section>
      </div>

      {record.features && (
        <section className="card">
          <div className="card-title">Input Features</div>
          <div className="card-subtitle">
            {dataset === 'ulb'
              ? `Time: ${record.features.time} s · Amount: ${formatAmount(record.features.amount, 'ulb')}`
              : 'What the model saw for this transaction'}
          </div>
          {dataset === 'ulb' ? <UlbFeatures features={record.features} /> : <SparkovFeatures features={record.features} />}
        </section>
      )}

      <div className="btn-row">
        <Link to="/predict" className="btn btn-primary">
          <ScanSearch size={16} /> Analyze Another Transaction
        </Link>
        <Link to="/history" className="btn btn-secondary">
          <HistoryIcon size={16} /> View History
        </Link>
      </div>

      <Disclaimer />
    </div>
  )
}

export default Result
