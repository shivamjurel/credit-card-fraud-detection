import { CheckCircle2, AlertTriangle } from 'lucide-react'
import { formatPercent, predictionTone } from '../utils/format'

/** Large card: "✓ Legitimate Transaction" or "⚠ Potential Fraud Detected". */
function PredictionCard({ prediction, fraudProbability, riskLevel }) {
  const isFraud = prediction === 'Fraud'
  const Icon = isFraud ? AlertTriangle : CheckCircle2

  return (
    <div className={`card prediction-card ${predictionTone(prediction)}`}>
      <div className="prediction-icon">
        <Icon size={34} />
      </div>
      <div className="prediction-title">
        {isFraud ? '⚠ Potential Fraud Detected' : '✓ Legitimate Transaction'}
      </div>
      <p style={{ color: 'var(--text-2)' }}>
        Fraud Score: <strong>{formatPercent(fraudProbability)}</strong>
        {'  ·  '}
        Risk Level: <strong>{riskLevel.toUpperCase()}</strong>
      </p>
    </div>
  )
}

export default PredictionCard
