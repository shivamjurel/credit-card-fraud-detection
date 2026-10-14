import { formatPercent, riskTone } from '../utils/format'

/**
 * Horizontal bar showing the model's fraud score with Low / Medium / High zones.
 * The zone boundaries come from the backend: each Spark model has its own tuned
 * threshold (High) and Medium starts at half of it (demo setting).
 */
function ProbabilityMeter({ probability, riskLevel, thresholds }) {
  const low = thresholds?.medium ?? 0.3
  const high = thresholds?.high ?? 0.7
  const percent = Math.min(Math.max(probability * 100, 0), 100)

  return (
    <div className="probability-meter">
      <div
        className="meter-track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
        aria-label="Fraud score"
      >
        <div className={`meter-fill ${riskTone(riskLevel)}`} style={{ width: `${Math.max(percent, 1.5)}%` }} />
        <div className="meter-marker" style={{ left: `${low * 100}%` }} />
        <div className="meter-marker" style={{ left: `${high * 100}%` }} />
      </div>
      <div className="meter-zones">
        <span style={{ left: `${(low * 100) / 2}%` }}>Low</span>
        <span style={{ left: `${((low + high) * 100) / 2}%` }}>Medium</span>
        <span style={{ left: `${Math.min(((high + 1) * 100) / 2, 94)}%` }}>High</span>
      </div>
      <div style={{ textAlign: 'center', marginTop: 8 }}>
        <div className="big-number">{formatPercent(probability)}</div>
        <div style={{ color: 'var(--muted)', fontSize: 13 }}>
          fraud score · flagged as fraud at ≥ {formatPercent(high)}
        </div>
      </div>
    </div>
  )
}

export default ProbabilityMeter
