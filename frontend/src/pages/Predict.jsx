import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, RotateCcw, ScanSearch, FlaskConical, ShieldAlert } from 'lucide-react'
import { predictTransaction, getSamples, getErrorMessage, getFieldErrors } from '../services/api'
import { PCA_FEATURES, MERCHANT_CATEGORIES, DAY_NAMES, DATASETS, formatCategory } from '../utils/format'
import ErrorMessage from '../components/ErrorMessage'
import Disclaimer from '../components/Disclaimer'

// ---------------------------------------------------------------------------
// Form definitions - validation limits are kept in sync with backend/schemas.py
// ---------------------------------------------------------------------------
const SPARKOV_SECTIONS = [
  {
    title: 'Transaction Details',
    subtitle: 'What was bought, when, and where.',
    fields: [
      { name: 'amount', label: 'Transaction Amount ($)', min: 0.01, max: 1_000_000, step: '0.01' },
      { name: 'category', label: 'Merchant Category', type: 'select',
        options: MERCHANT_CATEGORIES.map((c) => ({ value: c, label: formatCategory(c) })) },
      { name: 'hour', label: 'Hour of Day (0–23)', min: 0, max: 23, integer: true, hint: '22–3 = night' },
      { name: 'day_of_week', label: 'Day of Week', type: 'select',
        options: DAY_NAMES.map((day, index) => ({ value: String(index + 1), label: day })) },
      { name: 'distance_km', label: 'Distance to Merchant (km)', min: 0, max: 20_000, hint: 'From customer home' },
    ],
  },
  {
    title: 'Customer',
    subtitle: 'Basic, non-identifying customer profile.',
    fields: [
      { name: 'gender', label: 'Gender', type: 'select', options: [{ value: 'F', label: 'Female' }, { value: 'M', label: 'Male' }] },
      { name: 'age', label: 'Age (years)', min: 14, max: 110, integer: true },
      { name: 'city_population', label: 'Home City Population', min: 0, max: 50_000_000, integer: true },
    ],
  },
  {
    title: 'Card History',
    subtitle: 'Behaviour of this card before the transaction (computed by Spark window functions in the pipeline).',
    fields: [
      { name: 'txn_count_24h', label: 'Transactions in Last 24 h', min: 0, max: 10_000, integer: true },
      { name: 'amount_last_24h', label: 'Amount Spent in Last 24 h ($)', min: 0, max: 10_000_000, step: '0.01' },
      { name: 'hours_since_last', label: 'Hours Since Previous Transaction', min: 0, max: 100_000 },
      { name: 'card_avg_amount', label: "Card's Usual Amount ($)", min: 0.01, max: 1_000_000, step: '0.01',
        hint: 'Average of earlier transactions' },
    ],
  },
]

const ULB_SECTIONS = [
  {
    title: 'Transaction Information',
    subtitle: 'Basic details of the transaction.',
    wide: true,
    fields: [
      { name: 'time', label: 'Transaction Time (seconds)', min: 0, max: 1_000_000,
        hint: 'Seconds since the first transaction in the dataset (0 – 172,792)' },
      { name: 'amount', label: 'Transaction Amount', min: 0, max: 1_000_000, step: '0.01', hint: 'Must be 0 or more' },
    ],
  },
  {
    title: 'PCA Features (V1 – V28)',
    subtitle: 'Anonymised features produced by PCA to protect cardholder privacy. Typical values lie between about -5 and +5.',
    fields: PCA_FEATURES.map((name) => ({ name, label: name.toUpperCase(), min: -500, max: 500 })),
  },
]

const FORMS = { sparkov: SPARKOV_SECTIONS, ulb: ULB_SECTIONS }
const allFields = (dataset) => FORMS[dataset].flatMap((section) => section.fields)
const emptyForm = (dataset) => Object.fromEntries(allFields(dataset).map((field) => [field.name, '']))

/** Basic frontend validation (the backend validates again with Pydantic). */
function validateForm(dataset, values) {
  const errors = {}
  for (const field of allFields(dataset)) {
    const raw = String(values[field.name]).trim()
    if (raw === '') {
      errors[field.name] = `${field.label.replace(/ \(.*\)$/, '')} is required`
      continue
    }
    if (field.type === 'select') continue
    const number = Number(raw)
    if (!Number.isFinite(number)) errors[field.name] = 'Must be a valid number'
    else if (field.integer && !Number.isInteger(number)) errors[field.name] = 'Must be a whole number'
    else if (number < 0 && field.min >= 0) errors[field.name] = `${field.label.replace(/ \(.*\)$/, '')} cannot be negative`
    else if (number < field.min) errors[field.name] = `Must be ≥ ${field.min}`
    else if (number > field.max) errors[field.name] = `Must be ≤ ${field.max.toLocaleString()}`
  }
  return errors
}

function toPayload(dataset, values) {
  return Object.fromEntries(
    allFields(dataset).map((field) => {
      const raw = values[field.name]
      if (field.name === 'category' || field.name === 'gender') return [field.name, raw]
      return [field.name, Number(raw)]
    }),
  )
}

function FormField({ field, value, error, onChange }) {
  return (
    <div className={`field ${error ? 'has-error' : ''}`}>
      <label htmlFor={field.name}>{field.label}</label>
      {field.type === 'select' ? (
        <select id={field.name} name={field.name} value={value} onChange={onChange} aria-invalid={Boolean(error)}>
          <option value="">Select…</option>
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      ) : (
        <input
          id={field.name}
          name={field.name}
          type="number"
          inputMode="decimal"
          step={field.step || (field.integer ? '1' : 'any')}
          value={value}
          onChange={onChange}
          placeholder="0"
          aria-invalid={Boolean(error)}
        />
      )}
      {error ? <span className="field-error">{error}</span> : field.hint && <span className="field-hint">{field.hint}</span>}
    </div>
  )
}

function Predict() {
  const navigate = useNavigate()
  const [dataset, setDataset] = useState('sparkov')
  const [values, setValues] = useState(() => emptyForm('sparkov'))
  const [errors, setErrors] = useState({})
  const [submitError, setSubmitError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [samples, setSamples] = useState(null)
  const [sampleIndex, setSampleIndex] = useState({})
  const [loadedSampleInfo, setLoadedSampleInfo] = useState('')

  const switchDataset = (next) => {
    if (next === dataset) return
    setDataset(next)
    setValues(emptyForm(next))
    setErrors({})
    setSubmitError('')
    setLoadedSampleInfo('')
  }

  const handleChange = (event) => {
    const { name, value } = event.target
    setValues((current) => ({ ...current, [name]: value }))
    if (errors[name]) setErrors((current) => ({ ...current, [name]: undefined }))
  }

  const handleReset = () => {
    setValues(emptyForm(dataset))
    setErrors({})
    setSubmitError('')
    setLoadedSampleInfo('')
  }

  /** Fill the form with a REAL transaction from the model's held-out test set. */
  const loadSample = async (kind) => {
    setSubmitError('')
    try {
      const data = samples || (await getSamples())
      setSamples(data)
      const list = data[dataset][kind]
      const key = `${dataset}-${kind}`
      const index = (sampleIndex[key] || 0) % list.length
      const sample = list[index]

      setValues(Object.fromEntries(allFields(dataset).map((field) => [field.name, String(sample[field.name])])))
      setErrors({})
      setSampleIndex((current) => ({ ...current, [key]: (current[key] || 0) + 1 }))
      const where = sample.merchant ? ` · ${sample.merchant}, ${sample.city} ${sample.state}` : ''
      setLoadedSampleInfo(
        `Loaded ${kind} sample ${index + 1}/${list.length}: a real test-set transaction (actual label: ${sample.actual_class})${where}.`,
      )
    } catch (error) {
      setSubmitError(getErrorMessage(error))
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setSubmitError('')

    const validationErrors = validateForm(dataset, values)
    setErrors(validationErrors)
    if (Object.keys(validationErrors).length > 0) {
      setSubmitError(`Please fix ${Object.keys(validationErrors).length} invalid field(s) before predicting.`)
      return
    }

    const payload = toPayload(dataset, values)
    setIsSubmitting(true)
    try {
      const result = await predictTransaction(dataset, payload)
      navigate(`/result/${result.transaction_id}`, { state: { result, input: payload } })
    } catch (error) {
      setErrors(getFieldErrors(error))
      setSubmitError(getErrorMessage(error))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Check a Transaction</h1>
          <p>Choose a dataset model, enter the transaction, or load a real sample from the test set.</p>
        </div>
      </div>

      <div className="segmented" role="tablist" aria-label="Dataset">
        {Object.entries(DATASETS).map(([key, info]) => (
          <button
            key={key}
            role="tab"
            aria-selected={dataset === key}
            className={dataset === key ? 'active' : ''}
            onClick={() => switchDataset(key)}
          >
            {info.label}
          </button>
        ))}
      </div>

      <div className="btn-row">
        <button type="button" className="btn btn-secondary btn-sm" onClick={() => loadSample('legitimate')}>
          <FlaskConical size={15} /> Load legitimate sample
        </button>
        <button type="button" className="btn btn-secondary btn-sm" onClick={() => loadSample('fraud')}>
          <ShieldAlert size={15} /> Load suspicious sample
        </button>
      </div>

      {loadedSampleInfo && (
        <div className="alert alert-info">
          <FlaskConical size={18} />
          <div>{loadedSampleInfo}</div>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {FORMS[dataset].map((section) => (
          <section className="card" key={section.title}>
            <div className="card-title">{section.title}</div>
            <div className="card-subtitle">{section.subtitle}</div>
            <div className={`field-grid ${section.wide || dataset === 'sparkov' ? 'wide' : ''}`}>
              {section.fields.map((field) => (
                <FormField
                  key={field.name}
                  field={field}
                  value={values[field.name]}
                  error={errors[field.name]}
                  onChange={handleChange}
                />
              ))}
            </div>
          </section>
        ))}

        <ErrorMessage title="Could not analyze transaction" message={submitError} />

        <div className="form-actions">
          <button type="button" className="btn btn-secondary" onClick={handleReset} disabled={isSubmitting}>
            <RotateCcw size={16} /> Reset
          </button>
          <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
            {isSubmitting ? <Loader2 size={16} className="spinner" /> : <ScanSearch size={16} />}
            {isSubmitting ? 'Analyzing with Spark…' : 'Predict Transaction'}
          </button>
        </div>
      </form>

      <Disclaimer />
    </div>
  )
}

export default Predict
