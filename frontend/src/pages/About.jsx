import { useEffect, useState } from 'react'
import { ArrowDown } from 'lucide-react'
import { getModelInfo } from '../services/api'
import { DATASETS } from '../utils/format'

const FLOW_STEPS = [
  { title: 'Raw Data (2 datasets)', note: '284,807 ULB + 1,852,394 Sparkov transactions (CSV)' },
  { title: 'Spark Ingestion & Cleaning', note: 'Schema, missing values, duplicates, PII removal → Parquet' },
  { title: 'EDA & Statistics', note: 'Spark SQL aggregations, t-test, chi-square test' },
  { title: 'Feature Engineering', note: 'Spark window functions: card history, time, distance, age' },
  { title: '5 Spark MLlib Models', note: 'LR · Decision Tree · Random Forest · GBT · Linear SVM' },
  { title: 'Evaluation & Selection', note: 'Precision, Recall, F1, ROC-AUC, PR-AUC; tuned threshold' },
  { title: 'Serving', note: 'FastAPI website · Structured Streaming · Power BI dashboard' },
]

const TECHNOLOGIES = [
  ['Apache Spark', 'PySpark 3.5'],
  ['Spark MLlib', '5 ML models'],
  ['Spark SQL', 'aggregations'],
  ['Structured Streaming', 'real-time'],
  ['Parquet', 'columnar storage'],
  ['Power BI', 'BI dashboard'],
  ['FastAPI', 'REST API'],
  ['React + Vite', 'web UI'],
  ['Recharts', 'charts'],
  ['SQLite', 'prediction store'],
  ['Pandas / Matplotlib', 'reporting'],
  ['Java 17 + Hadoop winutils', 'Spark runtime'],
]

const SDGS = [
  ['SDG 8', 'Decent Work & Economic Growth', 'Target 8.10 - stronger, safer financial institutions and wider access to trustworthy banking.'],
  ['SDG 9', 'Industry, Innovation & Infrastructure', 'Scalable Big Data + AI infrastructure for the financial sector.'],
  ['SDG 16', 'Peace, Justice & Strong Institutions', 'Target 16.4 - reduce illicit financial flows by detecting fraud early.'],
]

const METRIC_ROWS = [
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['f1_score', 'F1-score'],
  ['roc_auc', 'ROC-AUC'],
  ['pr_auc', 'PR-AUC'],
  ['train_seconds', 'Training time (s)'],
]

function ComparisonTable({ info }) {
  const models = Object.keys(info.models || {})
  if (models.length === 0) return null
  return (
    <div className="table-wrap">
      <table className="metric-table">
        <thead>
          <tr>
            <th>Metric (test set)</th>
            {models.map((name) => (
              <th key={name}>{name}{name === info.selected_model ? ' ★' : ''}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map(([key, label]) => (
            <tr key={key}>
              <td>{label}</td>
              {models.map((name) => {
                const value = info.models[name][key]
                return (
                  <td key={name} style={name === info.selected_model ? { fontWeight: 700 } : undefined}>
                    {key === 'train_seconds' ? value : `${(value * 100).toFixed(1)}%`}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function About() {
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null)) // metrics tables are optional - page still works offline
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>About CreditGuard</h1>
          <p>An end-to-end Big Data analytics pipeline for credit card fraud detection.</p>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="card prose">
          <div className="card-title">What is Credit Card Fraud Detection?</div>
          <p>
            Credit card fraud is the use of someone&apos;s card or card details to make transactions they did not
            authorise. Banks process millions of transactions a day, so fraud must be found automatically and fast.
          </p>
          <p>
            Machine learning learns patterns associated with fraudulent activity from historical, labelled
            transactions (supervised learning) and scores new transactions (binary classification). Because the data
            is large and keeps arriving, the processing is done with <strong>Apache Spark</strong>, which spreads the
            work over many CPU cores - or many machines in a cluster.
          </p>
          <div className="card-title" style={{ marginTop: 18 }}>Datasets</div>
          <ul>
            <li>
              <strong>{DATASETS.ulb.label}</strong> - 284,807 real transactions (Sept 2013), 492 frauds; features
              V1-V28 anonymised with PCA.
            </li>
            <li>
              <strong>{DATASETS.sparkov.label}</strong> - 1,852,394 simulated US transactions (2019-2020), 9,651 frauds;
              readable columns (category, amount, time, location, customer age).
            </li>
          </ul>
        </section>

        <section className="card">
          <div className="card-title">How does the system work?</div>
          <div className="card-subtitle">The Big Data pipeline (spark_pipeline/ folder)</div>
          <div className="flow">
            {FLOW_STEPS.map((step, index) => (
              <div key={step.title} style={{ display: 'contents' }}>
                <div className="flow-step">
                  {step.title}
                  <small>{step.note}</small>
                </div>
                {index < FLOW_STEPS.length - 1 && <ArrowDown size={18} className="flow-arrow" />}
              </div>
            ))}
          </div>
        </section>
      </div>

      {modelInfo &&
        Object.entries(modelInfo.datasets).map(([key, info]) => (
          <section className="card" key={key}>
            <div className="card-title">Model Comparison - {DATASETS[key]?.label}</div>
            <div className="card-subtitle">
              5 PySpark MLlib models. Selected: <strong>{info.selected_model}</strong> (best F1 on the validation set).
              Decision threshold tuned per model on validation data.
            </div>
            <ComparisonTable info={info} />
          </section>
        ))}

      <section className="card">
        <div className="card-title">Sustainable Development Goals (SDGs)</div>
        <div className="card-subtitle">How the project contributes to the UN SDGs</div>
        <div className="grid grid-3">
          {SDGS.map(([code, title, text]) => (
            <div className="detail-item" key={code}>
              <dt>{code}</dt>
              <dd style={{ fontSize: 15 }}>{title}</dd>
              <p className="field-hint" style={{ marginTop: 6 }}>{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="card-title">Technologies</div>
        <div className="card-subtitle">Big Data + ML + BI stack</div>
        <div className="tech-list">
          {TECHNOLOGIES.map(([name, role]) => (
            <span className="tech-pill" key={name}>
              {name}
              <small>{role}</small>
            </span>
          ))}
        </div>
      </section>

      <section className="card prose">
        <div className="card-title">Limitations</div>
        <ul>
          <li>This is an academic demonstration - it must not be used for real financial decision-making.</li>
          <li>The models do not catch every fraud (see recall) and occasionally raise false alarms.</li>
          <li>Dataset 2 is simulated; Dataset 1 is real but only covers 2 days in 2013.</li>
          <li>Risk levels (Medium = half of the fraud threshold) are demonstration settings, not banking rules.</li>
          <li>Spark runs in local mode on one computer; the same code can run on a cluster or Databricks.</li>
        </ul>
      </section>
    </div>
  )
}

export default About
