import { useCallback, useEffect, useState } from 'react'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LabelList,
} from 'recharts'
import { Activity, AlertTriangle, CheckCircle2, Percent, RefreshCw, Radio, Banknote } from 'lucide-react'
import { getDashboard, getErrorMessage } from '../services/api'
import { DATASETS, formatAmount, formatCategory } from '../utils/format'
import StatCard from '../components/StatCard'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import EmptyState from '../components/EmptyState'

// Status colours (same in both themes): green = good/low, amber = medium, red = fraud/high
const STATUS = { good: '#0ca30c', warning: '#fab219', critical: '#d03b3b' }
const ACCENT = '#2a78d6'

// Axis / grid colours per theme
const CHROME = {
  light: { grid: '#e8e9ec', axis: '#7b8190', surface: '#ffffff' },
  dark: { grid: '#262a31', axis: '#8b919c', surface: '#16191e' },
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const LIVE_REFRESH_MS = 4000

/** "2026-09-29 14:00" -> "29 Sep 14:00", "2026-09-29" -> "29 Sep" */
function formatPeriod(period) {
  const [date, hour] = period.split(' ')
  const [, month, day] = date.split('-')
  return `${Number(day)} ${MONTHS[Number(month) - 1]}${hour ? ` ${hour}` : ''}`
}

function ChartTooltip({ active, payload, label, labelFormatter }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      {label !== undefined && (
        <div className="chart-tooltip-title">{labelFormatter ? labelFormatter(label) : label}</div>
      )}
      {payload.map((item) => (
        <div key={item.name}>
          <span className="legend-swatch" style={{ background: item.color || item.payload.color }} />
          {item.name}: <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  )
}

function Legend({ items }) {
  return (
    <div className="chart-legend">
      {items.map((item) => (
        <span key={item.label}>
          <span className="legend-swatch" style={{ background: item.color }} />
          {item.label}
          {item.value !== undefined && <strong style={{ marginLeft: 6 }}>{item.value}</strong>}
        </span>
      ))}
    </div>
  )
}

function Dashboard({ theme }) {
  const [stats, setStats] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [dataset, setDataset] = useState('')
  const [live, setLive] = useState(false)
  const chrome = CHROME[theme] || CHROME.light

  const loadStats = useCallback(
    async ({ quiet = false } = {}) => {
      if (!quiet) setIsLoading(true)
      setError('')
      try {
        setStats(await getDashboard(dataset))
      } catch (err) {
        setError(getErrorMessage(err))
      } finally {
        if (!quiet) setIsLoading(false)
      }
    },
    [dataset],
  )

  useEffect(() => {
    loadStats()
  }, [loadStats])

  // Live mode: re-fetch every few seconds (useful while the streaming demo runs)
  useEffect(() => {
    if (!live) return undefined
    const intervalId = setInterval(() => loadStats({ quiet: true }), LIVE_REFRESH_MS)
    return () => clearInterval(intervalId)
  }, [live, loadStats])

  const header = (
    <>
      <div className="page-header">
        <div>
          <h1>Analytics Dashboard</h1>
          <p>Statistics calculated from every prediction stored in the SQLite database.</p>
        </div>
        <div className="btn-row">
          <button
            className={`btn btn-sm ${live ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setLive((current) => !current)}
            aria-pressed={live}
            title="Auto-refresh every 4 seconds"
          >
            <Radio size={15} /> {live ? 'Live: on' : 'Live refresh'}
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadStats()} disabled={isLoading}>
            <RefreshCw size={15} className={isLoading ? 'spinner' : ''} /> Refresh
          </button>
        </div>
      </div>
      <div className="segmented" role="tablist" aria-label="Dataset filter">
        {[['', 'All datasets'], ...Object.entries(DATASETS).map(([key, info]) => [key, info.short])].map(([key, label]) => (
          <button key={key || 'all'} className={dataset === key ? 'active' : ''} onClick={() => setDataset(key)}>
            {label}
          </button>
        ))}
      </div>
    </>
  )

  if (isLoading && !stats) return <div className="page">{header}<Loading message="Loading dashboard…" /></div>

  if (error && !stats) {
    return (
      <div className="page">
        {header}
        <ErrorMessage title="Could not load dashboard" message={error} onRetry={() => loadStats()} />
      </div>
    )
  }

  if (!stats) return null

  if (stats.total_transactions === 0) {
    return (
      <div className="page">
        {header}
        <div className="card">
          <EmptyState message="The dashboard fills up as you analyze transactions or run the streaming demo." />
        </div>
      </div>
    )
  }

  const pieData = [
    { name: 'Legitimate', value: stats.legitimate_transactions, color: STATUS.good },
    { name: 'Fraud', value: stats.fraud_transactions, color: STATUS.critical },
  ]

  const riskData = [
    { name: 'Low', value: stats.risk_distribution.Low, color: STATUS.good },
    { name: 'Medium', value: stats.risk_distribution.Medium, color: STATUS.warning },
    { name: 'High', value: stats.risk_distribution.High, color: STATUS.critical },
  ]

  const categoryData = Object.entries(stats.fraud_by_category)
    .map(([category, value]) => ({ name: formatCategory(category), value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)

  const streamed = stats.by_source.stream || 0
  const datasetsPresent = Object.keys(stats.by_dataset)
  const mixedCurrencies = datasetsPresent.length > 1
  const amountDataset = dataset || datasetsPresent[0] || 'sparkov'

  const axisProps = {
    stroke: chrome.axis,
    tick: { fill: chrome.axis, fontSize: 12 },
    tickLine: false,
    axisLine: { stroke: chrome.grid },
  }

  return (
    <div className="page">
      {header}
      <ErrorMessage title="Refresh failed" message={error} />

      <section className="grid grid-4">
        <StatCard label="Total Transactions" value={stats.total_transactions.toLocaleString()} icon={Activity}
          hint={`${(stats.by_source.web || 0).toLocaleString()} web · ${streamed.toLocaleString()} streamed`} />
        <StatCard label="Fraudulent Transactions" value={stats.fraud_transactions.toLocaleString()} icon={AlertTriangle}
          tone="high" hint="Predicted as fraud" />
        <StatCard label="Legitimate Transactions" value={stats.legitimate_transactions.toLocaleString()}
          icon={CheckCircle2} tone="low" hint="Predicted as legitimate" />
        <StatCard label="Fraud Rate" value={`${stats.fraud_percentage}%`} icon={Percent} tone="medium"
          hint={`Avg. fraud score ${(stats.average_fraud_probability * 100).toFixed(1)}%`} />
      </section>

      <section className="grid grid-2">
        <StatCard label="Amount in Flagged Transactions" value={formatAmount(stats.fraud_amount, amountDataset)}
          icon={Banknote} tone="high"
          hint={`${stats.total_amount ? ((stats.fraud_amount / stats.total_amount) * 100).toFixed(1) : 0}% of ${formatAmount(stats.total_amount, amountDataset)} analysed${mixedCurrencies ? ' ($ and € mixed - filter a dataset for exact totals)' : ''}`} />
        <StatCard label="Real-Time Stream" value={streamed.toLocaleString()} icon={Radio}
          hint={streamed ? 'Transactions scored by Spark Structured Streaming' : 'Run the streaming demo to see live data'} />
      </section>

      <section className="grid grid-2">
        <div className="card">
          <div className="card-title">Legitimate vs Fraudulent</div>
          <div className="card-subtitle">Share of all predictions</div>
          <div className="chart-box">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="85%"
                  paddingAngle={pieData.every((d) => d.value > 0) ? 2 : 0} stroke={chrome.surface} strokeWidth={2}
                  isAnimationActive={false}>
                  {pieData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                </Pie>
                <text x="50%" y="47%" textAnchor="middle" fill="currentColor" fontSize={26} fontWeight={700}>
                  {stats.fraud_percentage}%
                </text>
                <text x="50%" y="57%" textAnchor="middle" fill={chrome.axis} fontSize={12}>fraud rate</text>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <Legend items={pieData.map((d) => ({ label: d.name, color: d.color, value: d.value.toLocaleString() }))} />
        </div>

        <div className="card">
          <div className="card-title">Risk Distribution</div>
          <div className="card-subtitle">Number of predictions per demo risk level</div>
          <div className="chart-box">
            <ResponsiveContainer>
              <BarChart data={riskData} margin={{ top: 24, right: 8, left: -8, bottom: 0 }} barCategoryGap="30%">
                <CartesianGrid stroke={chrome.grid} vertical={false} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis allowDecimals={false} {...axisProps} axisLine={false} />
                <Tooltip content={<ChartTooltip />} cursor={{ fill: chrome.grid, opacity: 0.5 }} />
                <Bar dataKey="value" name="Transactions" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {riskData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                  <LabelList dataKey="value" position="top" fill={chrome.axis} fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Legend items={riskData.map((d) => ({ label: d.name, color: d.color }))} />
        </div>
      </section>

      <section className="grid grid-2">
        <div className="card">
          <div className="card-title">Fraud Predictions Over Time</div>
          <div className="card-subtitle">Predictions per {stats.timeline_granularity} (time the prediction was made)</div>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={stats.timeline} margin={{ top: 12, right: 16, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={chrome.grid} vertical={false} />
                <XAxis dataKey="period" tickFormatter={formatPeriod} {...axisProps} minTickGap={24} />
                <YAxis allowDecimals={false} {...axisProps} axisLine={false} />
                <Tooltip content={<ChartTooltip labelFormatter={formatPeriod} />}
                  cursor={{ stroke: chrome.axis, strokeDasharray: '3 3' }} />
                <Line type="linear" dataKey="fraud" name="Fraud" stroke={STATUS.critical} strokeWidth={2}
                  dot={{ r: 4, strokeWidth: 2, fill: chrome.surface }} activeDot={{ r: 6 }} isAnimationActive={false} />
                <Line type="linear" dataKey="legitimate" name="Legitimate" stroke={STATUS.good} strokeWidth={2}
                  dot={{ r: 4, strokeWidth: 2, fill: chrome.surface }} activeDot={{ r: 6 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Legend items={[{ label: 'Fraud', color: STATUS.critical }, { label: 'Legitimate', color: STATUS.good }]} />
        </div>

        <div className="card">
          <div className="card-title">Fraud by Merchant Category</div>
          <div className="card-subtitle">Flagged transactions per category (Dataset 2 only - Dataset 1 has no categories)</div>
          {categoryData.length === 0 ? (
            <div className="empty-state" style={{ padding: 32 }}>No flagged Dataset 2 transactions yet.</div>
          ) : (
            <div className="chart-box">
              <ResponsiveContainer>
                <BarChart data={categoryData} layout="vertical" margin={{ top: 4, right: 32, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke={chrome.grid} horizontal={false} />
                  <XAxis type="number" allowDecimals={false} {...axisProps} />
                  <YAxis type="category" dataKey="name" width={140} {...axisProps} axisLine={false} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: chrome.grid, opacity: 0.5 }} />
                  <Bar dataKey="value" name="Flagged" fill={ACCENT} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                    <LabelList dataKey="value" position="right" fill={chrome.axis} fontSize={12} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

export default Dashboard
