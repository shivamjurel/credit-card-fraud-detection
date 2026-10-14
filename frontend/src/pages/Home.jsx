import { Link } from 'react-router-dom'
import { BrainCircuit, Zap, BarChart3, ScanSearch, LayoutDashboard, GraduationCap } from 'lucide-react'
import Disclaimer from '../components/Disclaimer'

const FEATURES = [
  {
    icon: BrainCircuit,
    title: 'Machine Learning',
    text: 'Five Apache Spark MLlib models compared on two datasets (2.1 million transactions). Random Forest selected by F1-score, not accuracy.',
  },
  {
    icon: Zap,
    title: 'Real-Time Prediction',
    text: 'Score a transaction instantly on this website, or watch Spark Structured Streaming score a live stream of transactions.',
  },
  {
    icon: BarChart3,
    title: 'Transaction Analytics',
    text: 'Predictions are stored in SQLite and shown on a live dashboard; the full dataset is exported for a Power BI report.',
  },
]

function Home() {
  return (
    <div className="page">
      <section className="hero">
        <span className="hero-eyebrow">
          <GraduationCap size={15} /> CSET-371 Big Data Analytics Project · Apache Spark
        </span>
        <h1>CreditGuard</h1>
        <h2>AI-Powered Credit Card Fraud Detection</h2>
        <p>
          Analyze transaction patterns and identify potentially fraudulent transactions using machine learning.
        </p>
        <div className="btn-row">
          <Link to="/predict" className="btn btn-primary">
            <ScanSearch size={18} /> Check Transaction
          </Link>
          <Link to="/dashboard" className="btn btn-secondary">
            <LayoutDashboard size={18} /> View Dashboard
          </Link>
        </div>
      </section>

      <section className="grid grid-3">
        {FEATURES.map(({ icon: Icon, title, text }) => (
          <div className="card feature-card" key={title}>
            <div className="stat-icon">
              <Icon size={20} />
            </div>
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </section>

      <Disclaimer />
    </div>
  )
}

export default Home
