/** A small card showing one statistic, e.g. "Total Transactions: 150". */
function StatCard({ label, value, hint, icon: Icon, tone = '' }) {
  return (
    <div className="card stat-card">
      {Icon && (
        <div className={`stat-icon ${tone}`}>
          <Icon size={20} />
        </div>
      )}
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {hint && <div className="stat-hint">{hint}</div>}
      </div>
    </div>
  )
}

export default StatCard
