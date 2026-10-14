/** Small formatting helpers shared by several pages. */

// Each dataset keeps its original currency: Sparkov = US dollars, ULB = euros
const currencyFormatters = {
  sparkov: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }),
  ulb: new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR' }),
}

export const formatAmount = (value, dataset = 'sparkov') =>
  (currencyFormatters[dataset] || currencyFormatters.sparkov).format(Number(value) || 0)

export const formatPercent = (probability, digits = 1) =>
  `${(Number(probability) * 100).toFixed(digits)}%`

export const formatDate = (isoString) =>
  new Date(isoString).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })

export const formatDateTime = (isoString) =>
  new Date(isoString).toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

/** CSS class suffix for a risk level or prediction: low / medium / high */
export const riskTone = (riskLevel) => (riskLevel || '').toLowerCase()

export const predictionTone = (prediction) => (prediction === 'Fraud' ? 'high' : 'low')

/** Names of the 28 anonymised PCA features: v1 ... v28 */
export const PCA_FEATURES = Array.from({ length: 28 }, (_, index) => `v${index + 1}`)

export const DATASETS = {
  sparkov: { short: 'Card Transactions', label: 'Dataset 2 · Sparkov card transactions' },
  ulb: { short: 'European PCA', label: 'Dataset 1 · ULB European cardholders (PCA)' },
}

export const datasetShort = (dataset) => DATASETS[dataset]?.short || dataset

export const MERCHANT_CATEGORIES = [
  'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness', 'home',
  'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel',
]

/** "shopping_net" -> "Shopping (online)" */
export function formatCategory(category) {
  if (!category) return '—'
  const [main, channel] = category.split('_')
  const names = {
    gas: 'Gas & transport', food: 'Food & dining', health: 'Health & fitness', kids: 'Kids & pets',
    personal: 'Personal care', misc: 'Miscellaneous',
  }
  const base = names[main] || main.charAt(0).toUpperCase() + main.slice(1)
  if (channel === 'net') return `${base} (online)`
  if (channel === 'pos') return `${base} (in store)`
  return base
}

export const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
