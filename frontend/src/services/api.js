/**
 * All communication with the FastAPI backend lives in this ONE file.
 * Components never call axios directly - they import these functions.
 */
import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

// ---------------- API functions ----------------

/** POST /predict/{dataset} - send transaction features, get prediction back ('sparkov' or 'ulb') */
export async function predictTransaction(dataset, transaction) {
  const response = await api.post(`/predict/${dataset}`, transaction)
  return response.data
}

/** GET /transactions - recent prediction history (optional filters: dataset, source) */
export async function getTransactions({ limit = 300, dataset, source } = {}) {
  const response = await api.get('/transactions', { params: { limit, dataset, source } })
  return response.data
}

/** GET /transactions/{id} - one transaction including all input features */
export async function getTransaction(id) {
  const response = await api.get(`/transactions/${id}`)
  return response.data
}

/** GET /dashboard - statistics computed from the SQLite database (optionally for one dataset) */
export async function getDashboard(dataset) {
  const response = await api.get('/dashboard', { params: { dataset: dataset || undefined } })
  return response.data
}

/** DELETE /transactions/{id} */
export async function deleteTransaction(id) {
  const response = await api.delete(`/transactions/${id}`)
  return response.data
}

/** GET /samples - real test-set rows for the "Load sample" buttons */
export async function getSamples() {
  const response = await api.get('/samples')
  return response.data
}

/** GET /model-info - trained model name and metrics */
export async function getModelInfo() {
  const response = await api.get('/model-info')
  return response.data
}

/** GET /health - is the backend up and the model loaded? */
export async function getHealth() {
  const response = await api.get('/health', { timeout: 8000 })
  return response.data
}

// ---------------- Error handling ----------------

/**
 * Convert any axios error into a short, user-friendly message.
 * Handles: backend down, network errors, validation errors,
 * model unavailable (503), database errors (500), not found (404).
 */
export function getErrorMessage(error) {
  if (!error.response) {
    if (error.code === 'ECONNABORTED') {
      return 'The server took too long to respond. Please try again.'
    }
    return 'Cannot reach the backend server. Make sure it is running (cd backend → uvicorn main:app --reload).'
  }

  const { status, data } = error.response
  const detail = typeof data?.detail === 'string' ? data.detail : null

  switch (status) {
    case 404:
      return detail || 'The requested record was not found.'
    case 422:
      return detail || 'Invalid input. Please check the form values.'
    case 503:
      return `Model unavailable: ${detail || 'the ML model is not loaded on the server.'}`
    case 500:
      return detail || 'A server/database error occurred. Please try again.'
    default:
      return detail || `Unexpected error (HTTP ${status}).`
  }
}

/** For 422 responses: map backend field errors to { fieldName: message } */
export function getFieldErrors(error) {
  const errors = error.response?.status === 422 ? error.response.data?.errors : null
  if (!Array.isArray(errors)) return {}
  return Object.fromEntries(errors.map((item) => [item.field, item.message]))
}
