import { Info } from 'lucide-react'

/** Required academic notice - shown on Home, Predict and Result pages. */
function Disclaimer() {
  return (
    <div className="alert alert-info disclaimer">
      <Info size={18} />
      <div>
        This application is an academic machine-learning demonstration and should not be used for real
        financial decision-making. It never asks for card numbers, CVV, PIN, OTP, expiry dates or bank
        credentials - only anonymised transaction features.
      </div>
    </div>
  )
}

export default Disclaimer
