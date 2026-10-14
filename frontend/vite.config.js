import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Polling makes hot-reload reliable inside OneDrive / network folders,
    // where normal file-change events are sometimes missed.
    watch: { usePolling: true, interval: 300 },
  },
  build: {
    chunkSizeWarningLimit: 1000, // Recharts is large; fine for a local academic project
  },
})
