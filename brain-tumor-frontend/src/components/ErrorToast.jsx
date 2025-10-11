import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, X } from 'lucide-react'

const ErrorToast = ({ error, onClose }) => {
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        onClose()
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [error, onClose])

  return (
    <AnimatePresence>
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          className="fixed bottom-6 right-6 max-w-md z-50"
        >
          <div className="bg-[#1e2332] border border-amber-900/50 rounded-lg shadow-2xl p-4 flex items-start gap-3">
            <div className="bg-amber-900/30 p-1.5 rounded">
              <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            </div>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-amber-200 mb-1">Analysis Error</h4>
              <p className="text-xs text-slate-300 leading-relaxed">{error}</p>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-300 transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default ErrorToast
