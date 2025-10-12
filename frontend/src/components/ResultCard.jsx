import { motion } from 'framer-motion'
import { CheckCircle2, RefreshCw, Activity, Clock, FileText } from 'lucide-react'

const ResultCard = ({ result, previewUrl, onReset }) => {
  const isTumor = result.is_tumor
  const prediction = result.prediction
  const tumorType = result.tumor_type
  const confidence = result.confidence
  const allProbabilities = result.all_probabilities || {}
  const gradcamImage = result.gradcam_image

  // Generate timestamp
  const timestamp = new Date().toLocaleString('en-GB', { 
    day: '2-digit', 
    month: 'short', 
    year: 'numeric',
    hour: '2-digit', 
    minute: '2-digit' 
  })

  // Generate scan ID
  const scanId = `MRI-${Math.floor(1000 + Math.random() * 9000)}`

  return (
    <motion.div
      key="result"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="medical-card rounded-lg overflow-hidden"
    >
      {/* Elegant Success Header */}
      <div className="bg-[#1e2332] border-b border-[#2d5f4c]/30 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-[#4a6b5a]" />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Completed</span>
            </div>
            <div className="w-px h-4 bg-slate-700" />
            <h2 className="text-lg font-semibold text-slate-200">
              Diagnostic Analysis
            </h2>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onReset}
            className="text-slate-400 hover:text-slate-200 p-2 rounded-md hover:bg-slate-800/50 transition-colors"
            title="New Analysis"
          >
            <RefreshCw className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Metadata Bar */}
        <div className="flex items-center gap-6 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <FileText className="w-3.5 h-3.5" />
            <span className="font-mono">{scanId}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-3.5 h-3.5" />
            <span>{timestamp}</span>
          </div>
        </div>

        {/* Tumor Classification - Simple Label Only */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-[#1e2332] rounded-lg p-5 border border-slate-700/50"
        >
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Tumor Classification
          </p>
          <p className="text-xl font-semibold text-slate-100">
            {isTumor && tumorType && tumorType !== 'None' ? tumorType : prediction}
          </p>
        </motion.div>

        {/* MRI Visualizations - MOVED UP */}
        <div className="grid md:grid-cols-2 gap-5">
          {/* Original Image */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-3"
          >
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Original MRI Scan
            </h4>
            <div className="rounded-lg overflow-hidden border border-slate-700/50 bg-[#1e2332]">
              <img
                src={previewUrl}
                alt="Original MRI"
                className="w-full h-64 object-contain bg-slate-900/50"
              />
            </div>
          </motion.div>

          {/* Grad-CAM Image */}
          {gradcamImage && isTumor && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="space-y-3"
            >
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Grad-CAM Heatmap
              </h4>
              <div className="rounded-lg overflow-hidden border border-slate-700/50 bg-[#1e2332]">
                <img
                  src={`data:image/png;base64,${gradcamImage}`}
                  alt="Grad-CAM"
                  className="w-full h-64 object-contain bg-slate-900/50"
                />
              </div>
              <p className="text-xs text-slate-500 italic leading-relaxed">
                Highlighted regions indicate areas of diagnostic significance identified by the neural network.
              </p>
            </motion.div>
          )}
        </div>

        {/* Classification Probabilities - MOVED BELOW IMAGES */}
        {Object.keys(allProbabilities).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-[#1e2332] rounded-lg p-6 border border-slate-700/50"
          >
            <div className="flex items-center gap-2 mb-5">
              <Activity className="w-4 h-4 text-slate-400" />
              <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                Classification Probabilities
              </h4>
            </div>
            <div className="space-y-4">
              {Object.entries(allProbabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, probability], index) => {
                  const percentage = (probability * 100).toFixed(1)
                  const isHighest = index === 0
                  const barColor = className.includes('No Tumor') 
                    ? '#4a6b5a' 
                    : isHighest 
                      ? '#8b4a5c' 
                      : '#4a4a5c'
                  
                  return (
                    <motion.div
                      key={className}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.35 + index * 0.05 }}
                      className="group"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className={`text-sm font-medium ${isHighest ? 'text-slate-200' : 'text-slate-400'}`}>
                          {className}
                        </span>
                        <span className={`text-sm font-semibold font-mono ${isHighest ? 'text-slate-100' : 'text-slate-400'}`}>
                          {percentage}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percentage}%` }}
                          transition={{ duration: 0.6, delay: 0.4 + index * 0.05, ease: "easeOut" }}
                          className="h-1.5 rounded-full transition-all duration-200 group-hover:opacity-80"
                          style={{ backgroundColor: barColor }}
                        />
                      </div>
                    </motion.div>
                  )
                })}
            </div>
          </motion.div>
        )}

        {/* Action Button */}
        <div className="pt-2">
          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={onReset}
            className="btn-secondary w-full py-3 rounded-md text-slate-200 font-semibold text-sm flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Analyze New Scan
          </motion.button>
        </div>
      </div>
    </motion.div>
  )
}

export default ResultCard
