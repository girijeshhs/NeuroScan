import { motion } from 'framer-motion'
import { AlertCircle, CheckCircle, RefreshCw, Activity } from 'lucide-react'

const ResultCard = ({ result, previewUrl, onReset }) => {
  const isTumor = result.is_tumor
  const prediction = result.prediction
  const tumorType = result.tumor_type
  const confidence = result.confidence
  const allProbabilities = result.all_probabilities || {}
  const gradcamImage = result.gradcam_image

  return (
    <motion.div
      key="result"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4 }}
      className="bg-white dark:bg-gray-800 rounded-3xl shadow-2xl overflow-hidden"
    >
      {/* Header */}
      <div className={`p-6 ${isTumor ? 'bg-gradient-to-r from-red-500 to-pink-600' : 'bg-gradient-to-r from-green-500 to-emerald-600'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isTumor ? (
              <AlertCircle className="w-8 h-8 text-white" />
            ) : (
              <CheckCircle className="w-8 h-8 text-white" />
            )}
            <div>
              <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
              <p className="text-white/90 text-sm">AI Diagnosis Result</p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.05, rotate: 180 }}
            whileTap={{ scale: 0.95 }}
            onClick={onReset}
            className="bg-white/20 hover:bg-white/30 text-white p-3 rounded-full transition-colors duration-300"
          >
            <RefreshCw className="w-5 h-5" />
          </motion.button>
        </div>
      </div>

      <div className="p-8 space-y-6">
        {/* Main Result */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className={`p-6 rounded-2xl ${isTumor ? 'bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800' : 'bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800'}`}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Diagnosis</p>
              <h3 className={`text-3xl font-bold ${isTumor ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                {prediction}
              </h3>
              {isTumor && tumorType && tumorType !== 'None' && (
                <p className="text-lg font-semibold text-red-700 dark:text-red-300 mt-2">
                  Type: {tumorType}
                </p>
              )}
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Confidence</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-gray-200">{confidence}</p>
            </div>
          </div>
        </motion.div>

        {/* Probabilities */}
        {Object.keys(allProbabilities).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-50 dark:bg-gray-700/50 rounded-2xl p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                Detailed Analysis
              </h4>
            </div>
            <div className="space-y-3">
              {Object.entries(allProbabilities).map(([className, probability], index) => {
                const percentage = (probability * 100).toFixed(2)
                return (
                  <motion.div
                    key={className}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {className}
                      </span>
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {percentage}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ duration: 0.8, delay: 0.5 + index * 0.1 }}
                        className={`h-2.5 rounded-full ${
                          className.includes('No Tumor')
                            ? 'bg-green-500'
                            : 'bg-gradient-to-r from-red-500 to-pink-500'
                        }`}
                      />
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>
        )}

        {/* Images Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Original Image */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="space-y-2"
          >
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Original MRI Scan
            </h4>
            <div className="rounded-xl overflow-hidden border-2 border-gray-200 dark:border-gray-600">
              <img
                src={previewUrl}
                alt="Original MRI"
                className="w-full h-64 object-contain bg-gray-100 dark:bg-gray-700"
              />
            </div>
          </motion.div>

          {/* Grad-CAM Image */}
          {gradcamImage && isTumor && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
              className="space-y-2"
            >
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Grad-CAM Visualization
              </h4>
              <div className="rounded-xl overflow-hidden border-2 border-red-200 dark:border-red-800">
                <img
                  src={`data:image/png;base64,${gradcamImage}`}
                  alt="Grad-CAM"
                  className="w-full h-64 object-contain bg-gray-100 dark:bg-gray-700"
                />
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                Red/yellow areas indicate regions where the AI detected tumor tissue
              </p>
            </motion.div>
          )}
        </div>

        {/* Reset Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onReset}
          className="w-full py-4 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
        >
          Analyze Another Scan
        </motion.button>
      </div>
    </motion.div>
  )
}

export default ResultCard
