import { useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Upload, Image as ImageIcon, X, ShieldCheck } from 'lucide-react'

const UploadBox = ({ onFileSelect, selectedFile, previewUrl, loading }) => {
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
  }, [])
  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleRemove = () => {
    onFileSelect(null)
  }

  const helperText = useMemo(() => {
    if (selectedFile) {
      return 'Ready to analyze. Click Analyze to generate predictions and Grad-CAM.'
    }
    return 'HIPAA-safe: images stay on your device until you submit for analysis.'
  }, [selectedFile])

  return (
    <div className="space-y-4">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="relative overflow-hidden rounded-lg border border-slate-700/50 bg-[#1e2332]"
      >
        <label
          htmlFor="file-upload"
          className="relative block cursor-pointer p-8"
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept="image/*"
            onChange={handleFileInput}
            disabled={loading}
          />

          {!previewUrl ? (
            <div className="flex flex-col items-center text-center space-y-4">
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="flex h-16 w-16 items-center justify-center rounded-lg bg-slate-800 border border-slate-700"
              >
                <Upload className="w-7 h-7 text-slate-400" />
              </motion.div>

              <div className="space-y-2">
                <h3 className="text-base font-medium text-slate-200">
                  Drop MRI scan or click to browse
                </h3>
                <p className="text-sm text-slate-400 max-w-md mx-auto leading-relaxed">
                  Accepted formats: JPG, PNG (DICOM exports). T1/T2-weighted axial slice images.
                </p>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/60 border border-slate-700/50 text-slate-400">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span className="text-xs font-medium">HIPAA Compliant · Secure Processing</span>
                </div>
              </div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative"
            >
              <div className="relative overflow-hidden rounded-lg border border-slate-700/50 bg-slate-900/50">
                <img
                  src={previewUrl}
                  alt="Selected MRI preview"
                  className="w-full max-h-80 object-contain"
                />

                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-900 via-slate-900/80 to-transparent p-4">
                  <div className="flex items-center gap-2 text-sm text-slate-300">
                    <ImageIcon className="w-4 h-4" />
                    <span className="font-medium truncate">{selectedFile?.name}</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">Normalized to 299×299px for Xception input</p>
                </div>

                {loading && (
                  <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center">
                    <div className="text-center">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                        className="w-8 h-8 border-2 border-slate-600 border-t-slate-300 rounded-full mx-auto mb-3"
                      />
                      <span className="text-xs uppercase tracking-wider text-slate-400">Processing</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="absolute top-3 right-3">
                <button
                  type="button"
                  onClick={handleRemove}
                  className="bg-slate-800/90 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 rounded-md text-xs font-medium text-slate-300 flex items-center gap-1.5 transition-colors"
                  disabled={loading}
                >
                  <X className="w-3.5 h-3.5" />
                  Remove
                </button>
              </div>
            </motion.div>
          )}
        </label>

        <div className="border-t border-slate-700/50 px-5 py-3 text-xs text-slate-400 bg-slate-900/30">
          {helperText}
        </div>
      </div>
    </div>
  )
}

export default UploadBox
