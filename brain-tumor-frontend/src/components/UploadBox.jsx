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
    <div className="space-y-6">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="relative overflow-hidden rounded-[26px] border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900/80 shadow-md"
      >        <label
          htmlFor="file-upload"
          className="relative block cursor-pointer p-8 sm:p-10"
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
            <div className="flex flex-col items-center text-center space-y-5">
              <motion.div
                whileHover={{ scale: 1.08 }}
                className="relative flex h-24 w-24 items-center justify-center rounded-[28px] bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 shadow-lg shadow-blue-500/40"
              >
                <div className="absolute inset-2 rounded-[24px] bg-white/90 dark:bg-gray-900/80" />
                <Upload className="relative w-10 h-10 text-blue-600 dark:text-blue-300" />
              </motion.div>

              <div className="space-y-3">
                <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">
                  Drop MRI image here or click to upload
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-300 max-w-md mx-auto">
                  We support DICOM exports such as JPG and PNG. The Grad-CAM visualization highlights probable tumor regions.
                </p>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-300">
                  <ShieldCheck className="w-4 h-4" />
                  <span className="text-xs font-medium">Secure · On-device preprocessing</span>
                </div>
              </div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative"
            >
              <div className="relative overflow-hidden rounded-3xl border border-white/40 dark:border-gray-700 bg-slate-900/20">
                <img
                  src={previewUrl}
                  alt="Selected MRI preview"
                  className="w-full max-h-[28rem] object-contain mix-blend-plus-lighter"
                />

                <div className="absolute inset-x-0 bottom-0 space-y-2 bg-gradient-to-t from-black/60 via-black/20 to-transparent p-5 text-white">
                  <div className="flex items-center gap-2 text-sm">
                    <ImageIcon className="w-5 h-5" />
                    <span className="font-medium truncate">{selectedFile?.name}</span>
                  </div>
                  <p className="text-xs text-white/80">Resolution is automatically normalized to 224×224px</p>
                </div>

                {loading && (
                  <div className="absolute inset-0 rounded-3xl bg-blue-500/10 backdrop-blur-sm">
                    <div className="scan-overlay" />
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-white">
                      <span className="text-sm uppercase tracking-[0.35em] text-white/70">Scanning</span>
                      <motion.div
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.4, repeat: Infinity }}
                        className="h-2 w-32 rounded-full bg-gradient-to-r from-blue-400 via-white to-blue-400"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="absolute top-4 right-4 flex gap-2">
                <button
                  type="button"
                  onClick={handleRemove}
                  className="rounded-full bg-white/80 dark:bg-gray-900/80 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 shadow-md hover:shadow-lg hover:bg-white dark:hover:bg-gray-900 transition"
                  disabled={loading}
                >
                  <div className="flex items-center gap-1">
                    <X className="w-4 h-4" />
                    Clear
                  </div>
                </button>
              </div>
            </motion.div>
          )}
        </label>

        <div className="relative border-t border-gray-200 dark:border-gray-800 px-6 py-4 text-sm text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-900/60">
          {helperText}
        </div>
      </div>
    </div>
  )
}

export default UploadBox
