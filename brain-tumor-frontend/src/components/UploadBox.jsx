import { useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

const UploadBox = ({ onFileSelect, selectedFile, previewUrl }) => {
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

  return (
    <div className="space-y-4">
      {!previewUrl ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="border-3 border-dashed border-gray-300 dark:border-gray-600 rounded-2xl p-12 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors duration-300 cursor-pointer bg-gray-50 dark:bg-gray-700/30"
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept="image/*"
            onChange={handleFileInput}
          />
          <label htmlFor="file-upload" className="cursor-pointer block">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="mx-auto w-20 h-20 mb-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center"
            >
              <Upload className="w-10 h-10 text-white" />
            </motion.div>
            <h3 className="text-xl font-semibold text-gray-700 dark:text-gray-200 mb-2">
              Upload MRI Scan
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              Drag and drop your MRI image here, or click to browse
            </p>
            <p className="text-sm text-gray-400 dark:text-gray-500">
              Supported formats: JPG, PNG, JPEG
            </p>
          </label>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative rounded-2xl overflow-hidden bg-gray-100 dark:bg-gray-700"
        >
          <img
            src={previewUrl}
            alt="Preview"
            className="w-full h-96 object-contain"
          />
          <div className="absolute top-4 right-4 flex gap-2">
            <button
              onClick={handleRemove}
              className="bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg transition-colors duration-300"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
            <div className="flex items-center gap-2 text-white">
              <ImageIcon className="w-5 h-5" />
              <span className="text-sm font-medium truncate">{selectedFile?.name}</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default UploadBox
