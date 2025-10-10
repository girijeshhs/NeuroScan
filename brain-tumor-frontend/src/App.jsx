import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

import HeroSection from './components/HeroSection'
import FeatureSection from './components/FeatureSection'
import WhyItMatters from './components/WhyItMatters'
import SiteFooter from './components/SiteFooter'
import UploadBox from './components/UploadBox'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorToast from './components/ErrorToast'

const API_URL = 'http://127.0.0.1:5000/predict'
const heroPhrases = [
  'AI-assisted triage support for radiologists',
  'Instant Grad-CAM visualization for every MRI',
  'Clinician-friendly interface for rapid decisions',
]

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [charIndex, setCharIndex] = useState(0)

  const uploadSectionRef = useRef(null)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  useEffect(() => {
    const currentPhrase = heroPhrases[phraseIndex]
    if (charIndex < currentPhrase.length) {
      const timeout = setTimeout(() => setCharIndex((prev) => prev + 1), 55)
      return () => clearTimeout(timeout)
    } else {
      const timeout = setTimeout(() => {
        setCharIndex(0)
        setPhraseIndex((prev) => (prev + 1) % heroPhrases.length)
      }, 1700)
      return () => clearTimeout(timeout)
    }
  }, [charIndex, phraseIndex])

  const displayedPhrase = heroPhrases[phraseIndex].slice(0, charIndex)

  const handleFileSelect = (file) => {
    if (!file) {
      setSelectedFile(null)
      setPreviewUrl(null)
      setResult(null)
      setError(null)
      return
    }

    setSelectedFile(file)
    setResult(null)
    setError(null)

    const reader = new FileReader()
    reader.onloadend = () => {
      setPreviewUrl(reader.result)
    }
    reader.readAsDataURL(file)
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(
        err.response?.data?.error ||
          'Failed to connect to the server. Make sure the Flask backend is running at http://127.0.0.1:5000'
      )
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-100 dark:from-gray-950 dark:via-gray-940 dark:to-gray-900 text-gray-900 dark:text-gray-100">
      <div className="pointer-events-none fixed inset-0 opacity-30 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-100/40 to-transparent dark:from-blue-900/20" />
      </div>

      <div className="relative">
        <HeroSection
          darkMode={darkMode}
          onToggleTheme={() => setDarkMode((prev) => !prev)}
          onGetStarted={scrollToUpload}
          typewriterText={displayedPhrase}
        />

        <FeatureSection />

        <section ref={uploadSectionRef} className="relative py-12 md:py-16 z-10">
          <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.4 }}
              className="rounded-[28px] border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950/90 shadow-xl"
            >
              <div className="px-6 sm:px-9 pt-8 pb-6">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-8">
                  <div>
                    <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white">
                      Upload MRI Scan
                    </h2>
                    <p className="mt-2 text-sm md:text-base text-gray-600 dark:text-gray-300">
                      Drag and drop an axial MRI slice. We preprocess, classify, and generate a Grad-CAM heatmap to highlight
                      potential tumor regions.
                    </p>
                  </div>
                  <div className="flex items-center gap-3 rounded-full bg-blue-500/10 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-300">
                    <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                    Flask API · TensorFlow 2 · VGG16 backbone
                  </div>
                </div>

                <UploadBox
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  previewUrl={previewUrl}
                  loading={loading}
                />

                <motion.button
                  whileHover={{ scale: selectedFile && !loading ? 1.02 : 1 }}
                  whileTap={{ scale: selectedFile && !loading ? 0.98 : 1 }}
                  onClick={handlePredict}
                  disabled={!selectedFile || loading}
                  className={`
                    w-full mt-8 py-4 rounded-full text-lg font-semibold transition-all duration-300
                    ${selectedFile && !loading
                      ? 'bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-xl shadow-blue-500/30 hover:shadow-2xl hover:shadow-blue-500/40'
                      : 'bg-white/55 dark:bg-gray-900/55 text-gray-400 cursor-not-allowed border border-white/60 dark:border-gray-800'}
                  `}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-3">
                      <LoadingSpinner />
                      Running inference...
                    </span>
                  ) : (
                    'Analyze MRI'
                  )}
                </motion.button>
              </div>
            </motion.div>

            <AnimatePresence>
              {result && (
                <motion.div
                  key="result-card"
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 40 }}
                  transition={{ duration: 0.5 }}
                  className="mt-12"
                >
                  <ResultCard result={result} previewUrl={previewUrl} onReset={handleReset} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>

        <WhyItMatters />

        <SiteFooter />
      </div>

      <ErrorToast error={error} onClose={() => setError(null)} />
    </div>
  )
}

export default App
