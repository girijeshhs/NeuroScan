import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

import Navbar from './components/Navbar.jsx'
import HomePage from './components/HomePage.jsx'
import TumorsPage from './components/TumorsPage.jsx'
import ModelPage from './components/ModelPage.jsx'
import FeatureSection from './components/FeatureSection'
import WhyItMatters from './components/WhyItMatters'
import SiteFooter from './components/SiteFooter'
import UploadBox from './components/UploadBox'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorToast from './components/ErrorToast'

const API_URL = 'http://127.0.0.1:5000/predict'

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [activeSection, setActiveSection] = useState('home')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const uploadSectionRef = useRef(null)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

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
    setActiveSection('analyze')
    setTimeout(() => {
      uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 100)
  }

  const handleNavigate = (section) => {
    setActiveSection(section)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const renderContent = () => {
    switch (activeSection) {
      case 'home':
        return <HomePage onGetStarted={scrollToUpload} />
      case 'tumors':
        return <TumorsPage />
      case 'model':
        return <ModelPage />
      case 'analyze':
        return (
          <>
            {/* Upload Section First */}
            <section ref={uploadSectionRef} className="relative pt-24 pb-8 z-10">
              <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="rounded-3xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950/90 shadow-xl overflow-hidden"
                >
                  <div className="px-6 sm:px-8 pt-8 pb-6">
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                      <div>
                        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white">
                          Upload MRI Scan
                        </h2>
                        <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-300">
                          Drag and drop an axial MRI slice. We preprocess, classify, and generate a Grad-CAM heatmap.
                        </p>
                      </div>
                      <div className="flex items-center gap-2 rounded-full bg-blue-500/10 px-3.5 py-1.5 text-xs font-medium text-blue-600 dark:text-blue-300 whitespace-nowrap">
                        <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                        Flask API · VGG16
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
                        w-full mt-6 py-3.5 rounded-full text-base font-semibold transition-all duration-300
                        ${selectedFile && !loading
                          ? 'bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40'
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
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 30 }}
                      transition={{ duration: 0.4 }}
                      className="mt-8"
                    >
                      <ResultCard result={result} previewUrl={previewUrl} onReset={handleReset} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </section>

            {/* Features Section Below Upload */}
            <FeatureSection />
            <WhyItMatters />
          </>
        )
      default:
        return <HomePage onGetStarted={scrollToUpload} />
    }
  }

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-100 dark:from-gray-950 dark:via-gray-940 dark:to-gray-900 text-gray-900 dark:text-gray-100">
      <div className="pointer-events-none fixed inset-0 opacity-30 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-100/40 to-transparent dark:from-blue-900/20" />
      </div>

      <Navbar
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode((prev) => !prev)}
        onNavigate={handleNavigate}
        activeSection={activeSection}
      />

      <div className="relative">
        {renderContent()}
        <SiteFooter />
      </div>

      <ErrorToast error={error} onClose={() => setError(null)} />
    </div>
  )
}

export default App
