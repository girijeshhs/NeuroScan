import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import { Activity } from 'lucide-react'

import Navbar from './components/Navbar.jsx'
import HomePage from './components/HomePage.jsx'
import TumorsPage from './components/TumorsPage.jsx'
import ModelPage from './components/ModelPage.jsx'
import FeatureSection from './components/FeatureSection'
import SiteFooter from './components/SiteFooter'
import UploadBox from './components/UploadBox'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorToast from './components/ErrorToast'

const API_URL = 'http://127.0.0.1:5000/predict'

function App() {
  const [darkMode] = useState(false) // Force light mode only
  const [activeSection, setActiveSection] = useState('home')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const uploadSectionRef = useRef(null)

  useEffect(() => {
    // Force remove dark mode class
    document.documentElement.classList.remove('dark')
    document.body.classList.remove('dark')
  }, [])

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
            {/* Side-by-Side Upload and Results Layout */}
            <section ref={uploadSectionRef} className="relative pt-16 pb-8 z-10 min-h-screen bg-gradient-to-b from-blue-50/30 to-white">
              <div className="relative max-w-[1400px] mx-auto px-8 sm:px-12">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                  {/* Left Side - Upload Section */}
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                    className="sticky top-16"
                  >
                    <div className="rounded-2xl border-2 border-gray-200 bg-white shadow-lg overflow-hidden">
                      <div className="px-6 pt-6 pb-5">
                        <div className="flex flex-col gap-3 mb-5">
                          <div>
                            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-gray-900">
                              Upload MRI Scan
                            </h2>
                            <p className="mt-1 text-sm text-gray-600 font-normal leading-relaxed">
                              Drag and drop an axial MRI slice. We preprocess, classify, and generate a Grad-CAM heatmap.
                            </p>
                          </div>
                          <div className="flex items-center gap-2 rounded-full bg-indigo-100 px-3 py-1 text-xs font-bold text-indigo-700 w-fit tracking-wide uppercase">
                            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
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
                            w-full mt-5 py-3 rounded-lg text-sm font-semibold tracking-normal transition-all duration-300
                            ${selectedFile && !loading
                              ? 'bg-gradient-to-r from-indigo-600 to-blue-600 text-white shadow-md hover:shadow-lg'
                              : 'bg-gray-100 text-gray-400 cursor-not-allowed border-2 border-gray-200'}
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
                    </div>
                  </motion.div>

                  {/* Right Side - Results Section */}
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <AnimatePresence mode="wait">
                      {result ? (
                        <motion.div
                          key="result-card"
                          initial={{ opacity: 0, scale: 0.96 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.96 }}
                          transition={{ duration: 0.3 }}
                        >
                          <ResultCard result={result} previewUrl={previewUrl} onReset={handleReset} />
                        </motion.div>
                      ) : (
                        <motion.div
                          key="placeholder"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 h-full min-h-[350px] flex items-center justify-center"
                        >
                          <div className="text-center px-6">
                            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-indigo-100 flex items-center justify-center">
                              <Activity className="w-7 h-7 text-indigo-600" />
                            </div>
                            <h3 className="text-base font-bold tracking-tight text-gray-900 mb-2">
                              Results will appear here
                            </h3>
                            <p className="text-sm text-gray-600 font-normal leading-relaxed max-w-xs mx-auto">
                              Upload an MRI scan and click "Analyze MRI" to see the prediction results and Grad-CAM visualization.
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                </div>
              </div>
            </section>

            {/* Features Section Below */}
            <FeatureSection />
          </>
        )
      default:
        return <HomePage onGetStarted={scrollToUpload} />
    }
  }

  return (
    <div className="relative min-h-screen bg-white text-gray-900" style={{ backgroundColor: '#ffffff' }}>
      {/* Subtle background pattern */}
      <div className="pointer-events-none fixed inset-0 opacity-[0.03] z-0">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 2px 2px, rgb(99, 102, 241) 1px, transparent 0)`,
          backgroundSize: '40px 40px'
        }} />
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
