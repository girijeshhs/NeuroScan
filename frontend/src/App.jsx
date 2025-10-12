import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import { Activity } from 'lucide-react'

import Navbar from './components/Navbar.jsx'
import HomePage from './components/HomePage.jsx'
import TumorsPage from './components/TumorsPage.jsx'
import ModelPage from './components/ModelPage.jsx'
import SiteFooter from './components/SiteFooter'
import UploadBox from './components/UploadBox'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorToast from './components/ErrorToast'

const API_URL = 'http://127.0.0.1:5000/predict'

function App() {
  const [darkMode] = useState(true) // Professional dark mode
  const [activeSection, setActiveSection] = useState('home')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])

  const uploadSectionRef = useRef(null)

  useEffect(() => {
    // Enable professional dark mode
    document.documentElement.classList.add('dark')
    document.body.classList.add('dark')
    
    // Load analysis history from localStorage
    const savedHistory = localStorage.getItem('analysisHistory')
    if (savedHistory) {
      try {
        setAnalysisHistory(JSON.parse(savedHistory))
      } catch (e) {
        console.error('Failed to parse history:', e)
      }
    }
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
      
      // Save to history
      const newHistoryItem = {
        id: `MRI-${Math.floor(1000 + Math.random() * 9000)}`,
        type: response.data.prediction,
        confidence: response.data.confidence,
        timestamp: new Date().toISOString(),
        date: new Date().toLocaleString('en-US', { 
          month: 'short', 
          day: 'numeric', 
          hour: '2-digit', 
          minute: '2-digit' 
        })
      }
      
      const updatedHistory = [newHistoryItem, ...analysisHistory].slice(0, 20) // Keep last 20
      setAnalysisHistory(updatedHistory)
      localStorage.setItem('analysisHistory', JSON.stringify(updatedHistory))
      
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

  const handleClearHistory = () => {
    setAnalysisHistory([])
    localStorage.removeItem('analysisHistory')
  }

  const renderContent = () => {
    switch (activeSection) {
      case 'home':
        return <HomePage 
          onGetStarted={scrollToUpload} 
          onNavigate={handleNavigate}
          analysisHistory={analysisHistory}
          onClearHistory={handleClearHistory}
        />
      case 'tumors':
        return <TumorsPage />
      case 'model':
        return <ModelPage />
      case 'analyze':
        return (
          <>
            {/* Side-by-Side Upload and Results Layout */}
            <section ref={uploadSectionRef} className="relative pt-20 pb-12 z-10 min-h-screen bg-[#0f1419]">
              <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                  {/* Left Side - Upload Section */}
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                    className="sticky top-20"
                  >
                    <div className="medical-card rounded-lg overflow-hidden">
                      <div className="px-6 pt-6 pb-5">
                        <div className="flex flex-col gap-3 mb-5">
                          <div className="space-y-2">
                            <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-slate-100">
                              Upload MRI Scan
                            </h2>
                            <p className="mt-2 text-sm text-slate-400 font-normal leading-relaxed">
                              Submit axial T1/T2-weighted MRI slice for automated classification and visual explanation via Grad-CAM heatmap generation.
                            </p>
                          </div>
                          <div className="flex items-center gap-2 rounded-md bg-burgundy/10 border border-burgundy/20 px-3 py-1.5 text-xs font-semibold text-slate-300 w-fit tracking-wide uppercase">
                            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                            VGG16 Neural Network Active
                          </div>
                        </div>

                        <UploadBox
                          onFileSelect={handleFileSelect}
                          selectedFile={selectedFile}
                          previewUrl={previewUrl}
                          loading={loading}
                        />

                        <motion.button
                          whileHover={{ scale: selectedFile && !loading ? 1.01 : 1 }}
                          whileTap={{ scale: selectedFile && !loading ? 0.99 : 1 }}
                          onClick={handlePredict}
                          disabled={!selectedFile || loading}
                          className={`
                            w-full mt-5 py-3 rounded-md text-sm font-semibold tracking-normal transition-all duration-200
                            ${selectedFile && !loading
                              ? 'btn-primary text-white'
                              : 'bg-slate-800/50 text-slate-600 cursor-not-allowed border border-slate-700/50'}
                          `}
                        >
                          {loading ? (
                            <span className="flex items-center justify-center gap-3">
                              <LoadingSpinner />
                              Processing Neural Analysis...
                            </span>
                          ) : (
                            'Begin Diagnostic Analysis'
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
                          className="medical-card rounded-lg border border-dashed border-slate-700 h-full min-h-[350px] flex items-center justify-center"
                        >
                          <div className="text-center px-6">
                            <div className="w-16 h-16 mx-auto mb-4 rounded-md bg-slate-800 flex items-center justify-center">
                              <Activity className="w-8 h-8 text-slate-500" />
                            </div>
                            <h3 className="text-base font-semibold tracking-tight text-slate-200 mb-2">
                              Diagnostic Results
                            </h3>
                            <p className="text-sm text-slate-400 font-normal leading-relaxed max-w-xs mx-auto">
                              Analysis results, classification confidence, and Grad-CAM visualization will be displayed here upon completion.
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                </div>
              </div>
            </section>
          </>
        )
      default:
        return <HomePage 
          onGetStarted={scrollToUpload} 
          onNavigate={handleNavigate}
          analysisHistory={analysisHistory}
          onClearHistory={handleClearHistory}
        />
    }
  }

  return (
    <div className="relative min-h-screen bg-[#0f1419] text-slate-100" style={{ backgroundColor: '#0f1419' }}>
      {/* Subtle texture pattern */}
      <div className="bg-texture pointer-events-none fixed inset-0 opacity-50 z-0" />

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
