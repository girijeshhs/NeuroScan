import { motion } from 'framer-motion'
import { Brain, Zap, TrendingUp } from 'lucide-react'

const HomePage = ({ onGetStarted }) => {
  const stats = [
    { label: 'Accuracy Rate', value: '96.4%', icon: TrendingUp },
    { label: 'Processing Time', value: '<4s', icon: Zap },
    { label: 'Models Trained', value: '3.2K+', icon: Brain },
  ]

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center pt-20 pb-12 bg-gradient-to-br from-indigo-50 via-white to-blue-50">
      <div className="relative max-w-6xl mx-auto px-6 sm:px-8">
        <div className="grid lg:grid-cols-2 gap-10 items-center">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="space-y-6"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-indigo-100 rounded-full">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-500 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-600"></span>
              </span>
              <span className="text-xs font-bold text-indigo-700 tracking-wide uppercase">
                AI-Powered Medical Diagnosis
              </span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black leading-[1.1] tracking-tight">
              <span className="text-gray-900">Detect Brain Tumors</span>
              <br />
              <span className="bg-gradient-to-r from-indigo-600 via-blue-600 to-purple-600 bg-clip-text text-transparent">
                with Advanced AI
              </span>
            </h1>

            <p className="text-base sm:text-lg text-gray-600 leading-relaxed max-w-xl font-normal">
              Advanced deep learning system for rapid MRI analysis. Upload scans, get instant predictions with explainable Grad-CAM visualization highlighting tumor regions.
            </p>

            <div className="flex flex-wrap gap-3">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={onGetStarted}
                className="px-7 py-3 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-semibold text-sm rounded-lg shadow-md hover:shadow-lg transition-all tracking-normal"
              >
                Start Analysis
              </motion.button>
              <button className="px-7 py-3 border-2 border-gray-300 text-gray-700 font-semibold text-sm rounded-lg hover:border-indigo-500 hover:text-indigo-600 transition-colors tracking-normal">
                Learn More
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 pt-6">
              {stats.map((stat, index) => {
                const Icon = stat.icon
                return (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    className="text-center"
                  >
                    <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-indigo-100 mb-1.5">
                      <Icon className="w-5 h-5 text-indigo-600" />
                    </div>
                    <p className="text-xl font-bold text-gray-900">
                      {stat.value}
                    </p>
                    <p className="text-xs text-gray-600 font-medium">{stat.label}</p>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>

          {/* Right Visual */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="relative"
          >
            <div className="relative w-full aspect-square max-w-md mx-auto">
              {/* Animated rings */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-0 rounded-full border-[3px] border-indigo-200"
              />
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-6 rounded-full border-[3px] border-dashed border-blue-200"
              />
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-12 rounded-full border-[3px] border-purple-200"
              />

              {/* Center brain icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-full blur-2xl opacity-30" />
                  <div className="relative bg-white p-10 rounded-full shadow-xl border-4 border-indigo-100">
                    <Brain className="w-24 h-24 text-indigo-600" />
                  </div>
                </div>
              </div>

              {/* Floating cards */}
              <motion.div
                animate={{ y: [0, -12, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
                className="absolute -top-2 -right-2 bg-white px-3 py-2 rounded-lg shadow-md border-2 border-green-200"
              >
                <p className="text-xs font-semibold text-gray-500">Detection</p>
                <p className="text-base font-bold text-green-600">96.4%</p>
              </motion.div>

              <motion.div
                animate={{ y: [0, 12, 0] }}
                transition={{ duration: 3.5, repeat: Infinity }}
                className="absolute -bottom-2 -left-2 bg-white px-3 py-2 rounded-lg shadow-md border-2 border-blue-200"
              >
                <p className="text-xs font-semibold text-gray-500">Grad-CAM</p>
                <p className="text-base font-bold text-blue-600">Active</p>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

export default HomePage
