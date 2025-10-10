import { motion } from 'framer-motion'
import { Brain, Zap, TrendingUp } from 'lucide-react'

const HomePage = ({ onGetStarted }) => {
  const stats = [
    { label: 'Accuracy Rate', value: '96.4%', icon: TrendingUp },
    { label: 'Processing Time', value: '<4s', icon: Zap },
    { label: 'Models Trained', value: '3.2K+', icon: Brain },
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-16">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="space-y-8"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-full">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
              </span>
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                AI-Powered Medical Diagnosis
              </span>
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black leading-tight">
              <span className="text-gray-900 dark:text-white">Detect Brain</span>
              <br />
              <span className="bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 bg-clip-text text-transparent">
                Tumors with AI
              </span>
            </h1>

            <p className="text-xl text-gray-600 dark:text-gray-300 leading-relaxed max-w-2xl">
              Advanced deep learning system for rapid MRI analysis. Upload scans, get instant predictions with explainable Grad-CAM visualization highlighting tumor regions.
            </p>

            <div className="flex flex-wrap gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onGetStarted}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all"
              >
                Start Analysis
              </motion.button>
              <button className="px-8 py-4 border-2 border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 font-semibold rounded-xl hover:border-blue-500 dark:hover:border-blue-500 transition-colors">
                Learn More
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 pt-8">
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
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/30 mb-2">
                      <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {stat.value}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
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
            <div className="relative w-full aspect-square max-w-lg mx-auto">
              {/* Animated rings */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-0 rounded-full border-4 border-blue-200 dark:border-blue-900/50"
              />
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-8 rounded-full border-4 border-dashed border-indigo-200 dark:border-indigo-900/50"
              />
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-16 rounded-full border-4 border-purple-200 dark:border-purple-900/50"
              />

              {/* Center brain icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full blur-3xl opacity-50" />
                  <div className="relative bg-white dark:bg-gray-900 p-12 rounded-full shadow-2xl">
                    <Brain className="w-32 h-32 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
              </div>

              {/* Floating cards */}
              <motion.div
                animate={{ y: [0, -15, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
                className="absolute -top-4 -right-4 bg-white dark:bg-gray-900 px-4 py-3 rounded-xl shadow-lg border border-gray-200 dark:border-gray-800"
              >
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Detection</p>
                <p className="text-lg font-bold text-green-600">96.4%</p>
              </motion.div>

              <motion.div
                animate={{ y: [0, 15, 0] }}
                transition={{ duration: 3.5, repeat: Infinity }}
                className="absolute -bottom-4 -left-4 bg-white dark:bg-gray-900 px-4 py-3 rounded-xl shadow-lg border border-gray-200 dark:border-gray-800"
              >
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Grad-CAM</p>
                <p className="text-lg font-bold text-blue-600">Active</p>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

export default HomePage
