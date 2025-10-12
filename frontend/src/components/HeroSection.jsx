import { motion } from 'framer-motion'
import { Brain, Sparkles, Moon, Sun } from 'lucide-react'

const heroVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: 'easeOut' },
  },
}

const floatingVariants = {
  animate: {
    y: [0, -10, 0],
    transition: {
      duration: 6,
      repeat: Infinity,
      ease: 'easeInOut',
    },
  },
}

const HeroSection = ({ darkMode, onToggleTheme, onGetStarted, typewriterText }) => {
  return (
    <section className="relative overflow-hidden z-10">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950" />

      <motion.div
        variants={heroVariants}
        initial="hidden"
        animate="visible"
        className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-20"
      >
        <div className="grid lg:grid-cols-[1.15fr_1fr] gap-10 items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/60 dark:bg-gray-800/80 backdrop-blur-md rounded-full shadow-lg shadow-blue-500/10">
              <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Next-Gen Medical Imaging</span>
            </div>

            <div className="space-y-6">
              <h1 className="text-4xl sm:text-5xl lg:text-[3.4rem] font-black leading-tight tracking-tight text-gray-900 dark:text-white">
                Brain Tumor Detection using <span className="bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-500 bg-clip-text text-transparent">Deep Learning</span>
              </h1>
              <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-2xl">
                Upload MRI scans and visualize tumor regions with AI-powered Grad-CAM for faster, more accurate clinical insights.
              </p>
              <div className="text-base sm:text-lg text-gray-700 dark:text-gray-400 font-medium">
                <span className="text-blue-600 dark:text-blue-400">{typewriterText}</span>
                <span className="animate-pulse">▋</span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                onClick={onGetStarted}
                className="px-8 py-3 rounded-full text-lg font-semibold text-white bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 transition-all"
              >
                Get Started
              </motion.button>

              <button
                onClick={onToggleTheme}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-white/60 dark:border-gray-700 bg-white/70 dark:bg-gray-900/60 backdrop-blur-md text-gray-700 dark:text-gray-200 shadow-sm hover:shadow-md transition"
              >
                {darkMode ? (
                  <Sun className="w-5 h-5 text-yellow-400" />
                ) : (
                  <Moon className="w-5 h-5 text-indigo-500" />
                )}
                <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm text-gray-600 dark:text-gray-300">
              <div className="rounded-2xl bg-white/70 dark:bg-gray-900/60 px-4 py-3 shadow-sm border border-white/60 dark:border-gray-800">
                <p className="text-xs uppercase tracking-wide text-blue-500 font-semibold">Dataset</p>
                <p className="mt-1 text-lg font-semibold text-gray-900 dark:text-white">3.2k+ MRI scans</p>
              </div>
              <div className="rounded-2xl bg-white/70 dark:bg-gray-900/60 px-4 py-3 shadow-sm border border-white/60 dark:border-gray-800">
                <p className="text-xs uppercase tracking-wide text-blue-500 font-semibold">Latency</p>
                <p className="mt-1 text-lg font-semibold text-gray-900 dark:text-white">&lt; 4s AVG</p>
              </div>
              <div className="rounded-2xl bg-white/70 dark:bg-gray-900/60 px-4 py-3 shadow-sm border border-white/60 dark:border-gray-800 hidden sm:block">
                <p className="text-xs uppercase tracking-wide text-blue-500 font-semibold">Explainability</p>
                <p className="mt-1 text-lg font-semibold text-gray-900 dark:text-white">Grad-CAM heatmaps</p>
              </div>
            </div>
          </div>

          <motion.div
            variants={floatingVariants}
            animate="animate"
            className="relative hidden lg:flex items-center justify-center"
          >
            <div className="relative h-80 w-80">
              <div className="absolute inset-6 rounded-[2.5rem] bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-lg" />
              <div className="absolute inset-12 rounded-full bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/40 dark:to-gray-900 flex items-center justify-center">
                <Brain className="w-32 h-32 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="absolute -left-6 top-1/2 -translate-y-1/2 px-4 py-3 rounded-2xl bg-white dark:bg-gray-900 shadow-md border border-gray-200 dark:border-gray-800">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">Grad-CAM Insights</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Highlighting critical regions</p>
              </div>
              <div className="absolute -right-6 top-1/2 -translate-y-1/2 px-4 py-3 rounded-2xl bg-white dark:bg-gray-900 shadow-md border border-gray-200 dark:border-gray-800">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">96% Accuracy</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Xception fine-tuned model</p>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}

export default HeroSection
