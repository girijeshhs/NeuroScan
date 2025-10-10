import { motion } from 'framer-motion'
import { Cpu, Activity, GaugeCircle, Zap } from 'lucide-react'

const featureIcons = {
  AI: Cpu,
  GradCAM: Activity,
  Accuracy: GaugeCircle,
  Instant: Zap,
}

const cardVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: (index) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: 0.2 + index * 0.1,
      duration: 0.5,
      ease: 'easeOut',
    },
  }),
}

const features = [
  {
    key: 'AI',
    title: 'AI-Powered Analysis',
    description: 'VGG16-based convolutional neural network trained on curated MRI datasets for reliable tumor detection.',
  },
  {
    key: 'GradCAM',
    title: 'Grad-CAM Visualization',
    description: 'Explainable AI overlays highlight critical regions to support clinician decision-making.',
  },
  {
    key: 'Accuracy',
    title: 'High Accuracy (96%)',
    description: 'Rigorous evaluation with cross-validation to ensure consistent predictions in production.',
  },
  {
    key: 'Instant',
    title: 'Instant Results',
    description: 'Optimized pipeline delivers predictions and heatmaps in just a few seconds.',
  },
]

const FeatureSection = () => {
  return (
    <section className="relative py-16 md:py-20 z-10 bg-gray-50/50 dark:bg-gray-900/30">
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="text-center max-w-3xl mx-auto mb-12"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white">
            Engineered for Clinical Workflows
          </h2>
          <p className="mt-4 text-base md:text-lg lg:text-xl text-gray-600 dark:text-gray-300">
            Seamlessly integrate AI-assisted diagnostics with intuitive visualization and fast inference.
          </p>
        </motion.div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => {
            const Icon = featureIcons[feature.key]
            return (
              <motion.div
                key={feature.title}
                custom={index}
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, amount: 0.3 }}
                className="group relative overflow-hidden rounded-3xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/80 shadow-lg hover:shadow-xl transition-all duration-300"
              >
                <div className="relative p-8 space-y-4">
                  <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 text-blue-600 dark:text-blue-400">
                    <Icon className="w-8 h-8" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                    {feature.title}
                  </h3>
                  <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Additional Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="mt-16 bg-gradient-to-br from-blue-500/10 via-indigo-500/10 to-purple-500/10 rounded-3xl p-8 md:p-12 border border-blue-200 dark:border-blue-800/30"
        >
          <h3 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Why Early Detection Matters
          </h3>
          <p className="text-base md:text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
            Brain tumors can progress rapidly, and subtle abnormalities are often difficult to detect in early stages. By combining deep learning diagnostics with explainable Grad-CAM visualization, clinicians receive actionable insights in seconds. This assists in triaging urgent cases, validating treatment plans, and improving patient outcomes through timely intervention.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

export default FeatureSection
