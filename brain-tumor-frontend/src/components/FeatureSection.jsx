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
    <section className="relative py-12 md:py-16 z-10">
      <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center max-w-2xl mx-auto"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white">
            Engineered for Clinical Workflows
          </h2>
          <p className="mt-4 text-base md:text-lg text-gray-600 dark:text-gray-300">
            Seamlessly integrate AI-assisted diagnostics with intuitive visualization and fast inference.
          </p>
        </motion.div>

  <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
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
                className="group relative overflow-hidden rounded-3xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/80 shadow-md hover:shadow-lg transition-shadow duration-300"
              >
                <div className="relative p-8 space-y-4">
                  <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-blue-500/10 text-blue-600 dark:text-blue-400">
                    <Icon className="w-7 h-7" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}

export default FeatureSection
