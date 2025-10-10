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
    <section className="relative py-20 md:py-28 z-10">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-50/30 to-indigo-50/30 dark:via-blue-950/10 dark:to-indigo-950/10" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="inline-block mb-4 px-4 py-2 bg-blue-500/10 rounded-full text-sm font-semibold text-blue-600 dark:text-blue-400"
          >
            ✨ Powered by AI
          </motion.div>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-gray-900 via-blue-900 to-indigo-900 dark:from-white dark:via-blue-100 dark:to-indigo-100 bg-clip-text text-transparent mb-6">
            Engineered for Clinical Workflows
          </h2>
          <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 leading-relaxed">
            Seamlessly integrate AI-assisted diagnostics with intuitive visualization and lightning-fast inference.
          </p>
        </motion.div>

        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => {
            const Icon = featureIcons[feature.key]
            const colors = {
              AI: 'from-blue-500 to-cyan-500',
              GradCAM: 'from-purple-500 to-pink-500',
              Accuracy: 'from-emerald-500 to-teal-500',
              Instant: 'from-amber-500 to-orange-500',
            }
            return (
              <motion.div
                key={feature.title}
                custom={index}
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, amount: 0.2 }}
                whileHover={{ y: -8, transition: { duration: 0.2 } }}
                className="group relative overflow-hidden rounded-3xl bg-white dark:bg-gray-900 shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                {/* Gradient border effect */}
                <div className={`absolute inset-0 bg-gradient-to-br ${colors[feature.key]} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />
                <div className="absolute inset-[1px] rounded-3xl bg-white dark:bg-gray-900" />
                
                <div className="relative p-8 space-y-5">
                  <motion.div
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    className={`flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br ${colors[feature.key]} text-white shadow-lg`}
                  >
                    <Icon className="w-8 h-8" />
                  </motion.div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    {feature.title}
                  </h3>
                  <p className="text-base text-gray-600 dark:text-gray-300 leading-relaxed">
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
