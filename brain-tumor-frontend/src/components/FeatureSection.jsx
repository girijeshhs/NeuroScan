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
    <section className="relative py-12 z-10 bg-gradient-to-b from-blue-50/50 to-white">
      <div className="relative max-w-[1400px] mx-auto px-8 sm:px-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="text-center mb-10"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4 }}
            className="inline-block mb-3 px-4 py-1.5 bg-indigo-100 rounded-full text-xs font-bold text-indigo-700 tracking-wide uppercase"
          >
            ✨ AI-Powered
          </motion.div>
          <h2 className="text-3xl md:text-4xl font-black tracking-tight text-gray-900 mb-3">
            Engineered for Clinical Workflows
          </h2>
          <p className="text-base text-gray-600 leading-relaxed font-normal max-w-2xl mx-auto">
            Seamlessly integrate AI-assisted diagnostics with intuitive visualization and lightning-fast inference.
          </p>
        </motion.div>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
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
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
                className="group relative overflow-hidden rounded-2xl bg-white border-2 border-gray-100 hover:border-indigo-200 hover:shadow-lg transition-all duration-300"
              >
                <div className="relative p-6 space-y-4">
                  <motion.div
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.5 }}
                    className={`flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br ${colors[feature.key]} text-white shadow-md`}
                  >
                    <Icon className="w-7 h-7" />
                  </motion.div>
                  <h3 className="text-lg font-bold text-gray-900 tracking-tight">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-600 leading-relaxed font-normal">
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
