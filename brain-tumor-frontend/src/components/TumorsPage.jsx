import { motion } from 'framer-motion'
import { AlertCircle, Brain, Activity, Microscope } from 'lucide-react'

const TumorsPage = () => {
  const tumorTypes = [
    {
      name: 'Glioma',
      icon: Brain,
      color: 'red',
      description: 'Most common and aggressive brain tumor originating from glial cells.',
      characteristics: [
        'Rapid growth rate',
        'Irregular borders',
        'High malignancy potential',
        'Multiple subtypes (astrocytoma, oligodendroglioma)',
      ],
      prevalence: '33% of all brain tumors',
      survival: 'Varies by grade (I-IV)',
    },
    {
      name: 'Meningioma',
      icon: Activity,
      color: 'orange',
      description: 'Tumor arising from the meninges (protective membranes around brain/spinal cord).',
      characteristics: [
        'Usually benign (90%)',
        'Slow-growing',
        'Well-defined borders',
        'Often asymptomatic until large',
      ],
      prevalence: '36% of all brain tumors',
      survival: '5-year: 92% (benign)',
    },
    {
      name: 'Pituitary Tumor',
      icon: Microscope,
      color: 'purple',
      description: 'Tumor in the pituitary gland affecting hormone production and regulation.',
      characteristics: [
        'Mostly benign (>95%)',
        'Hormone-secreting or non-functional',
        'Can cause vision problems',
        'Treatable with surgery/medication',
      ],
      prevalence: '15% of all brain tumors',
      survival: '5-year: >95%',
    },
  ]

  const colorClasses = {
    red: {
      bg: 'bg-red-100 dark:bg-red-900/30',
      border: 'border-red-200 dark:border-red-800',
      text: 'text-red-600 dark:text-red-400',
      icon: 'bg-red-500',
    },
    orange: {
      bg: 'bg-orange-100 dark:bg-orange-900/30',
      border: 'border-orange-200 dark:border-orange-800',
      text: 'text-orange-600 dark:text-orange-400',
      icon: 'bg-orange-500',
    },
    purple: {
      bg: 'bg-purple-100 dark:bg-purple-900/30',
      border: 'border-purple-200 dark:border-purple-800',
      text: 'text-purple-600 dark:text-purple-400',
      icon: 'bg-purple-500',
    },
  }

  return (
    <section className="relative min-h-screen pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-red-100 dark:bg-red-900/30 rounded-full mb-6">
            <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
            <span className="text-sm font-medium text-red-700 dark:text-red-300">
              Medical Reference Guide
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-gray-900 dark:text-white mb-4">
            Brain Tumor Types
          </h2>
          <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto font-normal leading-relaxed">
            Our AI model detects and classifies three primary types of brain tumors with high accuracy.
            Understanding each type helps in diagnosis and treatment planning.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {tumorTypes.map((tumor, index) => {
            const Icon = tumor.icon
            const colors = colorClasses[tumor.color]
            return (
              <motion.div
                key={tumor.name}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.15 }}
                className={`rounded-2xl border-2 ${colors.border} ${colors.bg} p-6 hover:shadow-xl transition-shadow`}
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`${colors.icon} p-3 rounded-xl shadow-lg`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className={`text-2xl font-bold tracking-tight ${colors.text} mb-2`}>
                      {tumor.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 font-normal leading-relaxed">
                      {tumor.description}
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 tracking-wide uppercase">
                      Characteristics:
                    </h4>
                    <ul className="space-y-1.5">
                      {tumor.characteristics.map((char, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400"
                        >
                          <span className={`${colors.text} mt-1`}>•</span>
                          <span>{char}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="pt-4 border-t border-gray-300 dark:border-gray-700 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        Prevalence:
                      </span>
                      <span className={`font-semibold ${colors.text}`}>
                        {tumor.prevalence}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        Survival Rate:
                      </span>
                      <span className={`font-semibold ${colors.text}`}>
                        {tumor.survival}
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Additional Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-16 bg-blue-50 dark:bg-blue-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800"
        >
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Early Detection Saves Lives
          </h3>
          <div className="grid md:grid-cols-2 gap-6 text-gray-700 dark:text-gray-300">
            <div>
              <h4 className="font-semibold text-lg mb-2">Why Early Detection Matters:</h4>
              <ul className="space-y-2 text-sm">
                <li>• Increases treatment success rate by up to 90%</li>
                <li>• Allows for less invasive surgical options</li>
                <li>• Reduces long-term neurological damage</li>
                <li>• Improves quality of life outcomes</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-lg mb-2">Our AI Advantage:</h4>
              <ul className="space-y-2 text-sm">
                <li>• 96.4% accuracy across all tumor types</li>
                <li>• Instant analysis in under 4 seconds</li>
                <li>• Grad-CAM visualization for transparency</li>
                <li>• Supports radiologist decision-making</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default TumorsPage
