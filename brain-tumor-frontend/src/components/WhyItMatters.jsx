import { motion } from 'framer-motion'

const WhyItMatters = () => {
  return (
    <section className="relative py-8 md:py-10 z-10 bg-gray-50/50 dark:bg-gray-900/30">
      <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.4 }}
          className="bg-white dark:bg-gray-900/90 rounded-2xl p-6 md:p-8 border border-gray-200 dark:border-gray-800 shadow-lg"
        >
          <motion.h3
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-4"
          >
            Why Early Detection Matters
          </motion.h3>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, duration: 0.4 }}
            className="text-base md:text-lg text-gray-600 dark:text-gray-300 leading-relaxed"
          >
            Brain tumors can progress rapidly, and subtle abnormalities are often difficult to detect in early stages. By combining deep learning diagnostics with explainable Grad-CAM visualization, clinicians receive actionable insights in seconds. This assists in triaging urgent cases, validating treatment plans, and improving patient outcomes through timely intervention.
          </motion.p>
        </motion.div>
      </div>
    </section>
  )
}

export default WhyItMatters
