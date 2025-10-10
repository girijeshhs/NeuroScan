import { motion } from 'framer-motion'
import { Layers, Database, Cpu, GitBranch, TrendingUp, Zap } from 'lucide-react'

const ModelPage = () => {
  const architecture = [
    {
      layer: 'Input Layer',
      details: '224×224×3 RGB MRI Image',
      neurons: '150,528',
    },
    {
      layer: 'Conv Block 1-2',
      details: '64 filters, 3×3, ReLU + MaxPool',
      neurons: '~1.8M',
    },
    {
      layer: 'Conv Block 3-4',
      details: '256-512 filters, 3×3, ReLU',
      neurons: '~7.1M',
    },
    {
      layer: 'Conv Block 5',
      details: '512 filters, Grad-CAM target',
      neurons: '~7.6M',
    },
    {
      layer: 'Dense Layers',
      details: 'FC-4096 → FC-4096 → Dropout',
      neurons: '~33.6M',
    },
    {
      layer: 'Output Layer',
      details: 'Softmax, 4 classes',
      neurons: '4',
    },
  ]

  const metrics = [
    { label: 'Total Parameters', value: '138M', icon: Database },
    { label: 'Training Epochs', value: '50', icon: TrendingUp },
    { label: 'Batch Size', value: '32', icon: Layers },
    { label: 'Learning Rate', value: '0.0001', icon: GitBranch },
    { label: 'Optimizer', value: 'Adam', icon: Cpu },
    { label: 'Inference Time', value: '<4s', icon: Zap },
  ]

  const performance = [
    { class: 'Glioma', accuracy: 97.2, precision: 96.8, recall: 97.6, f1: 97.2 },
    { class: 'Meningioma', accuracy: 96.1, precision: 95.4, recall: 96.8, f1: 96.1 },
    { class: 'Pituitary', accuracy: 98.3, precision: 98.1, recall: 98.5, f1: 98.3 },
    { class: 'No Tumor', accuracy: 94.2, precision: 93.7, recall: 94.8, f1: 94.2 },
  ]

  return (
    <section className="relative min-h-screen pt-20 pb-12 bg-white">
      <div className="max-w-[1400px] mx-auto px-8 sm:px-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-full mb-6">
            <Cpu className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <span className="text-sm font-medium text-indigo-700 dark:text-indigo-300">
              Technical Specifications
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-gray-900 dark:text-white mb-4">
            VGG16 Deep Learning Model
          </h2>
          <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto font-normal leading-relaxed">
            Fine-tuned convolutional neural network architecture optimized for medical imaging classification
            with explainable AI via Grad-CAM visualization.
          </p>
        </motion.div>

        {/* Model Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
          {metrics.map((metric, index) => {
            const Icon = metric.icon
            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800 text-center"
              >
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 mb-2">
                  <Icon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
                  {metric.value}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 font-medium tracking-wide uppercase">
                  {metric.label}
                </p>
              </motion.div>
            )
          })}
        </div>

        {/* Architecture Table */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800 mb-16"
        >
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            Network Architecture
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b-2 border-gray-200 dark:border-gray-800">
                  <th className="pb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Layer
                  </th>
                  <th className="pb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Configuration
                  </th>
                  <th className="pb-3 text-sm font-semibold text-gray-700 dark:text-gray-300 text-right">
                    Parameters
                  </th>
                </tr>
              </thead>
              <tbody>
                {architecture.map((layer, index) => (
                  <motion.tr
                    key={layer.layer}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-4 font-medium text-gray-900 dark:text-white">
                      {layer.layer}
                    </td>
                    <td className="py-4 text-gray-600 dark:text-gray-400">
                      {layer.details}
                    </td>
                    <td className="py-4 text-right font-mono text-indigo-600 dark:text-indigo-400">
                      {layer.neurons}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-950/30 dark:to-blue-950/30 rounded-2xl p-8 border border-indigo-200 dark:border-indigo-800"
        >
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            Classification Performance
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {performance.map((perf, index) => (
              <motion.div
                key={perf.class}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                className="bg-white dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-800"
              >
                <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                  {perf.class}
                </h4>
                <div className="space-y-3">
                  {['accuracy', 'precision', 'recall', 'f1'].map((metric) => (
                    <div key={metric}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="capitalize text-gray-600 dark:text-gray-400">
                          {metric === 'f1' ? 'F1-Score' : metric}:
                        </span>
                        <span className="font-semibold text-gray-900 dark:text-white">
                          {perf[metric]}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${perf[metric]}%` }}
                          transition={{ duration: 1, delay: 0.8 + index * 0.1 }}
                          className="bg-gradient-to-r from-indigo-500 to-blue-500 h-2 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Training Details */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="mt-12 grid md:grid-cols-2 gap-8"
        >
          <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
            <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Training Configuration
            </h4>
            <ul className="space-y-3 text-gray-700 dark:text-gray-300">
              <li className="flex justify-between">
                <span>Dataset Size:</span>
                <span className="font-semibold">3,264 MRI scans</span>
              </li>
              <li className="flex justify-between">
                <span>Train/Val/Test Split:</span>
                <span className="font-semibold">70/15/15</span>
              </li>
              <li className="flex justify-between">
                <span>Data Augmentation:</span>
                <span className="font-semibold">Rotation, Flip, Zoom</span>
              </li>
              <li className="flex justify-between">
                <span>Loss Function:</span>
                <span className="font-semibold">Categorical Cross-Entropy</span>
              </li>
              <li className="flex justify-between">
                <span>Early Stopping:</span>
                <span className="font-semibold">Patience: 10 epochs</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
            <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Grad-CAM Explainability
            </h4>
            <ul className="space-y-3 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 dark:text-indigo-400 mt-1">•</span>
                <span>Generates heatmaps from last convolutional layer (block5_conv3)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 dark:text-indigo-400 mt-1">•</span>
                <span>Highlights regions that influenced the tumor classification</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 dark:text-indigo-400 mt-1">•</span>
                <span>JET colormap: Red/yellow = high activation zones</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 dark:text-indigo-400 mt-1">•</span>
                <span>Aids radiologists in understanding AI decision-making</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 dark:text-indigo-400 mt-1">•</span>
                <span>Only generated when tumor is detected (no false visualization)</span>
              </li>
            </ul>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default ModelPage
