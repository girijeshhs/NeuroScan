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
    <section className="relative min-h-screen pt-20 pb-16 bg-gray-50">
      <div className="max-w-[1400px] mx-auto px-8 sm:px-12">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-5 py-2 bg-indigo-600 rounded-full mb-6 shadow-lg">
            <Cpu className="w-4 h-4 text-white" />
            <span className="text-sm font-bold text-white uppercase tracking-wide">
              Technical Specifications
            </span>
          </div>
          <h2 className="text-5xl md:text-6xl font-black text-gray-900 mb-6">
            VGG16 Deep Learning Model
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Fine-tuned convolutional neural network architecture optimized for medical imaging classification with explainable AI via Grad-CAM visualization.
          </p>
        </div>

        {/* Model Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 mb-16">
          {metrics.map((metric) => {
            const Icon = metric.icon
            return (
              <div
                key={metric.label}
                className="bg-white rounded-xl p-6 border-2 border-gray-200 text-center shadow-md hover:shadow-xl transition-shadow"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-indigo-100 mb-3">
                  <Icon className="w-6 h-6 text-indigo-600" />
                </div>
                <p className="text-2xl font-black text-gray-900 tracking-tight">
                  {metric.value}
                </p>
                <p className="text-xs text-gray-600 font-bold tracking-wide uppercase">
                  {metric.label}
                </p>
              </div>
            )
          })}
        </div>

        {/* Architecture Table */}
        <div className="bg-white rounded-2xl p-8 border-2 border-gray-200 mb-16 shadow-lg">
          <h3 className="text-3xl font-black text-gray-900 mb-8 flex items-center gap-3">
            <Layers className="w-8 h-8 text-indigo-600" />
            Network Architecture
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b-2 border-gray-300">
                  <th className="pb-4 text-sm font-bold text-gray-700 uppercase tracking-wide">
                    Layer
                  </th>
                  <th className="pb-4 text-sm font-bold text-gray-700 uppercase tracking-wide">
                    Configuration
                  </th>
                  <th className="pb-4 text-sm font-bold text-gray-700 uppercase tracking-wide text-right">
                    Parameters
                  </th>
                </tr>
              </thead>
              <tbody>
                {architecture.map((layer) => (
                  <tr
                    key={layer.layer}
                    className="border-b border-gray-200 hover:bg-gray-50 transition-colors"
                  >
                    <td className="py-5 font-bold text-gray-900 text-base">
                      {layer.layer}
                    </td>
                    <td className="py-5 text-gray-600 text-base">
                      {layer.details}
                    </td>
                    <td className="py-5 text-right font-mono text-indigo-600 font-bold text-base">
                      {layer.neurons}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl p-10 border-2 border-indigo-200 shadow-lg">
          <h3 className="text-3xl font-black text-gray-900 mb-8 flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-indigo-600" />
            Classification Performance
          </h3>
          <div className="grid md:grid-cols-2 gap-8">
            {performance.map((perf) => (
              <div
                key={perf.class}
                className="bg-white rounded-xl p-8 border-2 border-gray-200 shadow-md"
              >
                <h4 className="text-2xl font-black text-gray-900 mb-6">
                  {perf.class}
                </h4>
                <div className="space-y-4">
                  {['accuracy', 'precision', 'recall', 'f1'].map((metric) => (
                    <div key={metric}>
                      <div className="flex justify-between text-base mb-2">
                        <span className="capitalize text-gray-600 font-bold">
                          {metric === 'f1' ? 'F1-Score' : metric}:
                        </span>
                        <span className="font-black text-gray-900">
                          {perf[metric]}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          style={{ width: `${perf[metric]}%` }}
                          className="bg-gradient-to-r from-indigo-500 to-blue-500 h-3 rounded-full transition-all"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export default ModelPage
