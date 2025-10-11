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
    <section className="relative min-h-screen pt-20 pb-16 bg-[#0f1419]">
      <div className="max-w-[1400px] mx-auto px-8 sm:px-12">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-[#1e2332] border border-[#1e40af]/50 rounded-md mb-6">
            <Cpu className="w-3.5 h-3.5 text-[#6b9bd1]" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              System Architecture
            </span>
          </div>
          <h2 className="text-5xl md:text-6xl font-bold text-slate-100 mb-6">
            VGG16 Neural Network
          </h2>
          <p className="text-lg text-slate-400 max-w-3xl mx-auto leading-relaxed">
            Transfer learning from ImageNet pre-trained weights, fine-tuned on medical imaging dataset with Grad-CAM explainability for clinical transparency.
          </p>
        </div>

        {/* Model Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
          {metrics.map((metric) => {
            const Icon = metric.icon
            return (
              <div
                key={metric.label}
                className="stat-card text-center"
              >
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-md bg-[#1e40af]/20 mb-2.5">
                  <Icon className="w-5 h-5 text-[#6b9bd1]" />
                </div>
                <p className="text-xl font-bold text-slate-200 tracking-tight mb-1">
                  {metric.value}
                </p>
                <p className="text-[10px] text-slate-500 font-semibold tracking-wider uppercase">
                  {metric.label}
                </p>
              </div>
            )
          })}
        </div>

        {/* Architecture Table */}
        <div className="medical-card p-8 mb-16">
          <h3 className="text-3xl font-bold text-slate-100 mb-8 flex items-center gap-3">
            <Layers className="w-7 h-7 text-[#6b9bd1]" />
            Network Architecture
          </h3>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Layer Type</th>
                  <th>Configuration</th>
                  <th className="text-right">Parameters</th>
                </tr>
              </thead>
              <tbody>
                {architecture.map((layer) => (
                  <tr key={layer.layer}>
                    <td className="font-semibold text-slate-200">
                      {layer.layer}
                    </td>
                    <td className="text-slate-400">
                      {layer.details}
                    </td>
                    <td className="text-right font-mono text-[#6b9bd1] font-semibold">
                      {layer.neurons}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="medical-card border border-[#1e40af]/50 p-8">
          <h3 className="text-3xl font-bold text-slate-100 mb-8 flex items-center gap-3">
            <TrendingUp className="w-7 h-7 text-[#6b9bd1]" />
            Classification Performance
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {performance.map((perf) => (
              <div
                key={perf.class}
                className="bg-[#1e2332] rounded-lg p-6 border border-slate-700/50"
              >
                <h4 className="text-xl font-semibold text-slate-200 mb-5">
                  {perf.class}
                </h4>
                <div className="space-y-3.5">
                  {['accuracy', 'precision', 'recall', 'f1'].map((metric) => (
                    <div key={metric}>
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className="capitalize text-slate-400 font-medium">
                          {metric === 'f1' ? 'F1-Score' : metric}
                        </span>
                        <span className="font-semibold text-slate-200">
                          {perf[metric]}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          style={{ width: `${perf[metric]}%` }}
                          className="bg-[#6b9bd1] h-1.5 rounded-full transition-all"
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
