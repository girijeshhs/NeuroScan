import { Layers, TrendingUp, ExternalLink, CheckCircle } from 'lucide-react'

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

  const keySpecs = [

    { label: 'Model Architecture', value: 'Xception Transfer Learning' },
    { label: 'Training Dataset', value: '7,023 MRI Scans' },
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
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold text-slate-100 mb-4">
            Xception Neural Network
          </h2>
          <p className="text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Transfer learning architecture fine-tuned for brain tumor classification with explainable AI visualization
          </p>
        </div>

        {/* Two-Column Layout */}
        <div className="grid lg:grid-cols-[40%_60%] gap-8 lg:gap-10">
          
          {/* LEFT COLUMN */}
          <div className="space-y-6">
            
            {/* Google Colab Access Card */}
            <div className="medical-card p-6">
              <h3 className="text-xl font-semibold text-slate-200 mb-2">
                Access Model Training
              </h3>
              <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                View complete training notebook and implementation details
              </p>
              <a
                href="https://colab.research.google.com/drive/1RTGzaVTu872MxNpOrRDVYHq7LXA9JxLz?usp=sharing"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-3 w-full bg-gradient-to-r from-[#4a6b5a] to-[#2d5f4c] hover:from-[#5a7b6a] hover:to-[#3d6f5c] text-white font-medium py-3.5 px-6 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M16.9414 4.9757a7.033 7.033 0 0 0-4.9308 2.0646 7.033 7.033 0 0 0-.1232 9.8068l2.395-2.395a3.6455 3.6455 0 0 1 5.1497-5.1478l2.397-2.3989a7.033 7.033 0 0 0-4.8877-1.9297zM7.07 4.9855a7.033 7.033 0 0 0-4.8878 1.9316l2.3911 2.3911a3.6434 3.6434 0 0 1 5.0227.1271l1.7341-2.9737-.0997-.0802A7.033 7.033 0 0 0 7.07 4.9855zm15.0093 2.1721l-2.3892 2.3911a3.6455 3.6455 0 0 1-5.1497 5.1497l-2.4067 2.4068a7.0362 7.0362 0 0 0 9.9456-9.9476zM1.932 7.1674a7.033 7.033 0 0 0-.002 9.6816l2.397-2.397a3.6434 3.6434 0 0 1-.004-4.8916L1.932 7.1674zm7.664 7.4235c-1.38 1.3816-3.5863 1.411-5.0168.1134l-2.397 2.395c2.4693 2.3328 6.263 2.5753 9.0072.5455l-1.5934-2.9939zm7.4045-7.4045l-2.9693 1.5819c2.0356 2.7442 1.8138 6.6575-.5088 9.0875l2.3911 2.3911c3.4111-3.4252 3.4111-9.0182 0-12.4485l-.5069-.5007z"/>
                </svg>
                <span className="text-[15px]">Open in Google Colab</span>
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>

            {/* Why Xception Section */}
            <div className="medical-card p-6">
              <h3 className="text-xl font-semibold text-slate-200 mb-4">
                Why Xception Neural Network?
              </h3>
              <div className="space-y-4 text-[15px] text-slate-300 leading-relaxed">
                <p>
                  Xception was selected for its exceptional performance in medical imaging classification through transfer learning. 
                  Pre-trained on ImageNet's 1.4 million images, the model brings advanced depthwise separable convolution capabilities that 
                  provide better parameter efficiency and superior feature extraction while maintaining high accuracy on our specialized brain tumor dataset.
                </p>
                <p>
                  The architecture's depthwise separable convolutions create highly efficient feature representations ideal for 
                  identifying subtle patterns in MRI scans. With 36 convolutional blocks, this deep structure enables the network to learn increasingly 
                  abstract tumor characteristics—from basic edges and textures in early layers to complex pathological 
                  features in deeper layers—all while using fewer parameters than traditional CNNs.
                </p>
                <p>
                  Clinical validation requires explainability. Xception's modular architecture facilitates Grad-CAM 
                  visualization through its activation layers, highlighting the exact regions influencing classification decisions. This transparency 
                  allows radiologists to verify the model's reasoning, ensuring diagnoses align with established clinical 
                  criteria and building trust in AI-assisted workflows.
                </p>
              </div>
            </div>

            {/* Key Specifications */}
            <div className="medical-card p-5 border border-slate-700/30">
              <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
                Key Specifications
              </h4>
              <div className="space-y-3">
                {keySpecs.map((spec) => (
                  <div key={spec.label} className="flex items-center justify-between">
                    <span className="text-sm text-slate-400">{spec.label}</span>
                    <span className="text-sm font-semibold text-slate-200">{spec.value}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>

          {/* RIGHT COLUMN */}
          <div className="space-y-8">
            
            {/* Architecture Table */}
            <div className="medical-card p-7">
              <h3 className="text-2xl font-semibold text-slate-100 mb-6 flex items-center gap-3">
                <Layers className="w-6 h-6 text-[#6b9bd1]" />
                Network Architecture
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left pb-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Layer Type
                      </th>
                      <th className="text-left pb-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Configuration
                      </th>
                      <th className="text-right pb-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Parameters
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {architecture.map((layer, index) => (
                      <tr 
                        key={layer.layer}
                        className={`border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors ${
                          index % 2 === 0 ? 'bg-slate-900/20' : ''
                        }`}
                      >
                        <td className="py-4 font-medium text-slate-200 text-[15px]">
                          {layer.layer}
                        </td>
                        <td className="py-4 text-slate-400 text-[14px]">
                          {layer.details}
                        </td>
                        <td className="py-4 text-right font-mono text-[#6b9bd1] font-semibold text-[14px]">
                          {layer.neurons}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Classification Performance */}
            <div className="medical-card p-7">
              <h3 className="text-2xl font-semibold text-slate-100 mb-6 flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-[#6b9bd1]" />
                Classification Performance
              </h3>
              <div className="grid md:grid-cols-2 gap-5">
                {performance.map((perf) => (
                  <div
                    key={perf.class}
                    className="bg-[#1a1d29] rounded-lg p-5 border border-slate-700/30"
                  >
                    <div className="flex items-center gap-2 mb-4">
                      <CheckCircle className="w-4 h-4 text-[#6b9bd1]" />
                      <h4 className="text-lg font-semibold text-slate-200">
                        {perf.class}
                      </h4>
                    </div>
                    <div className="space-y-3">
                      {['accuracy', 'precision', 'recall', 'f1'].map((metric) => (
                        <div key={metric}>
                          <div className="flex justify-between text-xs mb-1.5">
                            <span className="capitalize text-slate-400 font-medium">
                              {metric === 'f1' ? 'F1-Score' : metric}
                            </span>
                            <span className="font-semibold text-slate-200 font-mono">
                              {perf[metric]}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                            <div
                              style={{ width: `${perf[metric]}%` }}
                              className="bg-[#6b9bd1] h-1.5 rounded-full transition-all duration-500"
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
        </div>
      </div>
    </section>
  )
}

export default ModelPage
