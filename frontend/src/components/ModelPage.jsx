import { ExternalLink, Github, Database } from 'lucide-react'

const ModelPage = () => {
  const keySpecs = [
    { label: 'Model Architecture', value: 'Xception Transfer Learning' },
    { label: 'Training Dataset', value: '7,023 MRI Scans' },
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
        <div className="grid lg:grid-cols-[45%_55%] gap-8 lg:gap-10">
          
          {/* LEFT COLUMN - Project Links */}
          <div className="space-y-6">
            
            {/* GitHub Repository */}
            <div className="medical-card p-6">
              <h3 className="text-xl font-semibold text-slate-200 mb-2">
                GitHub Repository
              </h3>
              <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                View source code, documentation, and project details
              </p>
              <a
                href="https://github.com/girijeshhs/ANN-BRAINTUMORPROJ"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-3 w-full bg-gradient-to-r from-[#24292e] to-[#1a1e22] hover:from-[#2f363d] hover:to-[#24292e] text-white font-medium py-3.5 px-6 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg border border-slate-700/30"
              >
                <Github className="w-5 h-5" />
                <span className="text-[15px]">View on GitHub</span>
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>

            {/* Google Colab Notebook */}
            <div className="medical-card p-6">
              <h3 className="text-xl font-semibold text-slate-200 mb-2">
                Google Colab Notebook
              </h3>
              <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                Access complete training notebook and implementation details
              </p>
              <a
                href="https://colab.research.google.com/drive/1gV5ulswcm5aEEOC5j5tA890DHuIbVjVZ?usp=sharing"
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

            {/* Dataset Link */}
            <div className="medical-card p-6">
              <h3 className="text-xl font-semibold text-slate-200 mb-2">
                Training Dataset
              </h3>
              <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                Access the brain tumor MRI dataset used for training
              </p>
              <a
                href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-3 w-full bg-gradient-to-r from-[#20beff] to-[#1a9dd1] hover:from-[#3dc8ff] hover:to-[#20beff] text-white font-medium py-3.5 px-6 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg"
              >
                <Database className="w-5 h-5" />
                <span className="text-[15px]">View Dataset on Kaggle</span>
                <ExternalLink className="w-4 h-4" />
              </a>
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

          {/* RIGHT COLUMN - Why Xception */}
          <div className="space-y-6">
            
            {/* Why Xception Section */}
            <div className="medical-card p-7">
              <h3 className="text-2xl font-semibold text-slate-100 mb-6">
                Why Xception Neural Network?
              </h3>
              <div className="space-y-5 text-[15px] text-slate-300 leading-relaxed">
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
                <div className="mt-6 pt-5 border-t border-slate-700/50">
                  <h4 className="text-lg font-semibold text-slate-200 mb-3">Key Advantages</h4>
                  <ul className="space-y-2 text-[14px]">
                    <li className="flex items-start gap-2">
                      <span className="text-[#6b9bd1] mt-1">•</span>
                      <span><strong className="text-slate-200">Transfer Learning:</strong> Leverages ImageNet pre-training for superior feature extraction</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-[#6b9bd1] mt-1">•</span>
                      <span><strong className="text-slate-200">Depthwise Separable Convolutions:</strong> Efficient parameter usage with high accuracy</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-[#6b9bd1] mt-1">•</span>
                      <span><strong className="text-slate-200">Deep Architecture:</strong> 36 convolutional blocks for complex pattern recognition</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-[#6b9bd1] mt-1">•</span>
                      <span><strong className="text-slate-200">Grad-CAM Integration:</strong> Provides explainable AI for clinical validation</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </section>
  )
}

export default ModelPage
