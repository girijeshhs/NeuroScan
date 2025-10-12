import { Brain, Zap, Eye, Shield } from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'VGG16-based convolutional neural network trained on curated MRI datasets for reliable tumor detection.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Eye,
    title: 'Grad-CAM Visualization',
    description: 'Explainable AI overlays highlight critical regions to support clinician decision-making.',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: Shield,
    title: 'High Accuracy (96%)',
    description: 'Rigorous evaluation with cross-validation to ensure consistent predictions in production.',
    color: 'from-emerald-500 to-teal-500',
  },
  {
    icon: Zap,
    title: 'Instant Results',
    description: 'Optimized pipeline delivers predictions and heatmaps in just a few seconds.',
    color: 'from-amber-500 to-orange-500',
  },
]

const FeatureSection = () => {
  return (
    <section className="relative py-16 bg-gray-50">
      <div className="max-w-[1400px] mx-auto px-8 sm:px-12">
        <div className="text-center mb-12">
          <div className="inline-block mb-4 px-5 py-2 bg-indigo-600 rounded-full text-xs font-bold text-white tracking-wide uppercase shadow-lg">
            AI-Powered Technology
          </div>
          <h2 className="text-4xl md:text-5xl font-black text-gray-900 mb-4">
            Clinical-Grade AI Analysis
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed max-w-3xl mx-auto">
            Advanced deep learning for rapid, accurate brain tumor detection with explainable results.
          </p>
        </div>

                <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative bg-white p-6 rounded-2xl shadow-md hover:shadow-2xl transition-shadow duration-200 border border-gray-200 hover:border-indigo-300"
            >
              <div className="relative">
                <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} mb-5 shadow-lg`}>
                  <feature.icon className="w-7 h-7 text-white" strokeWidth={2.5} />
                </div>
                <h3 className="text-xl font-black text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default FeatureSection
