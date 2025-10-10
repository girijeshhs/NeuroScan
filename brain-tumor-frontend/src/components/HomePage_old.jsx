import { Brain, Zap, TrendingUp, CheckCircle, Shield, Clock, Sparkles } from 'lucide-react'

const HomePage = ({ onGetStarted }) => {
  const stats = [
    { label: 'Accuracy Rate', value: '96.4%', icon: TrendingUp, color: 'from-emerald-500 to-teal-500' },
    { label: 'Processing Time', value: '<4s', icon: Zap, color: 'from-amber-500 to-orange-500' },
    { label: 'Models Trained', value: '3.2K+', icon: Brain, color: 'from-purple-500 to-pink-500' },
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-24 pb-20 overflow-hidden">
      {/* Premium Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 -left-20 w-96 h-96 bg-gradient-to-br from-indigo-400/20 to-purple-400/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 -right-20 w-96 h-96 bg-gradient-to-br from-blue-400/20 to-cyan-400/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-br from-purple-300/10 to-pink-300/10 rounded-full blur-3xl" />
      </div>

      <div className="max-w-7xl mx-auto px-12 relative z-10">
        <div className="grid lg:grid-cols-2 gap-20 items-center">
          {/* Left Content */}
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-5 py-2 bg-indigo-600 rounded-full shadow-lg">
              <CheckCircle className="w-4 h-4 text-white" />
              <span className="text-xs font-bold text-white tracking-wide uppercase">
                AI-Powered Medical Diagnosis
              </span>
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black leading-[1.05] tracking-tight">
              <span className="text-gray-900">Brain Tumor</span>
              <br />
              <span className="bg-gradient-to-r from-indigo-600 via-blue-600 to-purple-600 bg-clip-text text-transparent">
                Detection AI
              </span>
            </h1>

            <p className="text-xl text-gray-600 leading-relaxed max-w-2xl">
              State-of-the-art deep learning for instant MRI analysis. Upload brain scans and receive accurate predictions with explainable Grad-CAM visualizations.
            </p>

            <div className="flex flex-wrap gap-4">
              <button
                onClick={onGetStarted}
                className="px-8 py-4 bg-indigo-600 text-white font-bold text-base rounded-xl shadow-xl hover:shadow-2xl hover:bg-indigo-700 transition-all"
              >
                Start Analysis →
              </button>
              <button className="px-8 py-4 border-2 border-gray-300 text-gray-700 font-bold text-base rounded-xl hover:border-indigo-600 hover:text-indigo-600 transition-all">
                View Demo
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 pt-4">
              {stats.map((stat, index) => {
                const Icon = stat.icon
                return (
                  <div
                    key={stat.label}
                    className="text-center"
                  >
                    <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-indigo-100 mb-3 shadow-md">
                      <Icon className="w-7 h-7 text-indigo-600" />
                    </div>
                    <p className="text-2xl font-black text-gray-900">
                      {stat.value}
                    </p>
                    <p className="text-sm text-gray-600 font-medium">{stat.label}</p>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right Visual */}
          <div className="relative">
            <div className="relative w-full aspect-square max-w-lg mx-auto">
              {/* Static elegant rings */}
              <div className="absolute inset-0 rounded-full border-4 border-indigo-100" />
              <div className="absolute inset-8 rounded-full border-4 border-blue-100" />
              <div className="absolute inset-16 rounded-full border-4 border-purple-100" />

              {/* Center brain icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-full blur-3xl opacity-20" />
                  <div className="relative bg-white p-12 rounded-full shadow-2xl border-4 border-indigo-200">
                    <Brain className="w-32 h-32 text-indigo-600" />
                  </div>
                </div>
              </div>

              {/* Static info cards */}
              <div className="absolute top-0 right-0 bg-white px-5 py-3 rounded-xl shadow-lg border-2 border-green-200">
                <p className="text-xs font-bold text-gray-500 uppercase">Accuracy</p>
                <p className="text-xl font-black text-green-600">96.4%</p>
              </div>

              <div className="absolute bottom-0 left-0 bg-white px-5 py-3 rounded-xl shadow-lg border-2 border-blue-200">
                <p className="text-xs font-bold text-gray-500 uppercase">Grad-CAM</p>
                <p className="text-xl font-black text-blue-600">Active</p>
              </div>
              
              <div className="absolute top-1/3 left-0 bg-white px-5 py-3 rounded-xl shadow-lg border-2 border-purple-200">
                <p className="text-xs font-bold text-gray-500 uppercase">Speed</p>
                <p className="text-xl font-black text-purple-600">&lt;4s</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default HomePage
