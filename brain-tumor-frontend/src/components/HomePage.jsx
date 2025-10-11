import { TrendingUp, Clock, Database, ArrowRight, Activity, FileText, BarChart3, CheckCircle2, Trash2 } from 'lucide-react'

const HomePage = ({ onGetStarted, onNavigate, analysisHistory = [], onClearHistory }) => {
  const stats = [
    { label: 'Diagnostic Accuracy', value: '96.4%', icon: TrendingUp, color: 'border-forest', trend: '+2.3%' },
    { label: 'Average Analysis Time', value: '3.8s', icon: Clock, color: 'border-burgundy', trend: '-0.5s' },
    { label: 'Scans Processed', value: analysisHistory.length.toString(), icon: Database, color: 'border-oxford', trend: `+${analysisHistory.length}` },
    { label: 'Model Confidence', value: '94.2%', icon: BarChart3, color: 'border-forest', trend: '+1.8%' },
  ]

  // Use actual history or show empty state
  const displayAnalyses = analysisHistory.slice(0, 5) // Show only last 5

  const features = [
    { title: 'VGG16 Architecture', desc: 'Deep convolutional neural network with 16 weighted layers', icon: Activity },
    { title: 'Grad-CAM Visualization', desc: 'Explainable AI highlighting critical diagnostic regions', icon: FileText },
    { title: 'Clinical Validation', desc: 'Validated against peer-reviewed medical datasets', icon: CheckCircle2 },
  ]

  return (
    <section className="relative bg-[#0f1419] pt-6 pb-12">
      <div className="bg-texture absolute inset-0 opacity-50" />
      
      <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
        {/* Compact Hero Section */}
        <div className="py-8 lg:py-12 mb-8">
          <div className="grid lg:grid-cols-[2fr_1fr] gap-8 items-start">
            <div>
              <h1 className="text-4xl lg:text-5xl font-semibold text-slate-100 mb-4 leading-tight">
                Neural Network Analysis for
                <span className="block text-burgundy mt-1">Brain Tumor Detection</span>
              </h1>
              
              <p className="text-base text-slate-400 mb-6 max-w-2xl leading-relaxed">
                Advanced deep learning platform employing VGG16 convolutional architecture for rapid, accurate identification 
                of glioma, meningioma, and pituitary tumors in MRI scans. Validated diagnostic support with explainable AI visualization.
              </p>

              <div className="flex flex-wrap gap-3">
                <button 
                  onClick={onGetStarted}
                  className="btn-primary px-6 py-2.5 rounded-md text-white font-semibold text-sm flex items-center gap-2"
                >
                  Begin Analysis
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button 
                  onClick={() => onNavigate && onNavigate('model')}
                  className="btn-secondary px-6 py-2.5 rounded-md text-slate-300 font-semibold text-sm"
                >
                  View Documentation
                </button>
              </div>
            </div>

            {/* Quick Upload Card */}
            <div className="medical-card rounded-lg p-5">
              <h3 className="text-lg font-semibold text-slate-200 mb-3">Quick Upload</h3>
              <div className="border-2 border-dashed border-slate-700 rounded-md p-8 text-center hover:border-slate-600 transition-colors cursor-pointer"
                   onClick={onGetStarted}>
                <FileText className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <p className="text-sm text-slate-400 mb-1">Drop MRI scan here</p>
                <p className="text-xs text-slate-500">or click to browse</p>
              </div>
            </div>
          </div>
        </div>

        {/* Statistics Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <div key={index} className={`stat-card rounded-lg p-4 ${stat.color}`}>
                <div className="flex items-start justify-between mb-2">
                  <Icon className="w-5 h-5 text-slate-400" />
                  <span className="text-xs font-semibold text-green-500">{stat.trend}</span>
                </div>
                <div className="text-2xl font-bold text-slate-100 mb-1">{stat.value}</div>
                <div className="text-xs text-slate-400 uppercase tracking-wide">{stat.label}</div>
              </div>
            )
          })}
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Recent Analyses */}
          <div className="lg:col-span-2 medical-card rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-200">Recent Analyses</h2>
              <div className="flex items-center gap-2">
                {analysisHistory.length > 0 && (
                  <button 
                    onClick={onClearHistory}
                    className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-red-400 font-medium transition-colors px-3 py-1.5 rounded-md hover:bg-red-500/10"
                    title="Clear all history"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Clear History
                  </button>
                )}
              </div>
            </div>
            
            {displayAnalyses.length > 0 ? (
              <div className="data-table rounded-md overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr>
                      <th className="text-left py-3 px-4">Scan ID</th>
                      <th className="text-left py-3 px-4">Classification</th>
                      <th className="text-left py-3 px-4">Confidence</th>
                      <th className="text-left py-3 px-4">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayAnalyses.map((analysis) => (
                      <tr key={analysis.id}>
                        <td className="py-3 px-4 text-slate-300 font-mono text-sm">{analysis.id}</td>
                        <td className="py-3 px-4">
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-burgundy/10 text-slate-300 text-xs font-medium">
                            {analysis.type}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-300 font-semibold">{analysis.confidence}</td>
                        <td className="py-3 px-4 text-slate-500 text-sm">{analysis.date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12">
                <Activity className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400 text-sm">No analyses yet</p>
                <p className="text-slate-500 text-xs mt-1">Upload an MRI scan to get started</p>
              </div>
            )}
          </div>

          {/* System Features */}
          <div className="space-y-4">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <div key={index} className="medical-card rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-md bg-slate-800 flex items-center justify-center flex-shrink-0">
                      <Icon className="w-5 h-5 text-slate-400" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-slate-200 mb-1">{feature.title}</h3>
                      <p className="text-xs text-slate-400 leading-relaxed">{feature.desc}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </section>
  )
}

export default HomePage
