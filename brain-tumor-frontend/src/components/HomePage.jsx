import { TrendingUp, Clock, Database, ArrowRight, Activity, FileText, BarChart3, CheckCircle2 } from 'lucide-react'

const HomePage = ({ onGetStarted }) => {
  const stats = [
    { label: 'Diagnostic Accuracy', value: '96.4%', icon: TrendingUp, color: 'border-forest', trend: '+2.3%' },
    { label: 'Average Analysis Time', value: '3.8s', icon: Clock, color: 'border-burgundy', trend: '-0.5s' },
    { label: 'Scans Processed', value: '3,247', icon: Database, color: 'border-oxford', trend: '+127' },
    { label: 'Model Confidence', value: '94.2%', icon: BarChart3, color: 'border-forest', trend: '+1.8%' },
  ]

  const recentAnalyses = [
    { id: 'MRI-2847', type: 'Glioma', confidence: '97.2%', date: '2 hours ago', status: 'complete' },
    { id: 'MRI-2846', type: 'Meningioma', confidence: '95.8%', date: '5 hours ago', status: 'complete' },
    { id: 'MRI-2845', type: 'No Tumor', confidence: '98.1%', date: '8 hours ago', status: 'complete' },
  ]

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
              <div className="inline-flex items-center gap-2 px-3 py-1 mb-4 rounded-full bg-burgundy/10 border border-burgundy/20">
                <div className="w-1.5 h-1.5 rounded-full bg-red-600 animate-pulse" />
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Clinical-Grade AI Diagnostics</span>
              </div>
              
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
                <button className="btn-secondary px-6 py-2.5 rounded-md text-slate-300 font-semibold text-sm">
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
              <button className="text-sm text-burgundy hover:text-red-500 font-medium">View All</button>
            </div>
            
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
                  {recentAnalyses.map((analysis) => (
                    <tr key={analysis.id} className="cursor-pointer">
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
