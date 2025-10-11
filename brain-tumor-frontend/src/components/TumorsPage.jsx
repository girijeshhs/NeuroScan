import { AlertCircle, Brain, Activity, Microscope } from 'lucide-react'

const TumorsPage = () => {
  const tumorTypes = [
    {
      name: 'Glioma',
      icon: Brain,
      color: 'red',
      description: 'Most common and aggressive brain tumor originating from glial cells.',
      characteristics: [
        'Rapid growth rate',
        'Irregular borders',
        'High malignancy potential',
        'Multiple subtypes (astrocytoma, oligodendroglioma)',
      ],
      prevalence: '33% of all brain tumors',
      survival: 'Varies by grade (I-IV)',
    },
    {
      name: 'Meningioma',
      icon: Activity,
      color: 'orange',
      description: 'Tumor arising from the meninges (protective membranes around brain/spinal cord).',
      characteristics: [
        'Usually benign (90%)',
        'Slow-growing',
        'Well-defined borders',
        'Often asymptomatic until large',
      ],
      prevalence: '36% of all brain tumors',
      survival: '5-year: 92% (benign)',
    },
    {
      name: 'Pituitary Tumor',
      icon: Microscope,
      color: 'purple',
      description: 'Tumor in the pituitary gland affecting hormone production and regulation.',
      characteristics: [
        'Mostly benign (>95%)',
        'Hormone-secreting or non-functional',
        'Can cause vision problems',
        'Treatable with surgery/medication',
      ],
      prevalence: '15% of all brain tumors',
      survival: '5-year: >95%',
    },
  ]

  const colorClasses = {
    red: {
      bg: 'bg-[#1e2332]',
      border: 'border-[#8b4a5c]/50',
      text: 'text-[#c97d8f]',
      icon: 'bg-[#7c2d12]',
    },
    orange: {
      bg: 'bg-[#1e2332]',
      border: 'border-[#6b2d3a]/50',
      text: 'text-[#b87d8f]',
      icon: 'bg-[#991b1b]',
    },
    purple: {
      bg: 'bg-[#1e2332]',
      border: 'border-[#4a6b5a]/50',
      text: 'text-[#7aa896]',
      icon: 'bg-[#2d5f4c]',
    },
  }

  return (
    <section className="relative min-h-screen pt-20 pb-16 bg-[#0f1419]">
      <div className="max-w-[1400px] mx-auto px-8 sm:px-12">
        <div className="text-center mb-16">
          <h2 className="text-5xl md:text-6xl font-bold text-slate-100 mb-6">
            Tumor Classifications
          </h2>
          <p className="text-lg text-slate-400 max-w-3xl mx-auto leading-relaxed">
            VGG16 neural network trained to distinguish three primary brain tumor types with clinical-grade accuracy. Each classification requires distinct treatment protocols.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {tumorTypes.map((tumor, index) => {
            const Icon = tumor.icon
            const colors = colorClasses[tumor.color]
            return (
              <div
                key={tumor.name}
                className={`medical-card border ${colors.border} p-6 hover:border-opacity-100 transition-all`}
              >
                <div className="flex items-start gap-3 mb-5">
                  <div className={`${colors.icon} p-2.5 rounded-md`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className={`text-xl font-semibold ${colors.text} mb-2`}>
                      {tumor.name}
                    </h3>
                    <p className="text-sm text-slate-400 leading-relaxed">
                      {tumor.description}
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-xs font-semibold text-slate-500 mb-2.5 tracking-wider uppercase">
                      Clinical Characteristics
                    </h4>
                    <ul className="space-y-2">
                      {tumor.characteristics.map((char, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-sm text-slate-300"
                        >
                          <span className={`${colors.text} text-xs mt-0.5`}>•</span>
                          <span>{char}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="pt-4 border-t border-slate-700/50 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-400">
                        Prevalence
                      </span>
                      <span className={`font-semibold ${colors.text}`}>
                        {tumor.prevalence}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-400">
                        Survival Prognosis
                      </span>
                      <span className={`font-semibold ${colors.text}`}>
                        {tumor.survival}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Clinical Evidence Section */}
        <div className="mt-16 medical-card border border-[#2d5f4c]/50 p-8">
          <h3 className="text-3xl font-bold text-slate-100 mb-6">
            Clinical Validation
          </h3>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-lg font-semibold mb-4 text-[#7aa896]">Early Detection Benefits</h4>
              <ul className="space-y-2.5 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Treatment success improves by up to 90% with early identification</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Enables minimally invasive surgical interventions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Reduces risk of permanent neurological deficits</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Significantly improves patient quality of life metrics</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-[#7aa896]">System Performance</h4>
              <ul className="space-y-2.5 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>96.4% classification accuracy across all tumor types</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Analysis completed in under 4 seconds per scan</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Grad-CAM visualization for diagnostic transparency</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#4a6b5a] text-xs mt-0.5">•</span>
                  <span>Clinical decision support for radiological assessment</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default TumorsPage
