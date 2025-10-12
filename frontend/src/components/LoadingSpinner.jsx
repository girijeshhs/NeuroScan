import { motion } from 'framer-motion'

const LoadingSpinner = () => {
  return (
    <motion.div
      className="inline-block w-6 h-6 border-3 border-white border-t-transparent rounded-full"
      animate={{ rotate: 360 }}
      transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }}
    />
  )
}

export default LoadingSpinner
