import { Github, Linkedin, Mail } from 'lucide-react'

const SiteFooter = () => {
  return (
    <footer className="relative border-t border-white/60 dark:border-gray-800 bg-white/80 dark:bg-gray-950/80 backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-10 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="text-center md:text-left">
          <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">
            Made by using Xception
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Powered by TensorFlow · Flask · React · Tailwind CSS
          </p>
        </div>

        <div className="flex items-center gap-4 text-gray-600 dark:text-gray-300">
          <a
            href="https://github.com/girijeshhs"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-full border border-transparent hover:border-blue-500/60 hover:text-blue-600 dark:hover:text-blue-400 transition"
            aria-label="GitHub"
          >
            <Github className="w-5 h-5" />
          </a>
          <a
            href="https://www.linkedin.com/in/girijeshhs"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-full border border-transparent hover:border-blue-500/60 hover:text-blue-600 dark:hover:text-blue-400 transition"
            aria-label="LinkedIn"
          >
            <Linkedin className="w-5 h-5" />
          </a>
          <a
            href="mailto:girijeshhs@example.com"
            className="p-2 rounded-full border border-transparent hover:border-blue-500/60 hover:text-blue-600 dark:hover:text-blue-400 transition"
            aria-label="Email"
          >
            <Mail className="w-5 h-5" />
          </a>
        </div>
      </div>
      <div className="text-center text-xs text-gray-500 dark:text-gray-500 pb-6">
        © {new Date().getFullYear()} Brain Tumor Detection. All rights reserved.
      </div>
    </footer>
  )
}

export default SiteFooter
