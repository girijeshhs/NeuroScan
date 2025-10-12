/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Roboto', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Roboto Mono', 'monospace'],
      },
      colors: {
        medical: {
          blue: '#4A90E2',
          darkblue: '#2C5F8D',
          light: '#E8F4F8',
        },
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      },
      letterSpacing: {
        tighter: '-0.02em',
        tight: '-0.01em',
      },
    },
  },
  plugins: [],
}
