import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DialogProvider } from './components/Dialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ToastProvider } from './components/Toast'
import { initTheme } from './lib/theme'
import './i18n'
import i18n, { getStoredLang, setStoredLang } from './i18n'
import './index.css'

initTheme()

function mount() {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <ErrorBoundary>
        <ToastProvider>
          <DialogProvider>
            <App />
          </DialogProvider>
        </ToastProvider>
      </ErrorBoundary>
    </React.StrictMode>,
  )
}

async function bootstrap() {
  // If user already has a stored preference, skip the API check entirely.
  if (!getStoredLang()) {
    try {
      const res = await fetch('/api/system/lang')
      if (res.ok) {
        const data = (await res.json()) as { lang: string | null }
        if (data.lang === 'en' || data.lang === 'zh') {
          setStoredLang(data.lang)
          await i18n.changeLanguage(data.lang)
        }
      }
    } catch {
      // ignore — default zh applies
    }
  }
  mount()
}

void bootstrap()
