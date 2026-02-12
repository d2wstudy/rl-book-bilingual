import { ref } from 'vue'

export type Lang = 'zh' | 'en'

const defaultLang = ref<Lang>('zh')

export function useLang() {
  function toggleDefault() {
    defaultLang.value = defaultLang.value === 'zh' ? 'en' : 'zh'
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('rl-book-lang', defaultLang.value)
    }
    applyDefaultLang()
  }

  function initLang() {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('rl-book-lang') as Lang | null
      if (saved === 'zh' || saved === 'en') {
        defaultLang.value = saved
      }
    }
  }

  return { defaultLang, toggleDefault, initLang }
}

/** Apply default language to all paragraph pairs that haven't been manually flipped */
export function applyDefaultLang() {
  if (typeof document === 'undefined') return
  const lang = defaultLang.value
  document.querySelectorAll('.bilingual-pair:not(.flipped-manual)').forEach((pair) => {
    const en = pair.querySelector('.bilingual-en') as HTMLElement
    const zh = pair.querySelector('.bilingual-zh') as HTMLElement
    if (!en || !zh) return
    if (lang === 'zh') {
      en.style.display = 'none'
      zh.style.display = ''
    } else {
      en.style.display = ''
      zh.style.display = 'none'
    }
  })
}
