import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { onMounted, watch, h } from 'vue'
import { useRoute } from 'vitepress'
import { useLang, applyDefaultLang } from './composables/useLang'
import { useAuth } from './composables/useAuth'
import LanguageToggle from './components/LanguageToggle.vue'
import LoginButton from './components/LoginButton.vue'
import Anno from './components/Anno.vue'
import AnnotationLayer from './components/AnnotationLayer.vue'
import ChapterComments from './components/ChapterComments.vue'
import './style.css'

let pairCounter = 0

export default {
  extends: DefaultTheme,

  enhanceApp({ app }) {
    app.component('LanguageToggle', LanguageToggle)
    app.component('LoginButton', LoginButton)
    app.component('Anno', Anno)
    app.component('AnnotationLayer', AnnotationLayer)
    app.component('ChapterComments', ChapterComments)
  },

  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => [h(LanguageToggle), h(LoginButton)],
      'doc-after': () => h(AnnotationLayer),
    })
  },

  setup() {
    const { defaultLang, initLang } = useLang()
    const { init: initAuth } = useAuth()
    const route = useRoute()

    onMounted(() => {
      initLang()
      initAuth()
      pairBilingualBlocks()
      applyDefaultLang()
    })

    watch(() => route.path, () => {
      pairCounter = 0
      setTimeout(() => {
        pairBilingualBlocks()
        applyDefaultLang()
      }, 100)
    })

    watch(defaultLang, () => {
      document.querySelectorAll('.bilingual-pair.flipped-manual').forEach((el) => {
        el.classList.remove('flipped-manual')
      })
      applyDefaultLang()
    })
  },
} satisfies Theme

/**
 * Find consecutive .bilingual-en + .bilingual-zh divs,
 * wrap them in a .bilingual-pair container with a flip button and a stable ID.
 */
function pairBilingualBlocks() {
  if (typeof document === 'undefined') return
  if (document.querySelector('.bilingual-pair')) return

  const content = document.querySelector('.vp-doc')
  if (!content) return

  const blocks = Array.from(content.querySelectorAll('.bilingual-en, .bilingual-zh'))

  for (let i = 0; i < blocks.length - 1; i++) {
    const en = blocks[i]
    const zh = blocks[i + 1]
    if (!en.classList.contains('bilingual-en') || !zh.classList.contains('bilingual-zh')) continue

    // Find nearest heading for a stable paragraph ID
    const heading = findPrecedingHeading(en)
    const headingSlug = heading ? heading.id || heading.textContent?.trim().slice(0, 20) : 'top'
    const pairId = `${headingSlug}-p${pairCounter++}`

    const pair = document.createElement('div')
    pair.className = 'bilingual-pair'
    pair.setAttribute('data-pair-id', pairId)

    const btn = document.createElement('button')
    btn.className = 'flip-btn'
    btn.title = '切换中/英文'
    btn.textContent = '译'
    btn.addEventListener('click', () => {
      pair.classList.add('flipped-manual')
      const enEl = pair.querySelector('.bilingual-en') as HTMLElement
      const zhEl = pair.querySelector('.bilingual-zh') as HTMLElement
      if (!enEl || !zhEl) return
      const enVisible = enEl.style.display !== 'none'
      enEl.style.display = enVisible ? 'none' : ''
      zhEl.style.display = enVisible ? '' : 'none'
      btn.textContent = enVisible ? '译' : '原'
    })

    en.parentNode!.insertBefore(pair, en)
    pair.appendChild(en)
    pair.appendChild(zh)
    pair.appendChild(btn)

    i++
  }
}

function findPrecedingHeading(el: Element): HTMLElement | null {
  let node = el.previousElementSibling
  while (node) {
    if (/^H[1-6]$/.test(node.tagName)) return node as HTMLElement
    node = node.previousElementSibling
  }
  return null
}
