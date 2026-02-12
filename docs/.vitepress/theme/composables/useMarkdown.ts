import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const ALLOWED_TAGS = [
  'p', 'br', 'strong', 'em', 'del', 'code', 'pre',
  'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
  'h3', 'h4', 'hr', 'img', 'table', 'thead', 'tbody',
  'tr', 'th', 'td',
]

const ALLOWED_ATTR = ['href', 'src', 'alt', 'title', 'target']

export function useMarkdown() {
  function renderMarkdown(src: string): string {
    if (!src) return ''
    const raw = marked.parse(src) as string
    return DOMPurify.sanitize(raw, { ALLOWED_TAGS, ALLOWED_ATTR })
  }

  return { renderMarkdown }
}
