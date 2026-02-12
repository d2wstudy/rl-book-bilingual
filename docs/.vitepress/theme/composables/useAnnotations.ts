import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import { findDiscussionWithComments, createDiscussion, addDiscussionComment } from './useGithubGql'

const CATEGORY_NAME = 'Notes'

export interface Annotation {
  id: string
  paragraphId: string
  startOffset: number
  endOffset: number
  selectedText: string
  /** Up to 32 chars before the selection (TextQuoteSelector context) */
  prefix: string
  /** Up to 32 chars after the selection (TextQuoteSelector context) */
  suffix: string
  note: string
  author: string
  authorAvatar: string
  createdAt: string
}

const annotations = ref<Map<string, Annotation[]>>(new Map())
const loaded = ref(false)

export function useAnnotations() {
  const { token } = useAuth()

  async function loadAnnotations(pagePath: string) {
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      const map = new Map<string, Annotation[]>()
      if (result) {
        for (const c of result.comments) {
          try {
            const data = JSON.parse(c.body)
            if (data.type !== 'annotation') continue
            const anno: Annotation = {
              id: c.id,
              paragraphId: data.paragraphId,
              startOffset: data.startOffset,
              endOffset: data.endOffset,
              selectedText: data.selectedText,
              prefix: data.prefix ?? '',
              suffix: data.suffix ?? '',
              note: data.note,
              author: c.author.login,
              authorAvatar: c.author.avatarUrl,
              createdAt: c.createdAt,
            }
            const list = map.get(anno.paragraphId) || []
            list.push(anno)
            map.set(anno.paragraphId, list)
          } catch {
            // Skip non-annotation comments
          }
        }
      }
      annotations.value = map
    } finally {
      loaded.value = true
    }
  }

  async function addAnnotation(
    pagePath: string,
    paragraphId: string,
    startOffset: number,
    endOffset: number,
    selectedText: string,
    note: string,
    prefix: string = '',
    suffix: string = '',
  ) {
    if (!token.value) return

    let result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
    let discussionId = result?.discussionId
    if (!discussionId) {
      discussionId = await createDiscussion(pagePath, CATEGORY_NAME, `读者笔记：${pagePath}`)
    }
    if (!discussionId) return

    const body = JSON.stringify({
      type: 'annotation',
      paragraphId,
      startOffset,
      endOffset,
      selectedText,
      prefix,
      suffix,
      note,
    })

    await addDiscussionComment(discussionId, body)
    await loadAnnotations(pagePath)
  }

  return {
    annotations: readonly(annotations),
    loaded: readonly(loaded),
    loadAnnotations,
    addAnnotation,
  }
}
