import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import {
  findDiscussionWithComments, createDiscussion,
  addDiscussionComment, addDiscussionReply,
} from './useGithubGql'
import {
  type ReactionGroup, type ThreadReply,
  mapReactions, mapReply, createReactionToggler,
} from './useDiscussionThread'

const CATEGORY_NAME = 'Notes'

export interface AnnotationAnchor {
  paragraphId: string
  startOffset: number
  endOffset: number
  selectedText: string
  prefix: string
  suffix: string
}

export interface AnnotationThread {
  id: string
  anchor: AnnotationAnchor
  note: string
  author: string
  authorAvatar: string
  createdAt: string
  replies: ThreadReply[]
  reactions: ReactionGroup[]
}

const annotations = ref<Map<string, AnnotationThread[]>>(new Map())
const loaded = ref(false)
let _discussionId: string | null = null

export function useAnnotations() {
  const { token } = useAuth()

  async function loadAnnotations(pagePath: string) {
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      const map = new Map<string, AnnotationThread[]>()
      _discussionId = null
      if (result) {
        _discussionId = result.discussionId
        for (const c of result.comments) {
          try {
            const data = JSON.parse(c.body)
            if (data.type !== 'annotation') continue
            const thread: AnnotationThread = {
              id: c.id,
              anchor: {
                paragraphId: data.paragraphId,
                startOffset: data.startOffset,
                endOffset: data.endOffset,
                selectedText: data.selectedText,
                prefix: data.prefix ?? '',
                suffix: data.suffix ?? '',
              },
              note: data.note,
              author: c.author.login,
              authorAvatar: c.author.avatarUrl,
              createdAt: c.createdAt,
              replies: (c.replies?.nodes || []).map(mapReply),
              reactions: mapReactions(c.reactionGroups),
            }
            const list = map.get(thread.anchor.paragraphId) || []
            list.push(thread)
            map.set(thread.anchor.paragraphId, list)
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

    let discussionId = _discussionId
    if (!discussionId) {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      discussionId = result?.discussionId ?? null
    }
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

  async function replyToAnnotation(pagePath: string, threadId: string, body: string) {
    if (!token.value || !_discussionId) return
    await addDiscussionReply(_discussionId, threadId, body)
    await loadAnnotations(pagePath)
  }

  const toggleReaction = createReactionToggler((subjectId) => {
    for (const threads of annotations.value.values()) {
      for (const t of threads) {
        if (t.id === subjectId) return t
        for (const r of t.replies) {
          if (r.id === subjectId) return r
        }
      }
    }
    return null
  })

  return {
    annotations: readonly(annotations),
    loaded: readonly(loaded),
    loadAnnotations,
    addAnnotation,
    replyToAnnotation,
    toggleReaction,
  }
}
