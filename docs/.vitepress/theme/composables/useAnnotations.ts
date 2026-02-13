import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import {
  findDiscussionWithComments, createDiscussion,
  addDiscussionComment, addDiscussionReply,
  purgeWorkerCache,
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
  /** For cross-paragraph annotations: one anchor per paragraph */
  segments?: AnnotationAnchor[]
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
let _loadPromise: Promise<void> | null = null
let _loadingPath: string | null = null

function storedDiscussionId(pagePath: string): string | null {
  try { return sessionStorage.getItem(`disc::${CATEGORY_NAME}::${pagePath}`) } catch { return null }
}
function storeDiscussionId(pagePath: string, id: string) {
  try { sessionStorage.setItem(`disc::${CATEGORY_NAME}::${pagePath}`, id) } catch {}
}

export function useAnnotations() {
  const { token } = useAuth()

  async function loadAnnotations(pagePath: string, force = false) {
    if (_loadPromise && _loadingPath === pagePath) {
      if (!force) return _loadPromise
      await _loadPromise
    }
    _loadingPath = pagePath
    _loadPromise = _doLoadAnnotations(pagePath).finally(() => {
      _loadPromise = null
      _loadingPath = null
    })
    return _loadPromise
  }

  async function _doLoadAnnotations(pagePath: string) {
    const knownId = _discussionId || storedDiscussionId(pagePath)
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME, knownId)
      const map = new Map<string, AnnotationThread[]>()
      _discussionId = null
      if (result) {
        _discussionId = result.discussionId
        if (result.discussionId) storeDiscussionId(pagePath, result.discussionId)
        for (const c of result.comments) {
          try {
            const data = JSON.parse(c.body)
            if (data.type !== 'annotation') continue
            const segments: AnnotationAnchor[] | undefined = Array.isArray(data.segments)
              ? data.segments.map((s: any) => ({
                  paragraphId: s.paragraphId,
                  startOffset: s.startOffset,
                  endOffset: s.endOffset,
                  selectedText: s.selectedText,
                  prefix: s.prefix ?? '',
                  suffix: s.suffix ?? '',
                }))
              : undefined
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
              segments,
              note: data.note,
              author: c.author.login,
              authorAvatar: c.author.avatarUrl,
              createdAt: c.createdAt,
              replies: (c.replies?.nodes || []).map(mapReply),
              reactions: mapReactions(c.reactionGroups),
            }
            // Index under all segment paragraphIds (or just the primary one)
            const pids = segments
              ? [...new Set(segments.map(s => s.paragraphId))]
              : [thread.anchor.paragraphId]
            for (const pid of pids) {
              const list = map.get(pid) || []
              list.push(thread)
              map.set(pid, list)
            }
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
    segments?: AnnotationAnchor[],
  ) {
    if (!token.value) return

    let discussionId = _discussionId
    if (!discussionId) {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      discussionId = result?.discussionId ?? null
    }
    if (!discussionId) {
      discussionId = await createDiscussion(pagePath, CATEGORY_NAME, `读者笔记：${pagePath}`)
      if (discussionId) storeDiscussionId(pagePath, discussionId)
    }
    if (!discussionId) return
    _discussionId = discussionId

    const body = JSON.stringify({
      type: 'annotation',
      paragraphId,
      startOffset,
      endOffset,
      selectedText,
      prefix,
      suffix,
      note,
      ...(segments ? { segments } : {}),
    })

    const newComment = await addDiscussionComment(discussionId, body)
    if (newComment) {
      // Append to local state directly — no re-fetch needed
      const thread: AnnotationThread = {
        id: newComment.id,
        anchor: { paragraphId, startOffset, endOffset, selectedText, prefix, suffix },
        segments,
        note,
        author: newComment.author.login,
        authorAvatar: newComment.author.avatarUrl,
        createdAt: newComment.createdAt,
        replies: [],
        reactions: mapReactions(newComment.reactionGroups),
      }
      const map = new Map(annotations.value)
      const pids = segments
        ? [...new Set(segments.map(s => s.paragraphId))]
        : [paragraphId]
      for (const pid of pids) {
        const list = [...(map.get(pid) || []), thread]
        map.set(pid, list)
      }
      annotations.value = map
    }
    await purgeWorkerCache(pagePath, CATEGORY_NAME, false, undefined, discussionId)
  }

  async function replyToAnnotation(pagePath: string, threadId: string, body: string) {
    if (!token.value || !_discussionId) return
    const newReply = await addDiscussionReply(_discussionId, threadId, body)
    if (newReply) {
      // Append reply to the parent thread locally — no re-fetch needed
      for (const threads of annotations.value.values()) {
        const parent = threads.find(t => t.id === threadId)
        if (parent) {
          parent.replies = [...parent.replies, mapReply(newReply)]
          break
        }
      }
    }
    await purgeWorkerCache(pagePath, CATEGORY_NAME, false, undefined, _discussionId)
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
