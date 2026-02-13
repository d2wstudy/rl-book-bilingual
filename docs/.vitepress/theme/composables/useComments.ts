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

const CATEGORY_NAME = 'Announcements'

// Re-export shared types for backward compatibility
export type { ReactionGroup }
export type Reply = ThreadReply

export interface Comment {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  replies: ThreadReply[]
  reactions: ReactionGroup[]
}

const comments = ref<Comment[]>([])
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

export function useComments() {
  const { token } = useAuth()

  async function loadComments(pagePath: string, force = false) {
    if (_loadPromise && _loadingPath === pagePath) {
      if (!force) return _loadPromise
      // Force: wait for in-flight request, then re-fetch with current auth state
      await _loadPromise
    }
    _loadingPath = pagePath
    _loadPromise = _doLoadComments(pagePath).finally(() => {
      _loadPromise = null
      _loadingPath = null
    })
    return _loadPromise
  }

  async function _doLoadComments(pagePath: string) {
    const knownId = _discussionId || storedDiscussionId(pagePath)
    comments.value = []
    loaded.value = false
    _discussionId = null
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME, knownId)
      if (result) {
        _discussionId = result.discussionId
        if (result.discussionId) storeDiscussionId(pagePath, result.discussionId)
        comments.value = result.comments.map((c: any) => ({
          id: c.id,
          body: c.body,
          author: c.author.login,
          authorAvatar: c.author.avatarUrl,
          createdAt: c.createdAt,
          replies: (c.replies?.nodes || []).map(mapReply),
          reactions: mapReactions(c.reactionGroups),
        }))
      }
    } finally {
      loaded.value = true
    }
  }

  async function addComment(pagePath: string, body: string) {
    if (!token.value) return
    let discussionId = _discussionId
    if (!discussionId) {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      discussionId = result?.discussionId ?? null
    }
    if (!discussionId) {
      discussionId = await createDiscussion(pagePath, CATEGORY_NAME, `章节讨论：${pagePath}`)
      if (discussionId) storeDiscussionId(pagePath, discussionId)
    }
    if (!discussionId) return
    _discussionId = discussionId

    const newComment = await addDiscussionComment(discussionId, body)
    if (newComment) {
      // Append to local state directly — no re-fetch needed
      comments.value = [...comments.value, {
        id: newComment.id,
        body: newComment.body,
        author: newComment.author.login,
        authorAvatar: newComment.author.avatarUrl,
        createdAt: newComment.createdAt,
        replies: [],
        reactions: mapReactions(newComment.reactionGroups),
      }]
    }
    await purgeWorkerCache(pagePath, CATEGORY_NAME, false, undefined, discussionId)
  }

  async function replyToComment(pagePath: string, commentId: string, body: string) {
    if (!token.value) return
    let discussionId = _discussionId
    if (!discussionId) {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      discussionId = result?.discussionId ?? null
    }
    if (!discussionId) return

    const newReply = await addDiscussionReply(discussionId, commentId, body)
    if (newReply) {
      // Append reply to the parent comment locally — no re-fetch needed
      const parent = comments.value.find(c => c.id === commentId)
      if (parent) {
        parent.replies = [...parent.replies, mapReply(newReply)]
      }
    }
    await purgeWorkerCache(pagePath, CATEGORY_NAME, false, undefined, discussionId)
  }

  const toggleReaction = createReactionToggler((subjectId) => {
    for (const c of comments.value) {
      if (c.id === subjectId) return c
      for (const r of c.replies) {
        if (r.id === subjectId) return r
      }
    }
    return null
  })

  return {
    comments: readonly(comments),
    loaded: readonly(loaded),
    loadComments,
    addComment,
    replyToComment,
    toggleReaction,
  }
}
