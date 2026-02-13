import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import {
  findDiscussionWithComments, createDiscussion,
  addDiscussionComment, addDiscussionReply,
  invalidateDiscussionCache,
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

export function useComments() {
  const { token } = useAuth()

  async function loadComments(pagePath: string) {
    // Deduplicate: if already loading the same path, reuse the promise
    if (_loadPromise && _loadingPath === pagePath) return _loadPromise
    _loadingPath = pagePath
    _loadPromise = _doLoadComments(pagePath).finally(() => {
      _loadPromise = null
      _loadingPath = null
    })
    return _loadPromise
  }

  async function _doLoadComments(pagePath: string) {
    comments.value = []
    loaded.value = false
    _discussionId = null
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      if (result) {
        _discussionId = result.discussionId
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
    }
    if (!discussionId) return

    await addDiscussionComment(discussionId, body)
    invalidateDiscussionCache(pagePath, CATEGORY_NAME)
    await loadComments(pagePath)
  }

  async function replyToComment(pagePath: string, commentId: string, body: string) {
    if (!token.value) return
    let discussionId = _discussionId
    if (!discussionId) {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      discussionId = result?.discussionId ?? null
    }
    if (!discussionId) return

    await addDiscussionReply(discussionId, commentId, body)
    invalidateDiscussionCache(pagePath, CATEGORY_NAME)
    await loadComments(pagePath)
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
