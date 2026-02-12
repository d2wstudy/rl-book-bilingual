import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import { findDiscussionWithComments, createDiscussion, addDiscussionComment, addDiscussionReply } from './useGithubGql'

const CATEGORY_NAME = 'Announcements'

export interface Reply {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
}

export interface Comment {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  replies: Reply[]
}

const comments = ref<Comment[]>([])
const loaded = ref(false)
let _discussionId: string | null = null

function mapReply(r: any): Reply {
  return {
    id: r.id,
    body: r.body,
    author: r.author.login,
    authorAvatar: r.author.avatarUrl,
    createdAt: r.createdAt,
  }
}

export function useComments() {
  const { token } = useAuth()

  async function loadComments(pagePath: string) {
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
    await loadComments(pagePath)
  }

  return {
    comments: readonly(comments),
    loaded: readonly(loaded),
    loadComments,
    addComment,
    replyToComment,
  }
}
