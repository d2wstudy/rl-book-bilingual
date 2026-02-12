import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import { findDiscussionWithComments, createDiscussion, addDiscussionComment } from './useGithubGql'

const CATEGORY_NAME = 'Announcements'

export interface Comment {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
}

const comments = ref<Comment[]>([])
const loaded = ref(false)

export function useComments() {
  const { token } = useAuth()

  async function loadComments(pagePath: string) {
    comments.value = []
    loaded.value = false
    try {
      const result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
      if (result) {
        comments.value = result.comments.map((c: any) => ({
          id: c.id,
          body: c.body,
          author: c.author.login,
          authorAvatar: c.author.avatarUrl,
          createdAt: c.createdAt,
        }))
      }
    } finally {
      loaded.value = true
    }
  }

  async function addComment(pagePath: string, body: string) {
    if (!token.value) return
    let result = await findDiscussionWithComments(pagePath, CATEGORY_NAME)
    let discussionId = result?.discussionId
    if (!discussionId) {
      discussionId = await createDiscussion(pagePath, CATEGORY_NAME, `章节讨论：${pagePath}`)
    }
    if (!discussionId) return

    await addDiscussionComment(discussionId, body)
    await loadComments(pagePath)
  }

  return {
    comments: readonly(comments),
    loaded: readonly(loaded),
    loadComments,
    addComment,
  }
}
