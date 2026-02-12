import { ref, readonly } from 'vue'
import { useAuth } from './useAuth'
import {
  findDiscussionWithComments, createDiscussion, addDiscussionComment,
  addDiscussionReply, addReaction, removeReaction,
} from './useGithubGql'

const CATEGORY_NAME = 'Announcements'

export interface ReactionGroup {
  content: string
  count: number
  viewerHasReacted: boolean
}

export interface Reply {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  reactions: ReactionGroup[]
}

export interface Comment {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  replies: Reply[]
  reactions: ReactionGroup[]
}

const comments = ref<Comment[]>([])
const loaded = ref(false)
let _discussionId: string | null = null

function mapReactions(groups: any[]): ReactionGroup[] {
  if (!groups) return []
  return groups
    .map((g: any) => ({
      content: g.content,
      count: g.reactors?.totalCount ?? g.users?.totalCount ?? 0,
      viewerHasReacted: g.viewerHasReacted ?? false,
    }))
    .filter((g: ReactionGroup) => g.count > 0 || g.viewerHasReacted)
}

function mapReply(r: any): Reply {
  return {
    id: r.id,
    body: r.body,
    author: r.author.login,
    authorAvatar: r.author.avatarUrl,
    createdAt: r.createdAt,
    reactions: mapReactions(r.reactionGroups),
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

  async function toggleReaction(subjectId: string, content: string) {
    if (!token.value) return

    // Find the reaction in comments or replies and toggle optimistically
    const target = findReactionTarget(subjectId)
    if (!target) return

    const existing = target.reactions.find(r => r.content === content)
    if (existing?.viewerHasReacted) {
      existing.count--
      existing.viewerHasReacted = false
      if (existing.count <= 0) {
        target.reactions.splice(target.reactions.indexOf(existing), 1)
      }
      await removeReaction(subjectId, content)
    } else if (existing) {
      existing.count++
      existing.viewerHasReacted = true
      await addReaction(subjectId, content)
    } else {
      target.reactions.push({ content, count: 1, viewerHasReacted: true })
      await addReaction(subjectId, content)
    }
  }

  function findReactionTarget(subjectId: string): { reactions: ReactionGroup[] } | null {
    for (const c of comments.value) {
      if (c.id === subjectId) return c
      for (const r of c.replies) {
        if (r.id === subjectId) return r
      }
    }
    return null
  }

  return {
    comments: readonly(comments),
    loaded: readonly(loaded),
    loadComments,
    addComment,
    replyToComment,
    toggleReaction,
  }
}
