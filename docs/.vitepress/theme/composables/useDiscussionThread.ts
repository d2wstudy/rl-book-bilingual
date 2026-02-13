import { useAuth } from './useAuth'
import { addReaction, removeReaction } from './useGithubGql'

// ---- Canonical reaction order (matches picker UI) ----

const REACTION_ORDER = ['THUMBS_UP', 'THUMBS_DOWN', 'HEART', 'LAUGH', 'HOORAY', 'ROCKET', 'EYES']
const REACTION_INDEX = Object.fromEntries(REACTION_ORDER.map((k, i) => [k, i]))

function sortReactions(reactions: ReactionGroup[]): ReactionGroup[] {
  return reactions.sort((a, b) => (REACTION_INDEX[a.content] ?? 99) - (REACTION_INDEX[b.content] ?? 99))
}

// ---- Shared types ----

export interface ReactionGroup {
  content: string
  count: number
  viewerHasReacted: boolean
}

export interface ThreadReply {
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  reactions: ReactionGroup[]
}

// ---- Shared mapping functions ----

export function mapReactions(groups: any[]): ReactionGroup[] {
  if (!groups) return []
  return sortReactions(
    groups
      .map((g: any) => ({
        content: g.content,
        count: g.reactors?.totalCount ?? g.users?.totalCount ?? 0,
        viewerHasReacted: g.viewerHasReacted ?? false,
      }))
      .filter((g: ReactionGroup) => g.count > 0 || g.viewerHasReacted)
  )
}

export function mapReply(r: any): ThreadReply {
  return {
    id: r.id,
    body: r.body,
    author: r.author.login,
    authorAvatar: r.author.avatarUrl,
    createdAt: r.createdAt,
    reactions: mapReactions(r.reactionGroups),
  }
}

// ---- Shared reaction toggler ----

export function createReactionToggler(
  findTarget: (subjectId: string) => { reactions: ReactionGroup[] } | null,
) {
  const { token } = useAuth()

  return async function toggleReaction(subjectId: string, content: string): Promise<{ delta: number } | undefined> {
    if (!token.value) return
    const target = findTarget(subjectId)
    if (!target) return

    const existing = target.reactions.find(r => r.content === content)
    if (existing?.viewerHasReacted) {
      existing.count--
      existing.viewerHasReacted = false
      if (existing.count <= 0) {
        target.reactions.splice(target.reactions.indexOf(existing), 1)
      }
      await removeReaction(subjectId, content)
      return { delta: -1 }
    } else if (existing) {
      existing.count++
      existing.viewerHasReacted = true
      await addReaction(subjectId, content)
      return { delta: 1 }
    } else {
      target.reactions.push({ content, count: 1, viewerHasReacted: true })
      sortReactions(target.reactions)
      await addReaction(subjectId, content)
      return { delta: 1 }
    }
  }
}

// ---- Shared time formatting ----

export function formatRelativeTime(iso: string): string {
  const d = new Date(iso)
  const now = Date.now()
  const diff = now - d.getTime()
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  if (diff < 2592000000) return `${Math.floor(diff / 86400000)} 天前`
  return d.toLocaleDateString('zh-CN')
}
