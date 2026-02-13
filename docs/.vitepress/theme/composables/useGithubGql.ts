import { useAuth } from './useAuth'

const REPO_OWNER = 'd2wstudy'
const REPO_NAME = 'rl-book-bilingual'
const GRAPHQL_URL = 'https://api.github.com/graphql'
const WORKER_URL = 'https://rl-book-auth.d2w.workers.dev'

// --- Deduplication layer ---
const _inflightDiscussions = new Map<string, Promise<any>>()

// Shared category ID cache (Promise-based to deduplicate concurrent calls)
let _categoriesPromise: Promise<Map<string, string>> | null = null

async function fetchCategories(): Promise<Map<string, string>> {
  const data = await gql(`query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      discussionCategories(first: 20) { nodes { id name } }
    }
  }`, { owner: REPO_OWNER, name: REPO_NAME })
  const map = new Map<string, string>()
  for (const c of data?.repository?.discussionCategories?.nodes || []) {
    map.set(c.name, c.id)
  }
  return map
}

export async function getCategoryId(categoryName: string): Promise<string | null> {
  const { token } = useAuth()
  if (!token.value) return null

  if (!_categoriesPromise) {
    _categoriesPromise = fetchCategories()
  }
  const categories = await _categoriesPromise
  return categories.get(categoryName) || null
}

/** Find a discussion by title in a category AND fetch its comments.
 *  All requests go through Worker proxy (shared cache across users).
 *  Authenticated users pass their token; unauthenticated fall back to PAT. */
export async function findDiscussionWithComments(
  pagePath: string,
  categoryName: string,
  knownDiscussionId?: string | null,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  const key = `${categoryName}::${pagePath}`

  // Deduplicate concurrent in-flight requests for the same category+page
  const inflight = _inflightDiscussions.get(key)
  if (inflight) return inflight

  const promise = fetchViaProxy(pagePath, categoryName, knownDiscussionId)
  _inflightDiscussions.set(key, promise)
  promise.finally(() => _inflightDiscussions.delete(key))
  return promise
}

/** Fetch discussions via Worker proxy (shared cache, uses user token if available) */
async function fetchViaProxy(
  pagePath: string,
  categoryName: string,
  knownDiscussionId?: string | null,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  try {
    const params = new URLSearchParams({ path: pagePath, category: categoryName })
    if (knownDiscussionId) params.set('id', knownDiscussionId)

    const headers: Record<string, string> = {}
    const { token } = useAuth()
    if (token.value) {
      headers['Authorization'] = `Bearer ${token.value}`
    }

    const resp = await fetch(`${WORKER_URL}/api/discussions?${params}`, { headers })
    if (!resp.ok) return null
    return await resp.json()
  } catch {
    console.warn('[GQL] Worker proxy request failed')
    return null
  }
}

/** Purge Worker cache for a specific page + category (call after write operations) */
export function purgeWorkerCache(pagePath: string, categoryName: string) {
  const params = new URLSearchParams({ path: pagePath, category: categoryName })
  fetch(`${WORKER_URL}/api/cache/purge?${params}`, { method: 'POST' }).catch(() => {})
}

/** Create a new discussion in a category */
export async function createDiscussion(pagePath: string, categoryName: string, bodyText: string): Promise<string | null> {
  const categoryId = await getCategoryId(categoryName)
  if (!categoryId) return null

  const repoData = await gql(`query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) { id }
  }`, { owner: REPO_OWNER, name: REPO_NAME })
  const repoId = repoData?.repository?.id
  if (!repoId) return null

  const result = await gql(`mutation($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
    createDiscussion(input: { repositoryId: $repoId, categoryId: $categoryId, title: $title, body: $body }) {
      discussion { id }
    }
  }`, { repoId, categoryId, title: pagePath, body: bodyText })
  return result?.createDiscussion?.discussion?.id || null
}

/** Add a comment to a discussion — returns the new comment data */
export async function addDiscussionComment(discussionId: string, body: string) {
  const data = await gql(`mutation($discussionId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, body: $body }) {
      comment {
        id body createdAt
        author { login avatarUrl }
        reactionGroups { content viewerHasReacted reactors { totalCount } }
      }
    }
  }`, { discussionId, body })
  return data?.addDiscussionComment?.comment || null
}

/** Add a reaction to a subject (comment or reply) */
export async function addReaction(subjectId: string, content: string) {
  return gql(`mutation($subjectId: ID!, $content: ReactionContent!) {
    addReaction(input: { subjectId: $subjectId, content: $content }) {
      reaction { content }
    }
  }`, { subjectId, content })
}

/** Remove a reaction from a subject */
export async function removeReaction(subjectId: string, content: string) {
  return gql(`mutation($subjectId: ID!, $content: ReactionContent!) {
    removeReaction(input: { subjectId: $subjectId, content: $content }) {
      reaction { content }
    }
  }`, { subjectId, content })
}

/** Add a reply to a discussion comment — returns the new reply data */
export async function addDiscussionReply(discussionId: string, replyToId: string, body: string) {
  const data = await gql(`mutation($discussionId: ID!, $replyToId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, replyToId: $replyToId, body: $body }) {
      comment {
        id body createdAt
        author { login avatarUrl }
        reactionGroups { content viewerHasReacted reactors { totalCount } }
      }
    }
  }`, { discussionId, replyToId, body })
  return data?.addDiscussionComment?.comment || null
}

/** Shared GraphQL helper — used for mutations only (reads go through Worker proxy) */
export async function gql(query: string, variables: Record<string, any>) {
  const { token } = useAuth()
  const t = token.value
  if (!t) return null

  // Dev: inject rateLimit field to monitor API point consumption
  const actualQuery = import.meta.env.DEV
    ? query.replace(/\{/, '{ rateLimit { cost remaining resetAt }')
    : query

  const resp = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${t}`,
    },
    body: JSON.stringify({ query: actualQuery, variables }),
  })
  if (resp.status === 403 || resp.status === 429) {
    console.warn('[GQL] GitHub API rate limit exceeded')
    return null
  }
  const json = await resp.json()
  if (json.errors) {
    console.warn('[GQL] GraphQL errors:', json.errors)
  }
  if (import.meta.env.DEV && json.data?.rateLimit) {
    const rl = json.data.rateLimit
    console.log(`[GQL] cost=${rl.cost} remaining=${rl.remaining} reset=${rl.resetAt}`)
  }
  return json.data
}
