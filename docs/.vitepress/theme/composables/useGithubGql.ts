import { useAuth } from './useAuth'

const REPO_OWNER = 'd2wstudy'
const REPO_NAME = 'rl-book-bilingual'
const GRAPHQL_URL = 'https://api.github.com/graphql'
const WORKER_URL = 'https://rl-book-auth.d2w.workers.dev'

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

/** Find a discussion by title in a category AND fetch its comments */
export async function findDiscussionWithComments(
  pagePath: string,
  categoryName: string,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  const { token } = useAuth()

  // Unauthenticated: use Worker proxy
  if (!token.value) {
    return fetchViaProxy(pagePath, categoryName)
  }

  // Authenticated: direct GraphQL
  const catId = await getCategoryId(categoryName)
  if (!catId) return null

  const data = await gql(`query($owner: String!, $name: String!, $categoryId: ID) {
    repository(owner: $owner, name: $name) {
      discussions(first: 50, categoryId: $categoryId, orderBy: {field: CREATED_AT, direction: DESC}) {
        nodes {
          id
          title
          comments(first: 100) {
            nodes {
              id body createdAt author { login avatarUrl }
              reactionGroups { content viewerHasReacted reactors { totalCount } }
              replies(first: 50) {
                nodes {
                  id body createdAt author { login avatarUrl }
                  reactionGroups { content viewerHasReacted reactors { totalCount } }
                }
              }
            }
          }
        }
      }
    }
  }`, { owner: REPO_OWNER, name: REPO_NAME, categoryId: catId })

  const nodes = data?.repository?.discussions?.nodes
  if (!nodes) return null
  const match = nodes.find((n: any) => n.title === pagePath)
  if (!match) return null
  return { discussionId: match.id, comments: match.comments.nodes }
}

/** Read-only proxy via Cloudflare Worker (for unauthenticated users) */
async function fetchViaProxy(
  pagePath: string,
  categoryName: string,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  try {
    const params = new URLSearchParams({ path: pagePath, category: categoryName })
    const resp = await fetch(`${WORKER_URL}/api/discussions?${params}`)
    if (!resp.ok) return null
    return await resp.json()
  } catch {
    console.warn('[GQL] Worker proxy request failed')
    return null
  }
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

/** Add a comment to a discussion */
export async function addDiscussionComment(discussionId: string, body: string) {
  return gql(`mutation($discussionId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, body: $body }) {
      comment { id }
    }
  }`, { discussionId, body })
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

/** Add a reply to a discussion comment */
export async function addDiscussionReply(discussionId: string, replyToId: string, body: string) {
  return gql(`mutation($discussionId: ID!, $replyToId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, replyToId: $replyToId, body: $body }) {
      comment { id }
    }
  }`, { discussionId, replyToId, body })
}

/** Shared GraphQL helper (requires auth) */
export async function gql(query: string, variables: Record<string, any>) {
  const { token } = useAuth()
  const t = token.value
  if (!t) return null
  const resp = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${t}`,
    },
    body: JSON.stringify({ query, variables }),
  })
  if (resp.status === 403 || resp.status === 429) {
    console.warn('[GQL] GitHub API rate limit exceeded')
    return null
  }
  const json = await resp.json()
  if (json.errors) {
    console.warn('[GQL] GraphQL errors:', json.errors)
  }
  return json.data
}
